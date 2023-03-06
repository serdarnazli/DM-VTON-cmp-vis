import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from models.afwm import AFWM, TVLoss 
from models.afwm_pb import AFWM as PBAFWM 
from models.networks import ResUnetGenerator, VGGLoss
from options.train_options import TrainOptions
from utils.utils import load_checkpoint_parallel, load_checkpoint_part_parallel, save_checkpoint


opt = TrainOptions().parse()
path = 'runs/' + opt.name
os.makedirs(path, exist_ok=True)
os.makedirs(opt.checkpoints_dir,exist_ok=True)


def CreateDataset(opt):
    #training with augumentation
    #from data.aligned_dataset import AlignedDataset_aug
    #dataset = AlignedDataset_aug()
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset


os.makedirs('sample', exist_ok=True)
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

torch.cuda.set_device(opt.gpu_ids[0])
torch.distributed.init_process_group(
    'nccl',
    init_method='env://'
)
device = torch.device(f'cuda:{opt.gpu_ids[0]}')

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)
train_sampler = DistributedSampler(train_data)
train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=False,
                          num_workers=16, pin_memory=True, sampler=train_sampler)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

PF_warp_model = AFWM(opt, 3)
PF_warp_model.train()
PF_warp_model.to(device)
#load_checkpoint_part_parallel(PF_warp_model, opt.PBAFN_warp_checkpoint, device)

PB_warp_model = PBAFWM(opt, 45)
PB_warp_model.eval()
PB_warp_model.to(device)
load_checkpoint_parallel(PB_warp_model, opt.PBAFN_warp_checkpoint, device)

PB_gen_model = ResUnetGenerator(8, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
PB_gen_model.eval()
PB_gen_model.to(device)
load_checkpoint_parallel(PB_gen_model, opt.PBAFN_gen_checkpoint, device)

PF_warp_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(PF_warp_model).to(device)

if opt.isTrain and len(opt.gpu_ids):
    PF_warp_model = torch.nn.parallel.DistributedDataParallel(PF_warp_model, device_ids=[opt.gpu_ids[0]])
    PB_warp_model = torch.nn.parallel.DistributedDataParallel(PB_warp_model, device_ids=[opt.gpu_ids[0]])
    PB_gen_model = torch.nn.parallel.DistributedDataParallel(PB_gen_model, device_ids=[opt.gpu_ids[0]])

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
criterionL2 = nn.MSELoss('sum')

# optimizer
params = [p for p in PF_warp_model.parameters()]
optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

params_part = []
for name, param in PF_warp_model.named_parameters():
    if 'cond_' in name or 'aflow_net.netRefine' in name and 'aflow_net.cond_style' not in name:
        params_part.append(param)
optimizer_part = torch.optim.Adam(params_part, lr=opt.lr, betas=(opt.beta1, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter

if opt.local_rank == 0:
    writer = SummaryWriter(path)

step = 0
step_per_batch = dataset_size

all_steps = dataset_size * (opt.niter + opt.niter_decay + 1 - start_epoch)
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    train_loss = 0
    train_fea_loss = 0
    train_flow_loss = 0

    train_sampler.set_epoch(epoch)

    for i, data in enumerate(train_loader):

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1
        save_fake = True

        t_mask = torch.FloatTensor((data['label'].cpu().numpy() == 7).astype(np.float64))
        data['label'] = data['label'] * (1 - t_mask) + t_mask * 4
        edge = data['edge']
        pre_clothes_edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int64))
        clothes = data['color']
        clothes = clothes * pre_clothes_edge
        edge_un = data['edge_un']
        pre_clothes_edge_un = torch.FloatTensor((edge_un.detach().numpy() > 0.5).astype(np.int64))
        clothes_un = data['color_un']
        clothes_un = clothes_un * pre_clothes_edge_un
        person_clothes_edge = torch.FloatTensor((data['label'].cpu().numpy() == 4).astype(np.int64))
        real_image = data['image']
        person_clothes = real_image * person_clothes_edge
        pose = data['pose']
        size = data['label'].size()
        oneHot_size1 = (size[0], 25, size[2], size[3])
        densepose = torch.cuda.FloatTensor(torch.Size(oneHot_size1)).zero_()
        densepose = densepose.scatter_(1, data['densepose'].data.long().to(device), 1.0)
        densepose_fore = data['densepose'] / 24
        face_mask = torch.FloatTensor((data['label'].cpu().numpy() == 1).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 12).astype(np.int64))
        other_clothes_mask = torch.FloatTensor((data['label'].cpu().numpy() == 5).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 6).astype(np.int64)) \
                             + torch.FloatTensor((data['label'].cpu().numpy() == 8).astype(np.int64)) + torch.FloatTensor((data['label'].cpu().numpy() == 9).astype(np.int64)) \
                             + torch.FloatTensor((data['label'].cpu().numpy() == 10).astype(np.int64))
        face_img = face_mask * real_image
        other_clothes_img = other_clothes_mask * real_image
        preserve_mask = torch.cat([face_mask, other_clothes_mask], 1)

        concat_un = torch.cat([preserve_mask.to(device), densepose, pose.to(device)], 1)
        flow_out_un = PB_warp_model(concat_un.to(device), clothes_un.to(device), pre_clothes_edge_un.to(device))
        warped_cloth_un, last_flow_un, cond_un_all, flow_un_all, delta_list_un, x_all_un, x_edge_all_un, delta_x_all_un, delta_y_all_un = flow_out_un
        warped_prod_edge_un = F.grid_sample(pre_clothes_edge_un.to(device), last_flow_un.permute(0, 2, 3, 1),
                                            mode='bilinear', padding_mode='zeros', align_corners=opt.align_corners)

        flow_out_sup = PB_warp_model(concat_un.to(device), clothes.to(device), pre_clothes_edge.to(device))
        warped_cloth_sup, last_flow_sup, cond_sup_all, flow_sup_all, delta_list_sup, x_all_sup, x_edge_all_sup, delta_x_all_sup, delta_y_all_sup = flow_out_sup

        arm_mask = torch.FloatTensor((data['label'].cpu().numpy() == 11).astype(np.float64)) + torch.FloatTensor((data['label'].cpu().numpy() == 13).astype(np.float64))
        hand_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 3).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 4).astype(np.int64))
        dense_preserve_mask = torch.FloatTensor((data['densepose'].cpu().numpy() == 15).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 16).astype(np.int64)) \
                              + torch.FloatTensor((data['densepose'].cpu().numpy() == 17).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 18).astype(np.int64)) \
                              + torch.FloatTensor((data['densepose'].cpu().numpy() == 19).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 20).astype(np.int64)) \
                              + torch.FloatTensor((data['densepose'].cpu().numpy() == 21).astype(np.int64)) + torch.FloatTensor((data['densepose'].cpu().numpy() == 22))
        hand_img = (arm_mask * hand_mask) * real_image
        dense_preserve_mask = dense_preserve_mask.to(device) * (1 - warped_prod_edge_un)
        preserve_region = face_img + other_clothes_img + hand_img

        gen_inputs_un = torch.cat([preserve_region.to(device), warped_cloth_un, warped_prod_edge_un, dense_preserve_mask], 1)
        gen_outputs_un = PB_gen_model(gen_inputs_un)
        p_rendered_un, m_composite_un = torch.split(gen_outputs_un, [3, 1], 1)
        p_rendered_un = torch.tanh(p_rendered_un)
        m_composite_un = torch.sigmoid(m_composite_un)
        m_composite_un = m_composite_un * warped_prod_edge_un
        p_tryon_un = warped_cloth_un * m_composite_un + p_rendered_un * (1 - m_composite_un)

        flow_out = PF_warp_model(p_tryon_un.detach(), clothes.to(device), pre_clothes_edge.to(device))
        warped_cloth, last_flow, cond_all, flow_all, delta_list, x_all, x_edge_all, delta_x_all, delta_y_all = flow_out
        warped_prod_edge = x_edge_all[4]

        epsilon = 0.001
        loss_smooth = sum([TVLoss(x) for x in delta_list])
        loss_all = 0
        loss_fea_sup_all = 0
        loss_flow_sup_all = 0

        l1_loss_batch = torch.abs(warped_cloth_sup.detach() - person_clothes.to(device))
        l1_loss_batch = l1_loss_batch.reshape(-1, 3 * 256 * 192) # opt.batchSize
        l1_loss_batch = l1_loss_batch.sum(dim=1) / (3 * 256 * 192)
        l1_loss_batch_pred = torch.abs(warped_cloth.detach() - person_clothes.to(device))
        l1_loss_batch_pred = l1_loss_batch_pred.reshape(-1, 3 * 256 * 192) # opt.batchSize
        l1_loss_batch_pred = l1_loss_batch_pred.sum(dim=1) / (3 * 256 * 192)
        weight = (l1_loss_batch < l1_loss_batch_pred).float()
        num_all = len(np.where(weight.cpu().numpy() > 0)[0])
        if num_all == 0:
            num_all = 1

        for num in range(5):
            cur_person_clothes = F.interpolate(person_clothes, scale_factor=0.5 ** (4 - num), mode='bilinear')
            cur_person_clothes_edge = F.interpolate(person_clothes_edge, scale_factor=0.5 ** (4 - num), mode='bilinear')
            loss_l1 = criterionL1(x_all[num], cur_person_clothes.to(device))
            loss_vgg = criterionVGG(x_all[num], cur_person_clothes.to(device))
            loss_edge = criterionL1(x_edge_all[num], cur_person_clothes_edge.to(device))
            b, c, h, w = delta_x_all[num].shape
            loss_flow_x = (delta_x_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x) / (b * c * h * w)
            loss_flow_y = (delta_y_all[num].pow(2) + epsilon * epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y) / (b * c * h * w)
            loss_second_smooth = loss_flow_x + loss_flow_y
            b1, c1, h1, w1 = cond_all[num].shape
            weight_all = weight.reshape(-1, 1, 1, 1).repeat(1, 256, h1, w1)
            cond_sup_loss = ((cond_sup_all[num].detach() - cond_all[num]) ** 2 * weight_all).sum() / (256 * h1 * w1 * num_all)
            loss_fea_sup_all = loss_fea_sup_all + (5 - num) * 0.04 * cond_sup_loss
            loss_all = loss_all + (num + 1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num + 1) * 2 * loss_edge + (num + 1) * 6 * loss_second_smooth + (5 - num) * 0.04 * cond_sup_loss
            if num >= 2:
                b1, c1, h1, w1 = flow_all[num].shape
                weight_all = weight.reshape(-1, 1, 1).repeat(1, h1, w1)
                flow_sup_loss = (torch.norm(flow_sup_all[num].detach() - flow_all[num], p=2, dim=1) * weight_all).sum() / (h1 * w1 * num_all)
                loss_flow_sup_all = loss_flow_sup_all + (num + 1) * 1 * flow_sup_loss
                loss_all = loss_all + (num + 1) * 1 * flow_sup_loss

        loss_all = 0.01 * loss_smooth + loss_all

        # sum per device losses
        train_loss += loss_all
        train_fea_loss += loss_fea_sup_all
        train_flow_loss += loss_flow_sup_all

        if epoch < opt.niter:
            optimizer_part.zero_grad()
            loss_all.backward()
            optimizer_part.step()
        else:
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

        ############## Display results and errors ##########
        path = 'sample/' + opt.name
        os.makedirs(path, exist_ok=True)
        ### display output images
        if step % 1000 == 0:
            if opt.local_rank == 0:
                a = real_image.float().to(device)
                b = p_tryon_un.detach()
                c = clothes.to(device)
                d = person_clothes.to(device)
                e = torch.cat([person_clothes_edge.to(device), person_clothes_edge.to(device), person_clothes_edge.to(device)], 1)
                f = torch.cat([densepose_fore.to(device), densepose_fore.to(device), densepose_fore.to(device)], 1)
                g = warped_cloth
                h = torch.cat([warped_prod_edge, warped_prod_edge, warped_prod_edge], 1)
                combine = torch.cat([a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0]], 2).squeeze()
                cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
                writer.add_image('combine', (combine.data + 1) / 2.0, step)
                rgb = (cv_img * 255).astype(np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample/' + opt.name + '/' + str(step) + '.jpg', bgr)

        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}/{}: {:.2%}]--[loss_all-{:.6f}]--[loss_fea-{:.6f}]--[loss_flow-{:.6f}]--[lrpf-{:.6f}]--[ETA-{}]'.format(
                                                                               now, epoch_iter,
                                                                               step, all_steps, step/all_steps, 
                                                                               loss_all, loss_fea_sup_all, loss_flow_sup_all, 
                                                                               PF_warp_model.module.old_lr, eta))

        if epoch_iter >= dataset_size:
            break

    # Visualize train loss
    train_loss /= len(train_loader)
    train_fea_loss /= len(train_loader)
    train_flow_loss /= len(train_loader)
    writer.add_scalar('train_loss', train_loss, epoch)
    writer.add_scalar('train_fea_loss', train_fea_loss, epoch)
    writer.add_scalar('train_flow_loss', train_flow_loss, epoch)

    # end of epoch
    iter_end_time = time.time()
    if opt.local_rank == 0:
        print('End of epoch %d / %d: train_loss: %.3f \t time: %d sec' %
                (epoch, opt.niter + opt.niter_decay, train_loss, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        if opt.local_rank == 0:
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_checkpoint(PF_warp_model.module,
                            os.path.join(opt.checkpoints_dir, opt.name, 'PFAFN_warp_epoch_%03d.pth' % (epoch + 1)))

    if epoch > opt.niter:
        PF_warp_model.module.update_learning_rate(optimizer)
