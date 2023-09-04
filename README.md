
# DM-VTON: Distilled Mobile Real-time Virtual Try-On

<div align="center">

  [[`Paper`](https://arxiv.org/abs/2308.13798)]
  [[`Colab Notebook`](https://colab.research.google.com/drive/1oLg0qe0nqLuIeaklzwbkk3IOKmMb0clk)]
  [[`Web Demo`](https://github.com/KiseKloset/KiseKloset)] 

  <img src="https://raw.githubusercontent.com/KiseKloset/DM-VTON/assets/promotion.png" width="35%"><br>
  

  This is official pytorch implementation of [DM-VTON: Distilled Mobile Real-time Virtual Try-On](https://arxiv.org/abs/2308.13798). DM-VTON is designed to be fast, lightweight, while maintaining the quality of the try-on image. It can achieve 40 frames per second on a single Nvidia Tesla T4 GPU and only take up 37 MB of memory.

  <img src="https://raw.githubusercontent.com/KiseKloset/DM-VTON/assets/model_diagram.png" class="left" width='100%'>

  <!-- <img src="https://raw.githubusercontent.com/KiseKloset/DM-VTON/assets/demo.png" class="left" width='100%'> -->

</div>


## <div align="center">📝 Documentation </div>
### Installation
This source code has been developed and tested with `python==3.10`, as well as `pytorch=1.13.1` and `torchvision==0.14.1`. We provide a script to install the environment:
```sh
bash scripts/install.sh
```

### Data Preparation
#### VITON
Because of copyright issues with the original [VITON dataset](https://arxiv.org/abs/1711.08447), we use a resized version provided by [CP-VTON](https://github.com/sergeywong/cp-vton). We followed the work of [Han et al.](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_ClothFlow_A_Flow-Based_Model_for_Clothed_Person_Generation_ICCV_2019_paper.pdf) to filter out duplicates and ensure no data leakage happens (VITON-Clean). You can download VITON-Clean dataset [here](https://drive.google.com/file/d/1-5FtBJtel-ujgKR_TqJEcN2KrhyjBcyp/view?usp=sharing).

| | VITON | VITON-Clean |
| :- | :-: | :-: |
| Training pairs | 14221 | 6824 |
| Testing pairs | 2032 | 416 |

Dataset folder structure:
```
├── VTON-Clean
|   ├── VITON_test
|   |   ├── test_pairs.txt
|   |   ├── test_img
│   │   ├── test_color
│   │   ├── test_edge
|   ├── VITON_traindata
|   |   ├── train_pairs.txt
|   |   ├── train_img       
│   │   │   ├── [000003_0.jpg | ...]  # Person
│   │   ├── train_color     
│   │   │   ├── [000003_1.jpg | ...]  # Garment
│   │   ├── train_edge      
│   │   │   ├── [000003_1.jpg | ...]  # Garment mask
│   │   ├── train_label     
│   │   │   ├── [000003_0.jpg | ...]  # Parsing map
│   │   ├── train_densepose 
│   │   │   ├── [000003_0.npy | ...]  # Densepose
│   │   ├── train_pose      
│   │   │   ├── [000003_0.json | ...] # Openpose
```

<!-- #### Custom dataset -->

### Inference
`test.py` run inference on image folders, then evaluate [FID](https://github.com/mseitzer/pytorch-fid), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), runtime and save results to `runs/TEST_DIR`. Check the sample script for running: `scripts/test.sh`. You can download the pretrained checkpoints [here](https://drive.google.com/drive/folders/1wfWGsR0vWC5LrA26xhj92ec_GoCKV80A).

*Note: to run and save separate results for each pair [person, garment], set `batch_size=1`*.

### Training
For each dataset, you need to train a Teacher network first to guide the Student network. DM-VTON uses [FS-VTON](https://arxiv.org/abs/2204.01046) as the Teacher. Each model is trained through 2 stages: first stage only trains warping module and stage 2 trains the entire model (warping module + generator). Check the sample scripts for training both Teacher network (`scripts/train_pb_warp` + `scripts/train_pb_e2e`) and Student network (`scripts/train_pf_warp` + `scripts/train_pf_e2e`). We also provide a Colab notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oLg0qe0nqLuIeaklzwbkk3IOKmMb0clk) as a quick tutorials.


## <div align="center">📈 Result</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/KiseKloset/DM-VTON/assets/fps.png" class="left" width='60%'>
</div>

### Results on VITON
| Methods | FID $\downarrow$ | Runtime (ms) $\downarrow$ | Memory (MB) $\downarrow$ |
| :- | :-: | :-: | :-: |
| ACGPN (CVPR20) | 33.3 | 153.6 | 565.9 |
| PF-AFN (CVPR21) | 27.3  | 35.8 | 293.3 |
| C-VTON (WACV22) | 37.1 | 66.9 | 168.6 |
| SDAFN (ECCV22) | 30.2 | 83.4  | 150.9 |
| FS-VTON (CVPR22) | 26.5 | 37.5 | 309.3 |
| OURS | 28.2 | 23.3 | 37.8 |

<!-- ## <div align="center">😎 Model Zoo </div> -->

## <div align="center">🙏 Acknowledgments </div>
This code is based on [PF-AFN](https://github.com/geyuying/PF-AFN).