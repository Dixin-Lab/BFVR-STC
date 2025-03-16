<div align="center">

<h1>Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency</h1>
<div>
    <a href='https://yutongwang1012.github.io/' target='_blank'>Yutong Wang</a>&emsp;
    <a href='https://openreview.net/profile?id=~Jiajie_Teng1' target='_blank'>Jiajie Teng</a>&emsp;
    <a href='https://openreview.net/profile?id=~Jiajiong_Cao1' target='_blank'>Jiajiong Cao</a>&emsp;
    <a href='https://openreview.net/profile?id=~Yuming_Li5' target='_blank'>Yuming Li</a>&emsp;
    <a href='https://openreview.net/profile?id=~Chenguang_Ma3' target='_blank'>Chenguang Ma</a>&emsp;
    <a href='https://hongtengxu.github.io/' target='_blank'>Hongteng Xu</a>&emsp;
    <a href='https://dixinluo.github.io/' target='_blank'>Dixin Luo</a>
</div>

<div>
    <h4 align="center">
        <a href="https://arxiv.org/abs/2411.16468" target='_blank'>
        <img src='https://img.shields.io/badge/arXiv-2411.16468-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
        </a>
        <a href="https://dixin-lab.github.io/project-page-BFVR-STC/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳Webpage-Project-blue">
        </a>
        </a>
    </h4>
</div>

<p align="center">
  🔥 For more results, visit our <a href="https://dixin-lab.github.io/project-page-BFVR-STC/"><strong>project page</strong></a> 🔥
  <br>
  ⭐ If you found this project helpful to your projects, please help star this repo. Thanks! 🤗
</p>

</div>

# Overview
| <img src="https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/stage1.png" alt="Stage1" width="800"> |
|:----------------------:|
| Network architecture of Stage 1 (Codebook learning).                       |
| <img src="https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/stage2.png" alt="Stage1" width="800"> |
| Network architecture of Stage 2 (Lookup transformer learning).                       |

**TL;DR**: STC is a novel video face enhancement framework that efficiently solves the BFVR and de-flickering tasks.

# Gallery
## Blind face video restoration 
| Degraded    | Enhanced     | Degraded      | Enhanced      |
|------------------|------------------|------------------|------------------|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-1-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-1-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-2-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-2-0.gif)|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-3-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-3-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-4-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-4-0.gif)|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-5-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-5-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-6-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/1-6-0.gif)|

## Brightness de-flickering 
| Degraded    | Enhanced     | Degraded      | Enhanced      |
|------------------|------------------|------------------|------------------|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-1-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-1-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-2-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-2-0.gif)|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-3-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-3-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-4-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/2-4-0.gif)|

## Pixel de-flickering 
| Degraded    | Enhanced     | Degraded      | Enhanced      |
|------------------|------------------|------------------|------------------|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-1-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-1-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-2-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-2-0.gif)|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-3-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-3-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-4-1.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/3-4-0.gif)|

## Pixel de-flickering (for synthesized talking head videos)
| Degraded    | Enhanced     | Degraded      | Enhanced      |
|------------------|------------------|------------------|------------------|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-1-1.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-1-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-1-2.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-2-0.gif)|
| ![GIF1](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-1-3.gif)| ![GIF2](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-3-0.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-1-4.gif)| ![GIF3](https://github.com/Dixin-Lab/BFVR-STC/blob/main/assets/figures/4-4-0.gif)|

# Getting Started

## Dependencies and Installation
Install required packages in `environment.yaml`
```
# git clone this repository
git clone https://github.com/Dixin-Lab/BFVR-STC
cd BFVR-STC

# create new anaconda env
conda env create -f environment.yaml
conda activate bfvr

# install python dependencies
conda install -c conda-forge dlib
conda install -c conda-forge ffmpeg
```

## Quick Inference

### Download Pre-trained Models
All pretrained models can also be automatically downloaded during the first inference. You can also download our pretrained models from [Releases](https://github.com/Dixin-Lab/BFVR-STC/releases/tag/init) to the weights folder.

### Training and Testing Data
VFHQ and VFHQ-Test dataset can be downloaded from the [webpage](https://liangbinxie.github.io/projects/vfhq/). The data processing functions can be found in the [utils]() directory.  

### Inference
🧑🏻 Blind Face Video Restoration
```
python scripts/infer_bfvr.py --input_path [video path] --output_base [output directory]
```
🧑🏻 Face Video Brightness De-flickering
```
python scripts/infer_deflicker.py --input_path [video path] --output_base [output directory]
```
🧑🏻 Face Video Pixel De-flickering
```
python scripts/infer_deflickersd.py --input_path [video path] --output_base [output directory]
```
### Evaluation 
The implementation of commonly used metrics, such as PSNR, SSIM, LPIPS, FVD, IDS and AKD, can be found in the [evaluation](https://github.com/Dixin-Lab/BFVR-STC/tree/main/evaluation). Face-Consistency and Flow-Score can be calculated by video evaluation benchmark [EvalCrafter](https://github.com/evalcrafter/EvalCrafter).

## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @article{wang2024efficient,
    title={Efficient Video Face Enhancement with Enhanced Spatial-Temporal Consistency},
    author={Yutong Wang and Jiajie Teng and Jiajiong Cao and Yuming Li and Chenguang Ma and Hongteng Xu and Dixin Luo},
    journal={arXiv preprint arXiv:2411.16468},
    year={2024}
}
   ```

## Acknowledgement
The code framework is mainly modified from [CodeFormer](https://github.com/sczhou/CodeFormer/). Please refer to the original repo for more usage and documents.

## Contact

If you have any question, please feel free to contact us via `yutongwang1012@gmail.com`.


