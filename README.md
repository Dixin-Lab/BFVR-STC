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
        <a href="https://hhhh1138.github.io/project-page-BFVR-STC/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳Webpage-Project-blue">
        </a>
        </a>
    </h4>
</div>

<p align="center">
  🔥 For more results, visit our <a href="https://hhhh1138.github.io/project-page-BFVR-STC/"><strong>project page</strong></a> 🔥
  <br>
  ⭐ If you found this project helpful to your projects, please help star this repo. Thanks! 🤗
</p>

</div>

# Overview
<p align="center">
<figure>
    <img src="assets/figures/stage1.png" width="800px"/>
    <figcaption>Network architecture of Stage I (Codebook learning).</figcaption>
</figure>
<figure>
    <img src="assets/figures/stage2.png" width="800px"/>
    <figcaption>Network architecture of Stage II (Lookup transformer learning).</figcaption>
</figure>
</p>

# Getting Started

## Dependencies and Installation
required packages in `requirements`
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

### Prepare Testing Data

### Inference

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
## Contact

If you have any question, please feel free to contact us via `yutongwang1012@gmail.com`.


