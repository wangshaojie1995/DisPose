# DisPose: Disentangling Pose Guidance for Controllable Human Image Animation
This repository is the official implementation of [DisPose](https://arxiv.org/abs/2412.09349).

[![arXiv](https://img.shields.io/badge/arXiv-2412.09349-b31b1b.svg)](https://arxiv.org/abs/2412.09349)

**📖 Table of Contents**
  - [🎨 Gallery](#-gallery)
  - [🧙 Method Overview](#-method-overview)
  - [🔧 Preparations](#-preparations)
    - [Setup repository and conda environment](#setup-repository-and-conda-environment)
    - [Prepare model weights](#prepare-model-weights)
  - [💫 Inference](#-inference)
  - [📣 Disclaimer](#-disclaimer)
  - [💞 Acknowledgements](#-acknowledgements)

## 🎨 Gallery

<table style="margin: 0 auto; border-collapse: collapse;">
    <tr>
        <td width="20%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://anonymous.4open.science/r/DisPose-AB1D/assets/case1.mp4" muted="false"></video>
        </td>
        <td width="20%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://anonymous.4open.science/r/DisPose-AB1D/assets/case2.mp4" muted="false"></video>
        </td>
        <td width="20%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://anonymous.4open.science/r/DisPose-AB1D/assets/case3.mp4" muted="false"></video>
        </td>
        <td width="20%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://anonymous.4open.science/r/DisPose-AB1D/assets/case4.mp4" muted="false"></video>
        </td>
        <td width="20%" style="border: none;">
            <video width="100%" height="auto" style="display: block; margin: 0px auto;" controls autoplay loop src="https://anonymous.4open.science/r/DisPose-AB1D/assets/case5.mp4" muted="false"></video>
        </td>
    </tr>

</table>

## 🧙 Method Overview
We present **DisPose** to mine more generalizable and effective control signals without additional dense input, which disentangles the sparse skeleton pose in human image animation into motion field guidance and keypoint correspondence.
<div align='center'>
<img src="https://anonymous.4open.science/r/DisPose-AB1D/pipeline.png" class="interpolation-image" alt="comparison." height="80%" width="80%" />
</div>


## 🔧 Preparations
### Setup repository and conda environment
The code requires `python>=3.10`, as well as `torch>=2.0.1` and `torchvision>=0.15.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. The demo has been tested on CUDA version of 12.4.
```
conda create -n dispose python==3.10
conda activate dispose
pip install -r requirements.txt
```

### Prepare model weights
1. Download the weights of  [DisPose](https://huggingface.co/lihxxx/DisPose) and put `DisPose.pth` into `./pretrained_weights/`.

2. Download the weights of other components and put them into `./pretrained_weights/`:
  - [stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main)
  - [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main)
  - [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
  - [MimicMotion](https://huggingface.co/tencent/MimicMotion/tree/main)
3. Downlaod the weights of [CMP](https://huggingface.co/MyNiuuu/MOFA-Video-Hybrid/resolve/main/models/cmp/experiments/semiauto_annot/resnet50_vip%2Bmpii_liteflow/checkpoints/ckpt_iter_42000.pth.tar) and put it into `./mimicmotion/modules/cmp/experiments/semiauto_annot/resnet50_vip+mpii_liteflow/checkpoints`

Finally, these weights should be organized in `./pretrained_weights/`. as follows:


```
./pretrained_weights/
|-- MimicMotion_1-1.pth
|-- DisPose.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.onnx
|   └── yolox_l.onnx
|-- stable-diffusion-v1-5
|-- stable-video-diffusion-img2vid-xt-1-1
```

## 💫 Inference

A sample configuration for testing is provided as `test.yaml`. You can also easily modify the various configurations according to your needs.

```
bash scripts/test.sh 
```

### Tips
- if your GPU memory is limited, try set env PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256.
- 

## 📣 Disclaimer
This is official code of DisPose.
All the copyrights of the demo images and videos are from community users. 
Feel free to contact us if you would like remove them.

## 💞 Acknowledgements
We sincerely appreciate the code release of the following projects: [MimicMotion](https://github.com/Tencent/MimicMotion), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [CMP](https://github.com/XiaohangZhan/conditional-motion-propagation).