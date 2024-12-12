# DisPose: Disentangling Pose Guidance for Controllable Human Image Animation
We present **DisPose** to mine more generalizable and effective control signals without additional dense input, which disentangles the sparse skeleton pose in human image animation into motion field guidance and keypoint correspondence.
<div align='center'>
<img src="https://anonymous.4open.science/r/DisPose-AB1D/pipeline.png" class="interpolation-image" alt="comparison." height="80%" width="80%" />
</div>

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

### Environment setup
The code requires `python>=3.10`, as well as `torch>=2.0.1` and `torchvision>=0.15.2`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. The demo has been tested on CUDA version of 12.4.
```
conda create -n dispose python==3.10
conda activate dispose
pip install -r requirements.txt
```

### Download checkpoints
1. Download the weights of  [DisPose](https://huggingface.co/lhxxxmad/DisPose/tree/main) and put `controlnet.pth` into `./pretrained_weights/`.

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
|-- controlnet.pth
|-- dwpose
|   |-- dw-ll_ucoco_384.onnx
|   └── yolox_l.onnx
|-- stable-diffusion-v1-5
|-- stable-video-diffusion-img2vid-xt-1-1
```

### Model inference

A sample configuration for testing is provided as `test.yaml`. You can also easily modify the various configurations according to your needs.

```
bash scripts/test.sh 
```