## DiffIVF PRCV 2025
 
   Codes for DiffIVF: Infrared-Visible Image Fusion Via  Diffusion Models For Object Detection.
## Overall Framework
   <img src="./fig/DiffIVF.png" width="99%"/>

## Abstract
   In this work, we propose an early-fusion of infrared images and visible images to improve performance of object detectors under complex environments. We frame the fusion of infrared and visible images as a generation task, and propose a simple yet effective scheme built on diffusion model with guided image to generate the fused images, called DiffIVF. How to quickly generate fused images with high texture and intensity fidelity is an important issue for object detection. We address these issues in the whole process of the diffusion model. In the forward process, a cosine noise schedule is proposed to improve the smoothness of the noise level, which stabilizes the diffusion process for the fast sampling. In the reverse process, a large sampling interval is utilized to accelerate the denoising process. To compensate for the information loss during fast sampling, the visible image signal is selected to guide the fusion at each denoising timestep, aiming at enriching textural details and structural information. The guided image weight averaged with an intermediate fused image inputs into a pre-trained U-net for the next time of reversing. Notice that the proposed method needs no training or fine-tuning, which enables us exclusively to make use of off-the-shelf models for fusion. Experimental results show that the proposed method obtains better image quality scores of textures and intensity fidelity than previous methods, thus triggering better detection performance for the infrared-visible fused images.

## Dataset
   Please download the dataset [MSRS](https://github.com/Linfeng-Tang/MSRS), [M3FD](https://github.com/JinyuanLiu-CV/TarDAL), [LLVIP](https://github.com/bupt-ai-cz/LLVIP) and put them into [datasets](datasets). You can download them from here.

## Enviroment
Important packages:
   ```bash
   lpips==0.1.4
   matplotlib==3.7.2
   numpy==1.24.3
   opencv_python==4.5.3.56
   packaging==23.1
   Pillow==10.0.0
   Pillow==9.4.0
   Pillow==10.0.1
   PyYAML==5.4.1
   PyYAML==6.0.1
   scikit_image==0.19.3
   scipy==1.9.3
   torch==1.8.1+cu111
   torchvision==0.9.1+cu111
   tqdm==4.62.0
   ```

## Pre-trained Model
   We adopt [guided-diffusion](https://github.com/openai/guided-diffusion) as our diffusion model, you can download the checkpoint "256x256_diffusion_uncond.pt" and paste it to [models](models). You can download them from here.

## Generate Fused Image
   We provide the code for the generate fused image. You can view our code to see how our method is implemented. You can run the following code to generate it.
   ```bash
   python main.py
   ```
## Qualitative fusion results.
   <img src="./fig/Day.png" width="99%"/>

   <img src="./fig/Night.png" width="99%"/>

