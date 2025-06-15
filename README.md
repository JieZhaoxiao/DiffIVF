## TDAdv ICONIP 2025
 
   The code repository for our paper TDAdv: Improving Transferability of Unrestricted Adversarial Examples with Text-Guided Diffusion.
## Overview
   <img src="./Overview.png" width="90%"/>
   
If the image doesn't display properly, you can click [here](Overview.png) to view our framework.
## Requirements

1. Hardware Requirements
    - GPU: 1x high-end NVIDIA GPU with at least 24GB memory
    - Memory: At least 40GB of storage memory

2. Software Requirements
    - Python: 3.10
    - CUDA: 12.2

   To install other requirements:

   ```bash
   pip install -r requirements.txt
   ```
3. Datasets
   - Please download the dataset [ImageNet-Compatible](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) and then change the settings of `--images_root` and `--label_path` in [main.py](main.py)

4. Pre-trained Models
   - We adopt [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) as our diffusion model, you can download and load the pretrained weight by setting `--pretrained_diffusion_path="stabilityai/stable-diffusion-2-1-base"` in [main.py](main.py). You can download them from here.
   - Other models used are [Vit-GPT2](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning), [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b), and [GIT](https://huggingface.co/microsoft/git-base). You can download them from here.

## Misleading Prompts Generation
  We provide the code for the misleading prompts generation by Vit-GPT2 and BLIP2. Other generative models are similar.  You can run the following code to generate prompts.
   ```bash
   python text_BLIP2.py
   
   python text_VitGPT.py
   ```

## Crafting Unrestricted Adversarial Examples
   We provide the code for the craft unrestricted adversarial examples. You can view our code to see how our method is implemented. You can run the following code to generate it.
   ```bash
   python main.py
   ```
## Evaluation
   
   We here provide unrestricted adversarial examples crafted for Res-50 using our method and its enhanced version. We store them in .[/output](output). Simply run eval_attack.py to perform attacks against the official PyTorch ResNet50 model. You can modify the attack parameter at the [eval_attack.py](eval_attack.py).  
   
   For eval attack, please run:

   ```bash
   python eval_attack.py
   ```

   You can also run eval_fid.py to get the evaluated FID result.

   For eval FID, please run:

   ```bash
   python eval_fid.py
   ```

   You can also run eval_iqa.py to get the evaluated SSIM, PSNR and LPIPS results.

   For eval SSIM, PSNR and LPIPS, please run:
   
   ```bash
   python eval_iqa.py 
   ```

## Visualization
   We provide more visual qualitative comparisons, so you can see the difference in image quality between our method and the comparison method. Please zoom in for a better view.

   <img src="./Qualitative comparison.png" width="90%"/>
