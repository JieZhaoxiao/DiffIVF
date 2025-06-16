from functools import partial
import os
import argparse
import yaml
import torch
from diffivf.unet import create_model
from diffivf.csfd_gdfm import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings

warnings.filterwarnings('ignore')


# Image reading utility
def image_read(path, mode='RGB'):
    """Read image with specified color mode"""
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB', 'GRAY', 'YCrCb'], 'Invalid color mode'

    if mode == 'RGB':
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        return np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    else:  # YCrCb mode
        return cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)


# YAML configuration loader
def load_yaml(file_path: str) -> dict:
    """Load YAML configuration file"""
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


if __name__ == '__main__':
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str, default='configs/diffusion_config.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./output')
    parser.add_argument('--lamb', type=float, default=0.2)  # Fusion weight parameter
    args = parser.parse_args()

    # Initialize logger and device
    logger = get_logger()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)

    # Initialize model and diffusion sampler
    model = create_model(**model_config).to(device).eval()
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model)

    # Setup directories
    test_folder = "demo"
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)

    # Process image pairs
    i = 0
    for img_name in os.listdir(os.path.join(test_folder, "ir")):
        # Load and preprocess images
        inf_img = image_read(os.path.join(test_folder, "vi", img_name), 'GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        vis_img = image_read(os.path.join(test_folder, "ir", img_name), 'GRAY')[np.newaxis, np.newaxis, ...] / 255.0

        # Normalize to [-1, 1]
        inf_img = inf_img * 2 - 1
        vis_img = vis_img * 2 - 1

        # Make dimensions divisible by 32
        scale = 32
        h, w = inf_img.shape[2:]
        h, w = h - h % scale, w - w % scale

        # Convert to PyTorch tensors
        inf_img = torch.FloatTensor(inf_img[:, :, :h, :w]).to(device)
        vis_img = torch.FloatTensor(vis_img[:, :, :h, :w]).to(device)

        logger.info(f"Inference for image {i}")

        # Set random seed for reproducibility
        torch.manual_seed(3407)
        x_start = torch.randn(inf_img.repeat(1, 3, 1, 1).shape, device=device)

        # Run diffusion sampling
        with torch.no_grad():
            sample = sample_fn(
                x_start=x_start,
                record=False,
                I=inf_img,
                V=vis_img,
                save_root=out_path,
                img_index=os.path.splitext(img_name)[0],
                lamb=args.lamb,
                rho=0.001
            )

        # Post-processing and saving
        sample = sample.detach().cpu().squeeze().numpy()
        sample = np.transpose(sample, (1, 2, 0))
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        sample = (sample * 255).astype(np.uint8)

        # Save directly to output directory
        imsave(os.path.join(out_path, f"{img_name.split('.')[0]}.png"), sample)
        i += 1