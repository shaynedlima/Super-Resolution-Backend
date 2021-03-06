import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import os


def forward_pass(lr_img, bucket_name, models, filename, halve=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    # Saved Model
    srgan_model = os.path.join(models, "srgan_sr_inpainting.pt")
    
    # Load models
    # srgan_generator = torch.load(srgan_model).to(device)
    srgan_generator = torch.load(srgan_model, map_location=torch.device("cpu"))
    srgan_generator.train(False)
    srgan_generator.eval()
    
    # Upload LR Image
    upload_gcp(bucket_name, lr_img, f"lr_{filename}")

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    
    # Upload SR Image
    upload_gcp(bucket_name, sr_img_srgan, f"sr_{filename}")
    #sr_img_srgan.save(os.path.join(results, f"sr_{filename}"))

    del srgan_generator
    del sr_img_srgan
    torch.cuda.empty_cache()
    return