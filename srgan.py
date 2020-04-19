import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import os


def forward_pass(img, results, models, filename, halve=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    print(device)

    # Saved Model
    srgan_model = os.path.join(models, "srgan.pt")
    

    # Load models
    # srgan_generator = torch.load(srgan_model).to(device)
    srgan_generator = torch.load(srgan_model, map_location=torch.device("cpu"))
    srgan_generator.train(False)
    srgan_generator.eval()

    # img = "./img/New Images/airport_lr.jpg"

    # Load image, downsample to obtain low-res version
    # hr_img = Image.open(img, mode="r")
    # hr_img = hr_img.convert('RGB')

    hr_img = img.convert('RGB')

    print(hr_img)
    # Want to input an image that has max. pixel side length of 500
    max_pixel_length = 150
    width_scale = int(hr_img.width/max_pixel_length)
    height_scale = int(hr_img.height/max_pixel_length)
    print(width_scale)
    print(height_scale)
    scale = width_scale if (width_scale>height_scale) else height_scale
    scale = 1 if scale==0 else scale

    print("Scale: ", scale)
    lr_img = hr_img.resize((int(hr_img.width / scale), int(hr_img.height / scale)),
                           Image.BILINEAR)

    print("LR Dimensions, W: ", lr_img.width, ", H: ", lr_img.height)

    # if halve:
    #     hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
    #                            Image.LANCZOS)
    # lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
    #                        Image.BILINEAR)

    lr_img.save(os.path.join(results, f"lr_{filename}"))

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    sr_img_srgan.save(os.path.join(results, f"sr_{filename}"))

    del srgan_generator
    del sr_img_srgan
    torch.cuda.empty_cache()
    return