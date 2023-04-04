
import argparse
import os

import torch
import tqdm
from clip import clip
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms
from simulacra_fit_linear_model import AestheticMeanPredictionLinearModel
from torch.nn import functional as F
import shutil

device='cuda'
clip_model_name = 'ViT-B/16'
clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
clip_model.eval().requires_grad_(False)
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])
model = AestheticMeanPredictionLinearModel(512)
model.load_state_dict(
    torch.load("models/sac_public_2022_06_29_vit_b_16_linear.pth")
)
model = model.to(device)

def sort_images(src, dest, decimal_places, operation):
    if not os.path.exists(dest):
        os.makedirs(dest)

    paths = []
    for root, dirs, files in os.walk(src, topdown=False):
        for name in files:
          
          if os.path.splitext(os.path.split(name)[1])[1].upper() not in ['.JPEG','.JPG','.JPE', '.PNG', '.WEBP']:
              continue
          
          paths.append(os.path.join(root, name))
    
    for path in tqdm.tqdm(paths):
        image = Image.open(path)
        filename = os.path.split(path)[1]
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        img = TF.resize(image, 224, transforms.InterpolationMode.LANCZOS)
        img = TF.center_crop(img, (224,224))
        img = TF.to_tensor(img).to(device)
        img = normalize(img)
        clip_image_embed = F.normalize(
            clip_model.encode_image(img[None, ...]).float(),
            dim=-1)
        score = str(round(model(clip_image_embed).item(),decimal_places))

        os.makedirs(os.path.join(dest,score),exist_ok=True)
        dest_filename = os.path.join(dest,score,filename)
        if operation == 'copy':
           shutil.copy(path,dest_filename)
        elif operation == 'move':
           shutil.move(path,dest_filename)
           
def main():
  parser = argparse.ArgumentParser(
      description='Sort images in src_img_folder into score folders in dst_img_folder')
  parser.add_argument('src_img_folder', type=str, help='Source folder containing the images')
  parser.add_argument('dst_img_folder', type=str, help='Destination folder to sort into')

  parser.add_argument('--decimal_places', type=int,
                      help='Number of decimal places to use for sorting resolution. 2 will create folders like 7.34. 3 Will create 7.345. (default: 1)', default=1)
  parser.add_argument('--operation', type=str, help='Should the program copy or move the images. (default: copy)', choices=['copy','move'], default='copy')
  args = parser.parse_args()
  sort_images(args.src_img_folder, args.dst_img_folder, args.decimal_places, args.operation)


if __name__ == '__main__':
  main()
