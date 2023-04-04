## Introduction
Use `score_sort.py` to organize a folder of images according to aesthetic score.

## Installation

Git clone this repository:

```
git clone https://github.com/theovercomer8/simulacra-aesthetic-models.git
```

Install pytorch if you don't already have it:

```
pip3 install torch torchvision
```

Then pip install our other dependencies:

```
pip3 install tqdm pillow torchvision scikit-learn numpy
```

If you don't already have it installed, you'll need to install CLIP:

```
git clone https://github.com/openai/CLIP.git
cd CLIP
pip3 install .
cd ..
```

## Usage

`python score_sort.py --help`

```
usage: sort_score.py [-h] [--decimal_places DECIMAL_PLACES] [--operation {copy,move}] src_img_folder dst_img_folder

Sort images in src_img_folder into score folders in dst_img_folder

positional arguments:
  src_img_folder        Source folder containing the images
  dst_img_folder        Destination folder to sort into

options:
  -h, --help            show this help message and exit
  --decimal_places DECIMAL_PLACES
                        Number of decimal places to use for sorting resolution. 2 will create folders like 7.34. 3 Will create 7.345. (default: 1)
  --operation {copy,move}
                        Should the program copy or move the images. (default: copy)
```
