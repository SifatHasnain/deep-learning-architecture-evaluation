import os
import glob
import re
from PIL import Image
import random

TRAIN_DIR = '/home/ibrahim/HyperParamTuning/deep-learning-model-evaluation/datasets/DogvCat/train/'
VALID_DIR = '/home/ibrahim/HyperParamTuning/deep-learning-model-evaluation/datasets/DogvCat/valid/'

def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def prep_images(paths, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for count, path in enumerate(paths):
        if count % 100 == 0:
            print(path)
        img = Image.open(path)
        basename = os.path.basename(path)
        path_out = os.path.join(out_dir, basename)
        img.save(path_out)



if __name__ == '__main__':
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key)
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key)
    print(len(train_cats), len(train_dogs))
    random.shuffle(train_cats)
    random.shuffle(train_dogs)
    val_cats = train_cats[:512]
    train_cats = train_cats[512:]

    val_dogs = train_dogs[:512]
    train_dogs = train_dogs[512:]

    prep_images(train_cats, f"{TRAIN_DIR}cat/")
    prep_images(train_dogs, f"{TRAIN_DIR}dog/")

    prep_images(val_cats, f"{VALID_DIR}cat/")
    prep_images(val_dogs, f"{VALID_DIR}dog/") 


