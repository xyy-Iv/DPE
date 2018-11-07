from os import listdir
from os.path import join

import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow

import torch.utils.data as data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.gray_path = join(image_dir, "a")
        self.origin_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.gray_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        input = load_img(join(self.gray_path, self.image_filenames[index]))
        input = self.transform(input)
        target = load_img(join(self.origin_path, self.image_filenames[index]))
        target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
