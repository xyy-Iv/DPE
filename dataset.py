from os import listdir
from os.path import join

import numpy as np
from PIL import Image
import cv2 as cv
from matplotlib.pyplot import imshow

import torch.utils.data as data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_gray(filepath):
    img = Image.open(filepath).convert('L')
    img = img.resize((512, 512), Image.BICUBIC)
    return img

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    return img

def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.gray_path = join(image_dir, "a")
        self.origin_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.gray_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor()]
                          #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        input = load_gray(join(self.gray_path, self.image_filenames[index]))
        input = self.transform(input)
        # print(input.shape)
        target = load_img(join(self.origin_path, self.image_filenames[index]))
        target = self.transform(target)
        # print(target.shape)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
