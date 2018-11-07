from __future__ import print_function
import os, sys

import torch
import torchvision.transforms as transforms

sys.path.append("./model/")

from dataset import *
from network import *

dataset = 'Danbooru'
model = 'checkpoint/Danbooru/netG_model_epoch_50.pth'
ngpu = 1
netG = torch.load(model)
netG.eval()

image_dir = 'dataset/' + dataset + '/test/a/'
image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.view(1, -1, 512, 512)

    if ngpu:
        netG = netG.cuda()
        input = input.cuda()

    out = netG(input)
    out = out.cpu()
    out_img = out.data[0]
    if not os.path.exists(os.path.join("result", dataset)):
        os.makedirs(os.path.join("result", dataset))
    save_img(out_img, "result/{}/{}".format(dataset, image_name))
