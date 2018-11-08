from __future__ import print_function
import os, sys
from math import log10

from tensorboardX import SummaryWriter

import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from dataset import *
from data import *
from network import *

sys.path.append("./model/")
from model import DPENet

dataset = 'Danbooru'
batchSize = 4
testBatchSize = 4
nEpochs = 200
input_channel = 3
output_channel = 3
ngf = 512
ndf = 512
lr = 0.0002
beta1 = 0.5
ngpu = 1
threads = 4
seed = 999
lamb = 10


writer = SummaryWriter(log_dir='./log', comment='loss')

if ngpu and not torch.cuda.is_available():
    raise Exception("No GPU, use CPU instead")

cudnn.benchmark = True

print("======== Loading Dataset ========")
root_path = "dataset/"
train_set = get_training_set(root_path + dataset)
test_set = get_test_set(root_path + dataset)
training_data_loader = DataLoader(dataset=train_set, num_workers=threads, batch_size=batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=threads, batch_size=batchSize, shuffle=False)

print("======== Building Model ========")
netG = DPENet.DPENet_gen(ngpu)

# loss function
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# optimizer
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
print('======== Networks initialized ========')
# print_network(netG)
# print_network(netD)

print("======== Initialization Done ========")

real_a = torch.FloatTensor(batchSize, input_channel, 512, 512)
real_b = torch.FloatTensor(batchSize, output_channel, 512, 512)

if ngpu:
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    criterionMSE = criterionMSE.cuda()

    real_a = real_a.cuda()
    real_b = real_b.cuda()

def train(epoch):
    for i, batch in enumerate(training_data_loader, 1):
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        # print(batch[0][1].shape)
        # a = batch[0][1].numpy().transpose((1, 2, 0))# Get the information of the incoming image type
        # data = a.astype(np.float64) / np.max(a) # normalize the data to 0 - 1
        # data = 255 * data # Now scale by 255
        # img = data.astype(np.uint8)
        # print(data.shape)
        # cv.imwrite('a.jpg', img)
        pred = netG(real_a)

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################

        optimizerG.zero_grad()
        loss_g_gan = criterionGAN(pred, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(pred, real_b) * lamb

        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward(retain_graph=True)

        optimizerG.step()

        #print(loss_d.shape, loss_g.shape)

        # print("===> Epoch[{}]({}/{})".format(
        #         epoch, 
        #         i, 
        #         len(training_data_loader)
        #     )
        # )
        
        # for visualization
        niter = epoch *len(training_data_loader) + 1
        writer.add_scalars('Gloss', {'Gen_train_loss': loss_g.data.item()}, niter)

        print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
                epoch, 
                i, 
                len(training_data_loader),
                loss_g.data.cpu().numpy()
            )
        )

def test(epoch):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0], batch[1]

        if ngpu:
            input = input.cuda()
            target = target.cuda()
        
        prediction = netG(input)  

        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.data.item())
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    writer.add_scalars('psnr', {'PSNR': avg_psnr/len(testing_data_loader)}, epoch)


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", dataset)):
        os.mkdir(os.path.join("checkpoint", dataset))

    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + dataset))

for epoch in range(1, nEpochs + 1):
    train(epoch)
    test(epoch)
    if epoch % 10 == 0:
        checkpoint(epoch)

writer.close()