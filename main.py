from RRDB_arch import RRDBNet
from U_Net_arch import Discriminator
import torch
from torch.optim import Adam
import torch.nn as nn

num_epochs = 5
dataloaderLG = [1,1,1]
dataloaderHG = [2,2,2]

netG = RRDBNet(3, 3)
netD = Discriminator()

loss = nn.BCELoss()

realLabel = torch.full((1280, 720), 1)
fakeLabel = torch.full((1280, 720), 0)

adam = Adam()

for epoch in range(num_epochs):
    for X, y in zip(dataloaderLG, dataloaderHG):

        # 判别器loss
        fakeImages = netD(X)
        fakeloss = loss(fakeImages, fakeLabel)
        realloss = loss(y, realLabel)

        l = fakeloss + realloss
        l.backward()
        adam.step()

        