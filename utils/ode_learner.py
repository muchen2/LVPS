from __future__ import print_function
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.cluster import KMeans
from torchdiffeq import odeint


class ODE_learner:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.beta = 0.001
        self.adbeta = self.beta
        self.device = device

    def loss_function(self, recon_img, img, mu, logvar, contrastive_loss, contrastive_weight=0.01):
        self.beta = torch.tensor(self.adbeta)
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(0, True).sum()
        MSE = F.mse_loss(recon_img, img, reduction='mean')
        total_loss = self.beta * KLD + MSE + contrastive_weight * contrastive_loss
        return total_loss, KLD

    def learn(self, data, num_epochs=1500, print_every=50, lr=1e-3, use_scheduler=True):
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99))
        epoch = num_epochs
        if use_scheduler:
          scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)
        seq_len = 10
        batchsize = 2
        for iters in range(epoch):
            accu_loss = 0
            for batch_idx, (video, pos_language, neg_language) in enumerate(data):
                #total_img = torch.reshape(item, [batchsize, 10, 3, 240, 320]).to(device)
                input_seq = video.permute(0, 1, 4, 2, 3).to(self.device)
                pos_language = pos_language.to(self.device)
                neg_language = neg_language.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar, contrastive_loss = self.model(input_seq, pos_language, neg_language)
                loss, KL = self.loss_function(recon.reshape(input_seq.shape), input_seq, mu, logvar, contrastive_loss)
                loss.backward()
                accu_loss += loss.item()
                optimizer.step()
            if use_scheduler:
              scheduler.step()
            if iters % print_every == 0:
              print('==>loss in epoch {} is {}'.format(iters, accu_loss/len(data)))