import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import Iterable
from torch.autograd import Variable
import torch
import numpy as np
from torchdiffeq import odeint
from models.bnn import BNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ODE(nn.Module):
    def __init__(self, num_channels=3, img_size=(64,64), method='naive language', batch_size=5):
        super(ODE, self).__init__()
        self.encoder = Encoder(num_channels=num_channels)
        self.rnn = RNN_FeatEncoder(batch_size=batch_size)
        self.decoder = Decoder(num_channels=num_channels)
        self.bnn = BNN(256+16, 256, act='tanh', n_hidden=128, bnn=False)
        self.bnn.draw_f()
        self.language_encoder = Language_Encoder()
        self.img_size = img_size
        self.method = method
        self.lan_mu_proj = nn.Linear(128, 128)
        self.lan_var_proj = nn.Linear(128, 128)
        self.vid_feat_proj = nn.Linear(256, 128)
        self.batch_size = batch_size
        

    def forward(self, input, pos_language=None, neg_language=None):
      """
        input is video trajectory in training, or single image in testing.
        language is language feature from VisualCommet
      """
      if pos_language is None:
        batchsize = input.shape[0]
        latent_total= self.encoder(input).view([batchsize, -1])
        state = torch.randn(batchsize, 128).to(device)
        init_state = latent_total
      else:
        batchsize = input.shape[0]
        if input.shape[1]==1:
          # Only given the first image
          if self.method == "naive language":
            latent_total= self.encoder(input.squeeze(1)).view([batchsize, 1, -1])
            language_feature = self.language_encoder(pos_language)
            state = torch.randn(batchsize, 128).to(device) * language_feature + language_feature
            init_state = latent_total[:,0]
          elif self.method == 'nce':
            latent_total= self.encoder(input.squeeze(1)).view([batchsize, 1, -1])
            language_feature = self.language_encoder(pos_language)
            mu = self.lan_mu_proj(language_feature)
            logvar = self.lan_var_proj(language_feature)
            state = self.decoder.reparametrize(mu, logvar)
            init_state = latent_total[:,0]

        else:
          contrastive_loss = 0
          # Training time, given a sequence
          latent_total= self.encoder(input.reshape([batchsize*10, 3, 64, 64])).view([batchsize, 10, -1])
          h = self.rnn.init_h.to(device)
          c = self.rnn.init_c.to(device)
          for i in reversed(range(10)):
              latent, h, c = self.rnn(latent_total[:, i:i+1, :].permute(1, 0, 2), h, c)
          state_dist = self.rnn.linearlay(latent.view([batchsize, 256]))

          if self.method == 'naive language':
            mu = state_dist[:, :128]
            logvar = state_dist[:, 128:]
            language_feature = self.language_encoder(pos_language)
            video_latent = self.decoder.reparametrize(mu, logvar)
            state = video_latent * language_feature + language_feature
            init_state = latent_total[:, 0]
          elif self.method == 'nce':
            video_feature = self.vid_feat_proj(state_dist)
            pos_language_feature = self.language_encoder(pos_language)
            neg_language_feature = self.language_encoder(neg_language)
            pos_sim = torch.sum(video_feature * pos_language_feature, dim=1)
            neg_sim = torch.sum(video_feature * neg_language_feature, dim=1)
            contrastive_loss = torch.mean(torch.exp(neg_sim) / (torch.exp(pos_sim) + 1e-10))
            mu = self.lan_mu_proj(pos_language_feature)
            logvar = self.lan_var_proj(neg_language_feature)
            state = self.decoder.reparametrize(mu, logvar)
            init_state = latent_total[:, 0]
          
      concat_state = torch.cat([state, init_state], 1)
      ts1 = torch.tensor(np.linspace(1, 10, 10)).to(device)
      output_latent = odeint(self.bnn, concat_state, ts1)
      recon = self.decoder(output_latent.reshape([batchsize*10, 256])).reshape([batchsize, 10, 3, 64, 64])
      if input.shape[1]==1:
        return recon
      else:
        return recon, mu, logvar, contrastive_loss

class Language_Encoder(nn.Module):
    def __init__(self):
        super(Language_Encoder, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
      out = self.relu(self.fc1(input))
      out = self.fc2(out)
      return out

class Encoder(nn.Module):
    def __init__(self, num_channels=3):
        super(Encoder, self).__init__()
        self.cnn1 = nn.Conv2d(num_channels, 4, kernel_size=4, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.cnn3 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        #self.fc1 = nn.Linear(15*20*32, 256)
        self.fc1 = nn.Linear(4*4*32, 256)
        self.fc2 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(32)

        #self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        # self.weight_init()

    def forward(self, input):
        out = self.relu1(self.bn1(self.cnn1(input)))
        out = self.relu2(self.bn2(self.cnn2(out)))
        out = self.relu3(self.bn3(self.cnn3(out)))
        out = self.relu4(self.bn4(self.cnn4(out)))
        out = self.relu5(self.fc1(out.reshape([-1, 4*4*32])))
        return self.fc2(out)

    def weight_init(self):
        for block in self._modules:
            if isinstance(self._modules[block], Iterable):
                for m in self._modules[block]:
                    if self.args.init_method == 'kaiming':
                        m.apply(kaiming_init)
                    elif self.args.init_method == 'xavier':
                        m.apply(xavier_uniform_init)
                    else:
                        m.apply(normal_init)
            else:
                if self.args.init_method == 'kaiming':
                    self._modules[block].apply(kaiming_init)
                elif self.args.init_method == 'xavier':
                    self._modules[block].apply(xavier_uniform_init)
                else:
                    self._modules[block].apply(normal_init)

class RNN_FeatEncoder(nn.Module):
    def __init__(self, batch_size=5):
        super(RNN_FeatEncoder, self).__init__()
        latent_size = 256
        self.lstm = nn.LSTM(latent_size//2, latent_size, 1)
        self.batch = batch_size
        self.init_h = torch.zeros([1, self.batch, latent_size], device=device)
        self.init_c = torch.zeros([1, self.batch, latent_size], device=device)
        self.fc3 = nn.Linear(latent_size, latent_size)
        self.fc4 = nn.Linear(latent_size, latent_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, h, c):
        out, (h, c) = self.lstm(input, (h, c))
        return out, h, c

    def linearlay(self, input):
        temp = self.relu(self.fc3(input))
        return self.fc4(temp)


class Decoder(nn.Module):
    def __init__(self, num_channels=3):
        super(Decoder, self).__init__()
        self.cnn2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.cnn3 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.cnn4 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.cnn5 = nn.ConvTranspose2d(4, num_channels, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(256, 256)
        #self.fc2 = nn.Linear(256, 15*20*32)
        self.fc2 = nn.Linear(256, 4*4*32)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu6 = nn.ReLU()

    def forward(self, z):
        out = self.relu1(self.fc1(z))
        out = self.relu2(self.fc2(out)).reshape([-1, 32, 4, 4])
        out = self.relu3(self.bn2(self.cnn2(out)))
        out = self.relu4(self.bn3(self.cnn3(out)))
        out = self.relu5(self.bn4(self.cnn4(out)))
        out = self.relu6(self.cnn5(out))
        return out

    def reparametrize(self, mu, logvar):
        std = logvar.div(2).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm3d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean=0, std=0.1):
    if isinstance(m, (nn.Linear, nn.Conv3d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

def xavier_uniform_init(m):
    if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
            m.bias.data.fill_(0.01)

    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
