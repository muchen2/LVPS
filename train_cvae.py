import os
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from skimage.transform import resize

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from models.cvae import SequenceConditionalVAE
from utils.datasets import UCFDataset


def pad_last(seq, num_pad):
  for _ in range(num_pad):
    seq = torch.cat([seq, seq[:, -1:]], dim=1)
  return seq

def train_cvae(data, num_channels, g_dim=128, z_dim=64, batch_size=5, num_epochs=10, lr=0.001, 
               beta=0.1, verbose=1, model=None, optimizer=None, use_language=False, contrastive_weight=0.1, video_concat=False,
               add_mult=False, no_var=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
      cvae = SequenceConditionalVAE(in_channels=num_channels, g_dim=g_dim, z_dim=z_dim, 
                                    device=device, use_language=use_language, lan_dim=512, 
                                    video_concat=video_concat, add_mult=add_mult, no_var=no_var).to(device)
    else:
      cvae = model.to(device)
    if optimizer is None:
      optimizer = optim.Adam(cvae.parameters(), lr=lr)
    
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.2)

    mse = []
    for epoch in range(num_epochs):
      epoch_mse = 0
      epoch_kld = 0
      for item in data:
        #print(item.shape)
        if not use_language:
          x_seq = item[0].to(device)
          x_seq = pad_last(x_seq, 6)
          assert x_seq.shape[1] == 16

          x_seq_cfirst = x_seq.permute(0, 4, 1, 2, 3)
          x0_cfirst = x_seq_cfirst[:, :, 0]

          x_pred, x, mu, log_var = cvae(x0_cfirst, x_seq_cfirst)
          losses = cvae.loss_function(x_pred, x, mu, log_var, kld_weight=beta)
          loss = losses['loss']
          recon_loss = losses['Reconstruction_Loss']
          kld_loss = losses['KLD']
          lan_loss = 0
        else:
          x_seq, pos_language, neg_language = item
          x_seq = x_seq.to(device)

def pad_last(seq, num_pad):
  for _ in range(num_pad):
    seq = torch.cat([seq, seq[:, -1:]], dim=1)
  return seq

def train(data, num_channels, g_dim=128, z_dim=64, batch_size=5, num_epochs=10, lr=0.001, 
               beta=0.1, verbose=1, model=None, optimizer=None, use_language=False, contrastive_weight=0.1, video_concat=False,
               add_mult=False, no_var=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
      cvae = SequenceConditionalVAE(in_channels=num_channels, g_dim=g_dim, z_dim=z_dim, 
                                    device=device, use_language=use_language, lan_dim=512, 
                                    video_concat=video_concat, add_mult=add_mult, no_var=no_var).to(device)
    else:
      cvae = model.to(device)
    if optimizer is None:
      optimizer = optim.Adam(cvae.parameters(), lr=lr)
    
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=600, gamma=0.2)

    mse = []
    for epoch in range(num_epochs):
      epoch_mse = 0
      epoch_kld = 0
      for item in data:
        #print(item.shape)
        if not use_language:
          x_seq = item[0].to(device)
          x_seq = pad_last(x_seq, 6)
          assert x_seq.shape[1] == 16

          x_seq_cfirst = x_seq.permute(0, 4, 1, 2, 3)
          x0_cfirst = x_seq_cfirst[:, :, 0]

          x_pred, x, mu, log_var = cvae(x0_cfirst, x_seq_cfirst)
          losses = cvae.loss_function(x_pred, x, mu, log_var, kld_weight=beta)
          loss = losses['loss']
          recon_loss = losses['Reconstruction_Loss']
          kld_loss = losses['KLD']
          lan_loss = 0
        else:
          x_seq, pos_language, neg_language = item
          x_seq = x_seq.to(device)
          pos_language = pos_language.to(device)
          neg_language = neg_language.to(device)
          x_seq = pad_last(x_seq, 6)
          assert x_seq.shape[1] == 16

          x_seq_cfirst = x_seq.permute(0, 4, 1, 2, 3)
          x0_cfirst = x_seq_cfirst[:, :, 0]
          
          x_pred, x, mu, log_var, contrastive_loss = cvae(x0_cfirst, x_seq_cfirst, pos_language, neg_language)
          losses = cvae.loss_function(x_pred, x, mu, log_var, kld_weight=beta)
          loss = losses['loss']
          recon_loss = losses['Reconstruction_Loss']
          kld_loss = losses["KLD"]
          if isinstance(contrastive_loss, torch.Tensor):
            lan_loss = torch.mean(contrastive_loss)
          else:
            lan_loss = 0
          loss += contrastive_weight * lan_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_mse += recon_loss
        epoch_kld += kld_loss
        
      if verbose > 0 and epoch % verbose == 0:
          print("Epoch: {}, Loss: {}, Recons Loss: {}, KLD Loss: {}, Contrastive Loss: {}".format(epoch+1, loss, recon_loss, beta*kld_loss, lan_loss))
      mse.append(epoch_mse)
        
    return cvae, optimizer, np.asarray(mse)

def plot(seq_list, seq_len=10, img_size=(240, 320), ind=False, savedir=None):
  #clip = lambda x: (x + 1) / 2
  clip = lambda x: x
  seq_concats = []
  for seq in seq_list:
    seq_concat = resize(clip(seq[0]), img_size, preserve_range=True)
    for i in range(1, seq_len):
      curr_img = clip(seq[i])
      curr_img = resize(curr_img, img_size, preserve_range=True)
      seq_concat = cv2.hconcat((seq_concat, curr_img ))
    seq_concats.append(seq_concat)
  if not ind:
    fig, axes = plt.subplots(len(seq_list), figsize=(8*10, 6*len(seq_list)), gridspec_kw = {'wspace':0, 'hspace':0})
    #axes[0].imshow(gt_concat, aspect='auto')
    #axes[1].imshow(pred_concat, aspect='auto')
    for i in range(len(seq_list)):
      ax = axes[i]
      ax.imshow(seq_concats[i], aspect='auto')
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.axis('off')
  else:
    for i in range(len(seq_list)):
      fig, ax = plt.subplots(1, figsize=(8,6))
      ax.imshow(seq_concats[i], aspect='auto')
      ax.set_xticklabels([])
      ax.set_yticklabels([])
      ax.axis('off')
      
  if not savedir is None:
    plt.savefig(savedir)
  else:
    plt.show()

# TODO: add argument parser
DATA_DIR = './data/Basketball_npy'
SAVE_DIR = './trained_model'
NUM_EPOCHS = 1
BATCH_SIZE = 1
if not os.path.exists(DATA_DIR):
  raise Exception("Data directory not found.")

if not os.path.exists(SAVE_DIR):
  os.makedirs(SAVE_DIR)

if __name__ == '__main__':
  torch.manual_seed(12345)
  batch_size=BATCH_SIZE
  train_set = UCFDataset(DATA_DIR, img_size=(64,64), start_index=0, end_index=20)
  indices_train = np.arange(20)
  train_loader = torch.utils.data.DataLoader(dataset=train_set, sampler=torch.utils.data.SubsetRandomSampler(indices_train), batch_size=batch_size, shuffle=False)

  model_none, optimizer, mse_none = train(train_loader, num_channels=3, lr=1e-4, z_dim=64, g_dim=128, beta=0.01, 
                                              batch_size=batch_size, num_epochs=NUM_EPOCHS, use_language=False, video_concat=False,
                                              verbose=20, model=None)

  model_lan, _, mse_lan = train(train_loader, num_channels=3, lr=1e-4, z_dim=64, 
                                      g_dim=128, beta=0, batch_size=batch_size, num_epochs=NUM_EPOCHS, 
                                      use_language=True, verbose=20, model=None, contrastive_weight=0.01, video_concat=False)

  model_lan_vid, _, mse_lan_vid = train(train_loader, num_channels=3, lr=1e-4, z_dim=64, g_dim=128, beta=0.01, batch_size=batch_size, 
                                           num_epochs=NUM_EPOCHS, use_language=True, verbose=20, model=None, contrastive_weight=0.005, video_concat=True)

  model_am, _, mse_am = train(train_loader, num_channels=3, lr=1e-4, z_dim=64, g_dim=128, beta=0.01, batch_size=batch_size, 
                                           num_epochs=NUM_EPOCHS, use_language=True, verbose=20, model=None, contrastive_weight=0, add_mult=True)
  
  model_novar, _, mse_novar = train(train_loader, num_channels=3, lr=1e-4, z_dim=64, g_dim=128, beta=0.01, batch_size=batch_size, 
                                           num_epochs=NUM_EPOCHS, use_language=True, verbose=20, model=None, contrastive_weight=0, add_mult=True, no_var=True)
  
  torch.save(model_none.state_dict(), os.path.join(SAVE_DIR, "cvae_none.pth"))
  torch.save(model_lan.state_dict(), os.path.join(SAVE_DIR, "cvae_lan.pth"))
  torch.save(model_lan_vid.state_dict(), os.path.join(SAVE_DIR, "cvae_lan_vid.pth"))
  torch.save(model_am.state_dict(), os.path.join(SAVE_DIR, "cvae_am.pth"))
  torch.save(model_novar.state_dict(), os.path.join(SAVE_DIR, "cvae_no_var.pth"))
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  x_seq_batch = next(iter(train_loader))
  x_seq = x_seq_batch[0][0]
  language = x_seq_batch[1].to(device)
  x0 = x_seq_batch[0][:, 0]
  x0 = x0.permute(0,3,1,2).to(device)

  #model_none = model
  model_none.eval()
  model_lan.eval()
  model_lan_vid.eval()
  model_am.eval()
  model_novar.eval()

  gen_seqs = model_none.generate(x0).detach()
  gen_seqs_lan = model_lan.generate(x0, language).detach()
  gen_seqs_lan_vid = model_lan_vid.generate(x0, language).detach()
  gen_seqs_am = model_am.generate(x0, language).detach()
  gen_seqs_novar =  model_novar.generate(x0, language).detach()

  gen_seq = gen_seqs[0]
  gen_seq_lan = gen_seqs_lan[0]
  gen_seq_lan_vid = gen_seqs_lan_vid[0]

  gen_seq_am = gen_seqs_am[0]
  gen_seq_novar = gen_seqs_novar[0]

  plot([x_seq.cpu().numpy(),
        gen_seq.permute(1,2,3,0).cpu().numpy(),
        gen_seq_lan.permute(1,2,3,0).cpu().numpy(),
        gen_seq_lan_vid.permute(1,2,3,0).cpu().numpy(),
        gen_seq_am.permute(1,2,3,0).cpu().numpy(),
        gen_seq_novar.permute(1,2,3,0).cpu().numpy()], ind=False, savedir="vae_plots.png")
  