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
from models.bnn import BNN
from models.ode import ODE
from utils.datasets import UCFDataset
from utils.ode_learner import ODE_learner


def plot(seq_list, seq_len=10):
  #clip = lambda x: (x + 1) / 2
  clip = lambda x: x
  seq_concats = []
  for seq in seq_list:
    seq_concat = clip(seq[0])
    for i in range(1, seq_len):
      seq_concat = cv2.hconcat((seq_concat, clip(seq[i]) ))
    seq_concats.append(seq_concat)
  
  fig, axes = plt.subplots(len(seq_list), figsize=(seq_len*8, len(seq_list)*6), gridspec_kw = {'wspace':0, 'hspace':0})
  for i in range(len(seq_list)):
    ax = axes[i]
    ax.imshow(seq_concats[i], aspect='auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis('off')
  plt.show()

DATA_DIR = './data/Basketball_npy'
SAVE_DIR = './trained_model'
NUM_EPOCHS = 1
BATCH_SIZE = 1
if not os.path.exists(DATA_DIR):
  raise Exception("Data directory not found.")

if not os.path.exists(SAVE_DIR):
  os.makedirs(SAVE_DIR)

if __name__ == '__main__':
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.manual_seed(12345)
  batch_size=BATCH_SIZE
  train_set = UCFDataset(DATA_DIR, start_index=0, end_index=2, img_size=(64,64))
  train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=False)

  ode_lan = ODE(method='naive language', batch_size=batch_size).to(device)
  trainer_lan = ODE_learner(ode_lan, device=device)
  trainer_lan.learn(train_loader, num_epochs=NUM_EPOCHS, lr=1e-3, use_scheduler=True)

  ode_nce = ODE(method='nce', batch_size=batch_size).to(device)
  trainer_nce = ODE_learner(ode_nce, device=device)
  trainer_nce.learn(train_loader, num_epochs=NUM_EPOCHS, lr=1e-3, use_scheduler=True)

  torch.save(ode_lan.state_dict(), os.path.join(SAVE_DIR, "ode_lan.pth"))
  torch.save(ode_nce.state_dict(), os.path.join(SAVE_DIR, "ode_nce.pth"))

  x_seq_batch = next(iter(train_loader))
  x_seq = x_seq_batch[0][0]
  language = x_seq_batch[1].to(device)
  x0 = x_seq_batch[0][:, 0:1]
  x0 = x0.permute(0,1,4,2,3).to(device)

  gen_seqs_lan = ode_lan(x0, language).detach()
  gen_seqs_nce = ode_nce(x0, language).detach()
  gen_seq_lan = gen_seqs_lan[0]
  gen_seq_nce = gen_seqs_nce[0]
  #print(gen_seq_lan.shape)

  plot([x_seq.cpu().numpy(),
        gen_seq_lan.permute(0,2,3,1).cpu().numpy(),
        gen_seq_nce.permute(0,2,3,1).cpu().numpy()])
        


  