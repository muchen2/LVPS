import torch
import numpy as np
import torch.utils.data as data
import pathlib
from skimage.transform import resize

class UCFDataset(data.Dataset):
  def __init__(self, root, img_size=None, start_index=0, end_index=7):
    self.root = root
    self.img_size = img_size
    print("Preload all data into memory")
    self.all_data = []
    fpaths = pathlib.Path(root).glob("*.npy")
    self.fpaths = [str(fp) for fp in fpaths]
    self.fpaths = sorted(self.fpaths)[start_index:end_index]
    for fp in self.fpaths:
      ndata = torch.FloatTensor(np.load(fp)[:,:,:,::-1].copy())
      if img_size is not None:
        ndata = self.resize_batch_images(ndata)
      self.all_data.append(ndata)
    print("Preload finishes")
    
    # Generate language feature for each video
    n_videos = len(self.all_data)
    self.lan_feats = [torch.rand(1) * torch.randn((512,)) + torch.rand(1) for _ in range(n_videos)]
  
  def __len__(self):
    return len(self.all_data)
  
  def __getitem__(self, index):
    video_data = self.all_data[index]
    pos_lan_feat = self.lan_feats[index]
    other_index = np.random.choice(list(range(index)) + list(range(index+1, len(self.all_data))), 1)[0]
    neg_lan_feat = self.lan_feats[other_index]
    video_data = video_data / 255.0
    return video_data, pos_lan_feat, neg_lan_feat
  
  def resize_batch_images(self, imgs):
    nimg_list = []
    for i in range(imgs.shape[0]):
      nimg = imgs[i]
      nimg = resize(nimg, self.img_size, preserve_range=True)
      nimg_list.append(nimg[np.newaxis, :])
    result = np.vstack(nimg_list)
    #print(result.shape)
    return result