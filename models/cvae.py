import torch
from torch import nn
from torch.nn import functional as F

class ImageEmbedder(nn.Module):

    def __init__(self, in_channels, output_size):
        super(ImageEmbedder, self).__init__()
        self.in_channels = in_channels

        base_width = output_size//2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_width//4, 3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(base_width//4, base_width//4, 3, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(base_width//4, base_width//2, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(base_width//2, base_width//2, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(base_width//2, base_width, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(base_width, base_width, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(base_width, base_width*2, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(8)
        )

    def forward(self, x):
        return self.net(x)


class SequenceEncoder(nn.Module):

    def __init__(self, in_channels, output_size):
        super(SequenceEncoder, self).__init__()
        base_width = output_size//8
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base_width, 3, padding=1),
            nn.BatchNorm3d(base_width),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(base_width, base_width, 3, padding=1),
            nn.BatchNorm3d(base_width),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(base_width, base_width*2, 3, padding=1),
            nn.BatchNorm3d(base_width*2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(base_width*2, base_width*4, 3, padding=1),
            nn.BatchNorm3d(base_width*4),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(base_width*4, base_width*8, 3, padding=1),
            nn.BatchNorm3d(base_width*8),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(base_width*8, base_width*8, 2)
        )

    def forward(self, x):
        return self.net(x)


class SequenceGenerator(nn.Module):

    def __init__(self, out_channels:int, z_dim=2000):
        super(SequenceGenerator, self).__init__()
        self.z_dim = z_dim
        base_width = 64
        self.net = nn.Sequential(
            nn.ConvTranspose3d(z_dim, base_width*8, (2,4,4)),
            nn.BatchNorm3d(base_width*8),
            nn.ReLU(),

            nn.ConvTranspose3d(base_width*8, base_width*4, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(base_width*4),
            nn.ReLU(),

            nn.ConvTranspose3d(base_width*4, base_width*2, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(base_width*2),
            nn.ReLU(),

            nn.ConvTranspose3d(base_width*2, base_width, (4,4,4), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(base_width),
            nn.ReLU(),

            nn.ConvTranspose3d(base_width, out_channels, (3,4,4), stride=(1,2,2), padding=(1,1,1)),
            nn.Sigmoid()
        )
    
    def forward(self, z_img, z_seq):
        z_combined = z_img * z_seq + z_img
        z_combined = z_combined.reshape(-1, self.z_dim, 1, 1, 1)
        out = self.net(z_combined)
        return out
        


class SequenceConditionalVAE(nn.Module):

    def __init__(self, in_channels: int, g_dim, z_dim, device, use_language=False, lan_dim=None, 
                 video_concat=False, add_mult=False, no_var=False):
        super(SequenceConditionalVAE, self).__init__()

        self.in_channels = in_channels
        self.z_dim = z_dim
        self.g_dim = g_dim
        self.device = device
        self.img_encoder = ImageEmbedder(in_channels, z_dim)
        self.seq_encoder = SequenceEncoder(in_channels, g_dim)
        self.seq_decoder = SequenceGenerator(in_channels, z_dim=z_dim)
        self.use_language = use_language
        self.video_concat = video_concat
        self.add_mult = add_mult
        self.no_var = no_var

        self.mu_proj = nn.Sequential(
            nn.Linear(g_dim, z_dim)
        )
        self.var_proj = nn.Sequential(
            nn.Linear(g_dim, z_dim)
        )

        if self.use_language:
          self.language_encoder = nn.Sequential(
              nn.Linear(lan_dim, lan_dim//2),
              nn.ReLU(),
              nn.Linear(lan_dim//2, g_dim),
          )
        
        if self.video_concat or self.add_mult:
          self.latent_proj = nn.Linear(g_dim, z_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x0, x_seq=None, pos_language=None, neg_language=None): 
        if x_seq is not None:
            assert (x0.shape[0] == x_seq.shape[0]), "Batch size of initial images and image sequences must be the same."
        if x_seq is None:
            assert (not self.training), "Model is not provided with context frames during training."
        z_img = self.img_encoder(x0).reshape(x0.shape[0], -1)
        if self.training:
            video_embed = self.seq_encoder(x_seq).reshape(x_seq.shape[0], -1)
            if self.use_language:
              if not self.add_mult:
                assert not (pos_language is None) and not (neg_language is None), "Pos or neg language input is None"
                pos_language_embed = self.language_encoder(pos_language)
                neg_language_embed = self.language_encoder(neg_language)
                pos_sim = torch.sum(video_embed * pos_language_embed, dim=1)
                neg_sim = torch.sum(video_embed * neg_language_embed, dim=1)
                contrastive_loss = torch.exp(neg_sim) / (torch.exp(pos_sim) + 1e-10)
                if not self.video_concat:
                  mu = self.mu_proj(pos_language_embed)
                  log_var = self.var_proj(pos_language_embed)
                  z_seq = self.reparameterize(mu, log_var)
                else:
                  #latent = torch.cat([pos_language_embed, video_embed], dim=-1)
                  #z_seq = self.concat_compressor(latent)
                  z_seq = self.latent_proj(pos_language_embed * video_embed + pos_language_embed)
                  mu = self.mu_proj(video_embed)
                  log_var = self.var_proj(video_embed)
              else:
                assert not (pos_language is None)
                contrastive_loss = 0
                pos_language_embed = self.language_encoder(pos_language)
                z_seq = self.latent_proj(pos_language_embed * video_embed + pos_language_embed)
                if not self.no_var:
                  mu = self.mu_proj(video_embed)
                  log_var = self.var_proj(video_embed)
                else:
                  mu = None
                  log_var = None

            else:
              mu = self.mu_proj(video_embed)
              log_var = self.var_proj(video_embed)
              z_seq = self.reparameterize(mu, log_var)
        else:
          if not self.use_language:
            z_seq = self.sample(x0.shape[0], self.device)
          else:
            assert not (pos_language is None)
              
            if self.video_concat or self.add_mult:
              # Suitable to both video_concat and add_mult
              z_video = torch.randn(x0.shape[0], self.g_dim).to(self.device)
              pos_language_embed = self.language_encoder(pos_language)
              #latent = torch.cat([pos_language_embed, z_video], dim=-1)
              #z_seq = self.concat_compressor(latent)
              z_seq = self.latent_proj(pos_language_embed * z_video + pos_language_embed)
            else:
              pos_language_embed = self.language_encoder(pos_language)
              mu = self.mu_proj(pos_language_embed)
              log_var = self.var_proj(pos_language_embed)
              z_seq = self.reparameterize(mu, log_var)
              
        
        if self.training:
          if self.use_language:
            return  [self.seq_decoder(z_img, z_seq), x_seq, mu, log_var, contrastive_loss]
          else:
            return [self.seq_decoder(z_img, z_seq), x_seq, mu, log_var]
        else:
          return self.seq_decoder(z_img, z_seq)
            
    def loss_function(self, recons_seq, orig_seq, mu, log_var, kld_weight=0.5) -> dict:
        recons_loss = F.mse_loss(recons_seq, orig_seq) / orig_seq.shape[0]
        if (not mu is None) and (not log_var is None):
          kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
          kld_loss = 0

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int
               ):
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        return z

    def generate(self, x0, language=None):
        assert (not self.training), "Call model.eval() before use model for inference"
        return self.forward(x0, pos_language=language)