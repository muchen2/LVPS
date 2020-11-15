import torch
from torch import nn
from torch.nn import functional as F

class ImageEncoder(nn.Module):

    def __init__(self, in_channels: int, encode_len: int):
        super(ImageEncoder, self).__init__()
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, encode_len//4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(encode_len//4, encode_len//2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(encode_len//2, encode_len//2, 3, padding=1),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(encode_len//2, encode_len, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(8)
        )

    def forward(self, x):
        B = x.shape[0]
        return self.net(x).reshape(B, -1)

class ImageDecoder(nn.Module):

    def __init__(self, out_channels:int, encode_len:int):
        super(ImageDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(encode_len, encode_len, 4),
            nn.BatchNorm2d(encode_len),
            nn.ReLU(),

            nn.ConvTranspose2d(encode_len, encode_len//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(encode_len//2),
            nn.ReLU(),

            nn.ConvTranspose2d(encode_len//2, encode_len//4, 4, stride=2, padding=1),
            nn.BatchNorm2d(encode_len//4),
            nn.ReLU(),

            nn.ConvTranspose2d(encode_len//4, encode_len//8, 4, stride=2, padding=1),
            nn.BatchNorm2d(encode_len//8),
            nn.ReLU(),

            nn.ConvTranspose2d(encode_len//8, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        B, C = x.shape
        x = x.reshape(B, C, 1, 1)
        return self.net(x)


class SequenceEncoder(nn.Module):

    def __init__(self, in_channels:int, z_dim:int):
        super(SequenceEncoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, z_dim//8, 3, padding=1),
            nn.BatchNorm3d(z_dim//8),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(z_dim//8, z_dim//8, 3, padding=1),
            nn.BatchNorm3d(z_dim//8),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2), stride=(1,2,2)),

            nn.Conv3d(z_dim//8, z_dim//4, 3, padding=1),
            nn.BatchNorm3d(z_dim//4),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(z_dim//4, z_dim//2, 3, padding=1),
            nn.BatchNorm3d(z_dim//2),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(z_dim//2, z_dim, 3, padding=1),
            nn.BatchNorm3d(z_dim),
            nn.ReLU(),
            nn.MaxPool3d((2,2,2), stride=(2,2,2)),

            nn.Conv3d(z_dim, z_dim, 2)
        )

    def forward(self, x):
        B = x.shape[0]
        return self.net(x).reshape(B, -1)

class SequenceGenerator(nn.Module):

    def __init__(self, out_channels:int, z_dim:int, device, in_channels=1, out_seqlen=16):
        super(SequenceGenerator, self).__init__()
        self.out_channels = out_channels
        self.out_seqlen = out_seqlen
        self.z_dim = z_dim
        self.gru = nn.GRU(z_dim, z_dim, batch_first=True)
        self.img_decoder = ImageDecoder(out_channels=out_channels, encode_len=z_dim)
        self.device = device
        self.out_seqlen = out_seqlen
    
    def forward(self, z_img, z_seq, img_shape):
        z_combined = z_img * z_seq + z_img
        B, _ = z_img.shape
        C, H, W = img_shape
        outputs = torch.zeros((B, C, self.out_seqlen, H, W), device=self.device)
        inputs = z_combined.unsqueeze(1)
        h0 = z_combined.unsqueeze(0)
        h = h0
        for t in range(self.out_seqlen):
            step_out, h  = self.gru(inputs, h)
            c_out = self.img_decoder(step_out.squeeze(1))
            c_out = c_out.unsqueeze(2)
            #print(c_out.shape)
            outputs[:, :, t, :, :] = c_out
        return outputs
                  


class ConditionalAdversarialVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 z_dim,
                 encode_len,
                 device,
                 img_size = 64):
        super(ConditionalAdversarialVAE, self).__init__()

        self.in_channels = in_channels
        self.img_size = img_size 
        self.z_dim = z_dim
        self.encode_len = encode_len 
        self.device = device
        self.img_encoder = ImageEncoder(in_channels, encode_len)
        self.seq_encoder = SequenceEncoder(in_channels, z_dim)
        self.seq_decoder = SequenceGenerator(in_channels, z_dim, device)
        self.img_shape = (in_channels, img_size, img_size)
        self.mu_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )
        self.var_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x0, x_seq=None): 
        if x_seq is not None:
            assert (x0.shape[0] == x_seq.shape[0]), "Batch size of initial images and image sequences must be the same."
        if x_seq is None:
            assert (not self.training), "Model is not provided with context frames during training."
        z_img = self.img_encoder(x0).reshape(x0.shape[0], -1)
        if self.training:
            z_seq = self.seq_encoder(x_seq).reshape(x_seq.shape[0], -1)
            mu = self.mu_proj(z_seq)
            log_var = self.var_proj(z_seq)
            z_seq = self.reparameterize(mu, log_var)
        else:
            z_seq = self.sample(x0.shape[0], self.device)

        #z = torch.cat([z, y], dim = 1)
        if self.training:
            return  [self.seq_decoder(z_img, z_seq, self.img_shape), x_seq, mu, log_var]
        else:
            return self.seq_decoder(z_img, z_seq, self.img_shape)
            
    def loss_function(self, recons_seq, orig_seq, mu, log_var, kld_weight=0.5) -> dict:
        recons_loss = F.mse_loss(recons_seq, orig_seq) / orig_seq.shape[0]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int
               ):
        z = torch.randn(num_samples, self.z_dim)
        z = z.to(current_device)
        return z

    def generate(self, x0):
        assert (not self.training), "Call model.eval() before use model for inference"
        return self.forward(x0)