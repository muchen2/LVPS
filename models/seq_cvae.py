import torch
from torch import nn
from torch.nn import functional as F

class ImageEmbedder(nn.Module):

    def __init__(self, in_channels: int):
        super(ImageEmbedder, self).__init__()
        self.in_channels = in_channels

        # Translated from content network
        """
        LUA CODE:
        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(3, conv_num/4, 3, 3))
        net:add(nn.ReLU())

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num/4, conv_num/4,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialMaxPooling(2,2))

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num/4, conv_num/2,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num/2, conv_num/2,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialMaxPooling(2,2))

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num/2, conv_num,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num, conv_num,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nValueError: __init__() requires a code object with 0 free vars, not 1
        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num, conv_num,3,3))
        net:add(nn.ReLU())

        net:add(nn.SpatialMaxPooling(2,2))

        net:add(nn.SpatialReflectionPadding(1,1,1,1))
        net:add(nn.SpatialConvolution(conv_num, conv_num*2,3,3))
        net:add(nn.ReLU())


        return net
        """
        base_width = 256
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

        """
        LUA CODE:
        local naf = 64
        encoder:add(nn.MulConstant(0.2))
        encoder:add(nn.VolumetricConvolution(channels, naf, 3, 3, 3, 1, 1, 1, 1, 1, 1))
        encoder:add(nn.VolumetricBatchNormalization(naf))
        encoder:add(nn.ReLU(true))
        encoder:add(nn.VolumetricMaxPooling(1, 2, 2))

        encoder:add(nn.VolumetricConvolution(naf, naf, 3, 3, 3, 1, 1, 1, 1, 1, 1))
        encoder:add(nn.VolumetricBatchNormalization(naf))
        encoder:add(nn.ReLU(true))
        encoder:add(nn.VolumetricMaxPooling(1, 2, 2))

        encoder:add(nn.VolumetricConvolution(naf, naf*2, 3, 3, 3, 1, 1, 1, 1, 1, 1))
        encoder:add(nn.VolumetricBatchNormalization(naf*2))
        encoder:add(nn.ReLU(true))             
        encoder:add(nn.VolumetricMaxPooling(2, 2, 2))   

        encoder:add(nn.VolumetricConvolution(naf*2, naf*4, 3, 3, 3, 1, 1, 1, 1, 1, 1))
        encoder:add(nn.VolumetricBatchNormalization(naf*4))
        encoder:add(nn.ReLU(true))
        encoder:add(nn.VolumetricMaxPooling(2, 2, 2))

        encoder:add(nn.VolumetricConvolution(naf*4, naf*8, 3, 3, 3, 1, 1, 1, 1, 1, 1))
        encoder:add(nn.VolumetricBatchNormalization(naf*8))
        encoder:add(nn.ReLU(true))
        encoder:add(nn.VolumetricMaxPooling(2, 2, 2))

        """
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
        """
        LUA CODE:
        local im_embedding = nn.Identity()() 
        local flow_embedding = nn.Identity()()

        local r000 = nn.CMulTable(2)({im_embedding, flow_embedding})
        local r001 = nn.CAddTable(2)({im_embedding, r000})

        local r002 = nn.VolumetricFullConvolution(z_dim, ngf * 8, 2, 4, 4)(r001)
        local r003 = nn.VolumetricBatchNormalization(ngf * 8)(r002)
        local r004 = nn.ReLU(true)(r003)

        local r005 = nn.VolumetricFullConvolution(ngf * 8, ngf * 4, 4,4,4, 2,2,2, 1,1,1)(r004)
        local r006 = nn.VolumetricBatchNormalization(ngf * 4)(r005)
        local r007 = nn.ReLU(true)(r006)

        local r008 = nn.VolumetricFullConvolution(ngf * 4, ngf * 2, 4,4,4, 2,2,2, 1,1,1)(r007)
        local r009 = nn.VolumetricBatchNormalization(ngf * 2)(r008)
        local r010 = nn.ReLU(true)(r009)

        local r011 = nn.VolumetricFullConvolution(ngf * 2, ngf, 4,4,4, 2,2,2, 1,1,1)(r010)
        local r012 = nn.VolumetricBatchNormalization(ngf)(r011)
        local r013 = nn.ReLU(true)(r012)

        local r014 = nn.VolumetricFullConvolution(ngf, ngf, 3,4,4, 1,2,2, 1,1,1)(r013)
        local r015 = nn.VolumetricBatchNormalization(ngf)(r014)
        local r016 = nn.ReLU(true)(r015)

        local r017 = nn.VolumetricFullConvolution(ngf, channels, 3,4,4, 1,2,2, 1,1,1)(r016)
        local r018 = nn.Tanh()(r017)
        local output = nn.MulConstant(5)(r018)
        """
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
            nn.Tanh()
        )
    
    def forward(self, z_img, z_seq):
        z_combined = z_img * z_seq + z_img
        z_combined = z_combined.reshape(-1, self.z_dim, 1, 1, 1)
        out = self.net(z_combined)
        return out
        


class SequenceConditionalVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 z_dim,
                 device,
                 img_size = 64):
        super(SequenceConditionalVAE, self).__init__()

        self.in_channels = in_channels
        self.img_size = img_size 
        self.z_dim = z_dim
        self.device = device
        self.img_encoder = ImageEmbedder(in_channels)
        self.seq_encoder = SequenceEncoder(in_channels)
        self.seq_decoder = SequenceGenerator(in_channels, z_dim=z_dim)
        base_width = 64
        self.mu_proj = nn.Sequential(
            nn.Linear(base_width*8, z_dim)
        )
        self.var_proj = nn.Sequential(
            nn.Linear(base_width*8, z_dim)
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
            return  [self.seq_decoder(z_img, z_seq), x_seq, mu, log_var]
        else:
            return self.seq_decoder(z_img, z_seq)
            
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