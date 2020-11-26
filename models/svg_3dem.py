from models.seq_cvae import SequenceEncoder
from models.svg.dcgan_64 import encoder, decoder
from models.svg.lstm import lstm
        


class SVG_3DEM(nn.Module):

    def __init__(self, in_channels, g_dim, z_dim, device, batch_size, rnn_size=256):
        super(SequenceConditionalVAE, self).__init__()

        self.in_channels = in_channels
        self.z_dim = z_dim
        self.device = device
        self.encoder = encoder(g_dim, in_channels)
        self.decoder = decoder(g_dim, in_channels)
        self.hidden_generator = lstm(g_dim + z_dim, g_dim, rnn_size, 1, batch_size)
        self.seq_encoder = SequenceEncoder(in_channels, z_dim)
        self.batch_size = batch_size

        self.mu_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )
        self.var_proj = nn.Sequential(
            nn.Linear(z_dim, z_dim)
        )
        self.z_proj = nn.Sequential(
            nn.Linear(g_dim + z_dim, g_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x0, x_seq=None, seq_len=16): 
        if x_seq is not None:
            assert (x0.shape[0] == x_seq.shape[0]), "Batch size of initial images and image sequences must be the same."
        if x_seq is None:
            assert (not self.training), "Model is not provided with context frames during training."
        
        if self.training:
            seq_len = x_seq.shape[1]
            embed_seq = [encoder(x_seq[:, i]) for i in range(seq_len)]
        else:
            x0_embed = encoder(x0)[0]

        if self.training:
            z_seq = self.seq_encoder(x_seq).reshape(batch_size, -1)
            mu = self.mu_proj(z_seq)
            log_var = self.var_proj(z_seq)
            z_seq = self.reparameterize(mu, log_var)
        else:
            z_seq = self.sample(batch_size, self.device)

        #z = torch.cat([z, y], dim = 1)
        """
        if self.training:
            return  [self.seq_decoder(z_img, z_seq), x_seq, mu, log_var]
        else:
            return self.seq_decoder(z_img, z_seq)
        """
        
        gen_frames = []
        self.hidden_generator.hidden = self.hidden_generator.init_hidden()
        if self.training: 
            for i in range(1, seq_len):
                h = embed_seq[i-1][0]
                h_pred = self.hidden_generator(torch.cat([h, z_seq], dim=-1))
                x_pred = decoder(h_pred)
                gen_frames.append(x_pred)
            return gen_frames, x_seq, mu, log_var

        else:
            h = x0_embed
            for i in range(1, seq_len):
                h = self.hidden_generator(torch.cat([h, z_seq], dim=-1))
                x_pred = decoder(h)
                gen_frames.append(x_pred)
            return gen_frames

            
    def loss_function(self, recons_seq, orig_seq, mu, log_var, kld_weight=0.5) -> dict:
        mse = 0
        if orig_seq.shape[1] == len(recons_seq) + 1:
            orig_seq = orig_seq[:, 1:]
        for i in range(len(recons_seq)):
            mse += F.mse_loss(recons_seq[i], orig_seq[:, i])
        mse = mse / len(recons_seq)

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