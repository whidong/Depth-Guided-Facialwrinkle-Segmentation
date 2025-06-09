import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class ConvGaussian(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_mu = nn.Linear(96, latent_dim)
        self.fc_logvar = nn.Linear(96, latent_dim)

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 128]
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class CLFMAdapter(nn.Module):
    def __init__(self, feature_channels, latent_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, feature_channels * 2),
            nn.ReLU()
        )

    def forward(self, feat, z):
        gamma_beta = self.mlp(z)  # [B, 2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feat + beta

def kl_loss(mu_q, logvar_q, mu_p, logvar_p):
    return 0.5 * torch.mean(
        torch.sum(
            logvar_p - logvar_q +
            (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1,
            dim=1
        )
    )

class ProbSwinUNetr(nn.Module):
    def __init__(self, unetr, latent_dim=96, freeze_encoder=False):
        super().__init__()
        self.unetr = unetr 
        self.latent_dim = latent_dim

        self.prior_net = ConvGaussian(in_channels=5, latent_dim=latent_dim)       # Prior net Depth+Texture
        self.posterior_net = ConvGaussian(in_channels=6, latent_dim=latent_dim)   # Posterior net Depth + Texture + GT

        # Adapter
        self.adapter = CLFMAdapter(48, latent_dim)

        if freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        print("[ProbUNet] Freezing encoder parameters.")
        for m in [self.unet.input_block, self.unet.enc1, self.unet.enc2, self.unet.enc3, self.unet.enc4]:
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, rgb, depth, texture, gt=None, is_training=True, beta=1.0):
        x_input = torch.cat([rgb, depth, texture], dim=1)
        z_prior, mu_p, logvar_p = self.prior_net(torch.cat([rgb, depth, texture], dim=1))

        if is_training and gt is not None:
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)  # (B, H, W) → (B, 1, H, W)
            z_post, mu_q, logvar_q = self.posterior_net(torch.cat([rgb, depth, texture, gt], dim=1))
            z = z_post
        else:
            z = z_prior

        # 1. SwinUNETR encoding
        hidden_states_out = self.unetr.swinViT(x_input)
        enc0 = self.unetr.encoder1(x_input)
        enc1 = self.unetr.encoder2(hidden_states_out[0])
        enc2 = self.unetr.encoder3(hidden_states_out[1])
        enc3 = self.unetr.encoder4(hidden_states_out[2])
        dec4 = self.unetr.encoder10(hidden_states_out[4])

        # 2. Decoding
        dec3 = self.unetr.decoder5(dec4, hidden_states_out[3])
        dec2 = self.unetr.decoder4(dec3, enc3)
        dec1 = self.unetr.decoder3(dec2, enc2)
        dec0 = self.unetr.decoder2(dec1, enc1)
        out = self.unetr.decoder1(dec0, enc0)

        # 3. CLFM
        out = self.adapter(out, z)
        
        # 4. Output
        logits = self.unetr.out(out)
        
        #d4 = self.unet.dec4(x5, x4)
        #d3 = self.unet.dec3(d4, x3)
        #d2 = self.unet.dec2(d3, x2)
        #d1 = self.unet.dec1(d2, x1)
        # out = self.unet.out_conv(d1)

        # loss
        if is_training and gt is not None:
            kld = kl_loss(mu_q, logvar_q, mu_p, logvar_p)
            return logits, kld * beta
        else:
            return logits