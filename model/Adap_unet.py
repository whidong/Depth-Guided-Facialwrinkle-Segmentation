import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class ConvGaussian(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        feat = self.encoder(x).squeeze(-1).squeeze(-1)  # [B, 128]
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class CLFMdapter(nn.Module):
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

class TileConcatFcomb(nn.Module):
    def __init__(self, feature_channels, latent_dim, no_convs_fcomb=3):
        super().__init__()
        in_channels = feature_channels + latent_dim

        layers = [nn.Conv2d(in_channels, feature_channels, kernel_size=1), nn.ReLU()]
        for _ in range(no_convs_fcomb - 2):
            layers.append(nn.Conv2d(feature_channels, feature_channels, kernel_size=1))
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        
    def forward(self, feat, z):
        """
        Args:
            feat: Feature map from UNet decoder, shape (B, C, H, W)
            z: Latent vector, shape (B, latent_dim)
        Returns:
            Segmentation output, shape (B, num_classes, H, W)
        """
        B, _, H, W = feat.shape
        z = z.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W)  # (B, latent_dim, H, W)
        x = torch.cat([feat, z], dim=1)  # (B, C + latent_dim, H, W)
        out = self.conv_layers(x)
        return out

class ProbUNet(nn.Module):
    def __init__(self, unet, latent_dim=128, freeze_encoder=False, use_checkpoint=True):
        super().__init__()
        self.unet = unet 
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint

        self.prior_net = ConvGaussian(in_channels=5, latent_dim=latent_dim)       # Depth+Texture
        self.posterior_net = ConvGaussian(in_channels=6, latent_dim=latent_dim)   # Depth + Texture + GT

        self.adapter = CLFMdapter(64, latent_dim)

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
        if self.use_checkpoint:
            # 1. Encoder
            x0 = checkpoint.checkpoint(self.unet.input_block, x_input, use_reentrant=False)
            x1 = checkpoint.checkpoint(self.unet.enc1, x0, use_reentrant=False)
            x2 = checkpoint.checkpoint(self.unet.enc2, x1, use_reentrant=False)
            x3 = checkpoint.checkpoint(self.unet.enc3, x2, use_reentrant=False)
            x4 = checkpoint.checkpoint(self.unet.enc4, x3, use_reentrant=False)
            x5 = checkpoint.checkpoint(self.unet.bottom, x4, use_reentrant=False)

            # 2. Decoder
            d4 = checkpoint.checkpoint(lambda a, b: self.unet.dec4(a, b), x5, x4, use_reentrant=False)
            d3 = checkpoint.checkpoint(lambda a, b: self.unet.dec3(a, b), d4, x3, use_reentrant=False)
            d2 = checkpoint.checkpoint(lambda a, b: self.unet.dec2(a, b), d3, x2, use_reentrant=False)
            d1 = checkpoint.checkpoint(lambda a, b: self.unet.dec1(a, b), d2, x1, use_reentrant=False)

            # 3. CLFM
            d1 = checkpoint.checkpoint(lambda x: self.adapter(x, z), d1, use_reentrant=False)
        
            out = checkpoint.checkpoint(self.unet.out_conv, d1, use_reentrant=False)
        else:
            # 1. Encoder
            x0 = self.unet.input_block(x_input)
            x1 = self.unet.enc1(x0)
            x2 = self.unet.enc2(x1)
            x3 = self.unet.enc3(x2)
            x4 = self.unet.enc4(x3)
            x5 = self.unet.bottom(x4)

            # 2. Decoder
            d4 = self.unet.dec4(x5, x4)
            d3 = self.unet.dec3(d4, x3)
            d2 = self.unet.dec2(d3, x2)
            d1 = self.unet.dec1(d2, x1)

            # 3. CLFM
            d1 = self.adapter(d1, z)
        
            out = self.unet.out_conv(d1)

        # loss
        if is_training and gt is not None:
            kld = kl_loss(mu_q, logvar_q, mu_p, logvar_p)
            return out, kld * beta
        else:
            return out