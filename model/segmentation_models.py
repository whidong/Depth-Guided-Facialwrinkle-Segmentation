from model.UNet_model import UNet 
from model.swin_unetr_model import SwinUNETR 
from model.Adap_unet import ProbUNet 
from model.Adap_Swinunetr import ProbSwinUNetr 


def create_model(model_type="swin_unetr", depth = (2, 2, 6, 2), in_channels=3, out_channels=1, feature_size=48, use_checkpoint=True, use_v2=True, pretrain = False, pretrain_path = "None", prob_model = None):
    """
    Create model
    Parameters:
        - model_type (str): swin_unetr, unet, prob_swin_unetr, prob_unet
        - depth (sequencial: int) : Swin UNETR number of swinTransformer block
        - in_channels (int): input channel
        - out_channels (int): output channel (class)
        - feature_size (int): Swin UNETR transformer embedding feature size
        - use_checkpoint (bool): Gradient checkpoint memory optimize
        - use_v2 (bool): select Swin UNETR backbone SwinTransformer or SwinTransformerV2
        - pretrain (bool) : Swin UNETR use pretrained imageNet SwinTransformerV2
        - pretrain_path (str): path of pretrain ckpt file
        - prob_model (model): 
            Accepts a backbone network instance (UNet or Swin UNETR) as input.
            Used when creating a probabilistic segmentation model combined with a CLFM (Conditional Latent Feature Modulator) module.
            Not required for standard models ('swin_unetr', 'unet').
    Returns:
        - model: select segmentation model
    """
    # Swin UNETR model
    if model_type.lower() == "swin_unetr":
        return SwinUNETR(
            img_size = (1024, 1024),
            in_channels = in_channels,
            out_channels = out_channels,
            feature_size = feature_size,
            use_checkpoint = use_checkpoint,
            spatial_dims = 2,
            depths = depth,
            num_heads = (3, 6, 12, 24),
            use_v2 = use_v2,
            pretrain = pretrain,
            pretrain_path = pretrain_path
        )
    
    elif model_type.lower() == "prob_swinunetr":
        return ProbSwinUNetr(
            unetr = prob_model,
            latent_dim = feature_size,
        )
    
    # UNet model
    elif model_type.lower() == "unet":
        return UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(64, 128, 256, 512, 1024),
            bilinear=True
        )
    elif model_type.lower() == "probunet":
        return ProbUNet(
            unet = prob_model,
            latent_dim = feature_size,
            use_checkpoint = use_checkpoint
        )
    
    else:
        print(model_type)
        raise ValueError("Invalid model_type. Choose either 'swin_unetr' or 'unet'.")


