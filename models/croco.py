# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------



import torch
import torch.nn as nn

from functools import partial

from models.blocks import Block, DecoderBlock, PatchEmbed
from models.pos_embed import get_2d_sincos_pos_embed
from models.masking import RandomMask


class CroCoNet(nn.Module):

    def __init__(self,
                 img_size=224,           # input image size
                 patch_size=16,          # patch_size 
                 mask_ratio=0.9,        # ratios of masked tokens 
                 enc_embed_dim=768,      # encoder feature dimension
                 enc_depth=12,           # encoder depth 
                 enc_num_heads=12,       # encoder number of heads in the transformer block 
                 dec_embed_dim=512,      # decoder feature dimension 
                 dec_depth=8,            # decoder depth 
                 dec_num_heads=16,       # decoder number of heads in the transformer block 
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_im2_in_dec=True,   # whether to apply normalization of the 'memory' = (second image) in the decoder 
                ):
                
        super(CroCoNet, self).__init__()
                
        # patch embeddings  (with initialization done as in MAE)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

        # mask generations
        self.mask_generator = RandomMask(self.patch_embed.num_patches, mask_ratio)

        # positional embedding of the encoder 
        enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
        self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())

        # transformer for the encoder 
        self.enc_blocks = nn.ModuleList([
            Block(enc_embed_dim, enc_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(enc_depth)])
        self.enc_norm = norm_layer(enc_embed_dim)
        
        # masked tokens 
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)

        # positional embedding of the decoder  
        dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
        self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
                        
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec)
            for i in range(dec_depth)])
        self.dec_norm = norm_layer(dec_embed_dim)
        
        # prediction head 
        self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)
        
        # initializer weights
        self.initialize_weights()           
        
    def initialize_weights(self):
        # patch embed 
        self.patch_embed._init_weights()
        # mask tokens
        torch.nn.init.normal_(self.mask_token, std=.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _encode_image(self, image, do_mask=False):
        """
        image has B x 3 x img_size x img_size 
        masking: whether to perform masking or not
        """
        # embed the image into patches  (x has size B x Npatches x C)
        x = self.patch_embed(image)              
        # add positional embedding without cls token  
        x = x + self.enc_pos_embed[None,...]
        # apply masking 
        B,N,C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
        else:
            B,N,C = x.size()
            masks = torch.zeros((B,N), dtype=bool)
        # now apply the transformer encoder and normalization        
        for blk in self.enc_blocks:        
            x = blk(x)
        x = self.enc_norm(x)
        return x, masks
 
    def _decoder(self, feat1, masks1, feat2):
        # encoder to decoder layer 
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B,Nenc,C = visf1.size()
        Ntotal = masks1.size(1)
        f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
        f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding and image embedding
        f1 = f1_ + self.dec_pos_embed
        f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1 
        out2 = f2 
        for blk in self.dec_blocks:
            out, out2 = blk(out, out2)
        out = self.dec_norm(out)
        return out

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        
        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    def forward(self, img1, img2):
        """
        img1: tensor of size B x 3 x img_size x img_size
        img2: tensor of size B x 3 x img_size x img_size
        
        out will be    B x N x (3*patch_size*patch_size)
        masks are also returned as B x N just in case 
        """
        # encoder of the masked first image 
        feat1, mask1 = self._encode_image(img1, do_mask=True)
        # encoder of the second image 
        feat2, _ = self._encode_image(img2, do_mask=False)
        # decoder 
        decfeat = self._decoder(feat1, mask1, feat2)
        # prediction head 
        out = self.prediction_head(decfeat)
        # get target
        target = self.patchify(img1)
        return out, mask1, target