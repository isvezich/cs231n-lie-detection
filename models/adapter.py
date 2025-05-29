import torch
import torch.nn as nn


################# Non-Linear Adapters ######################

# The adapter is placed in-between MHSA and FFN with skip connection, instead of a parallel configuration

class w2v2_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_nlp, self).__init__()

        self.attention = nn.Sequential(
            transformer_encoder.attention,
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )
        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):
        mhsa = self.attention(x) + x
        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


class vit_adapter_nlp(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_nlp, self).__init__()

        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        self.adapter = nn.Sequential(
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=768, bias=True),
            nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        )

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        mhsa = self.drop(attention) + x

        adapter_seq = self.adapter(mhsa) + mhsa
        ffn = self.feed_forward(adapter_seq) + adapter_seq
        return ffn


class mvit_adapter_nlp(nn.Module):
    def __init__(self, transformer_block):
        super(mvit_adapter_nlp, self).__init__()
        self.norm1 = transformer_block.norm1
        self.attn = transformer_block.attn
        self.norm2 = transformer_block.norm2
        self.mlp = transformer_block.mlp
        self.drop_path = transformer_block.drop_path

        self.adapter = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(transformer_block.dim, 256),
            nn.ReLU(),
            nn.Linear(256, transformer_block.dim),
            nn.LayerNorm(transformer_block.dim)
        )

    def forward(self, x):
        x_res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x) + x_res

        x_res = x
        x = self.adapter(x) + x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x) + x_res
        return x



################# Convolutional Adapters ####################

# Conv pass runs parallel to MHSA and/or FFN.
# We use 1d cnn for both audio and visiual modalities

class Efficient_Conv_Pass(nn.Module):
    def __init__(self,):
        super(Efficient_Conv_Pass, self).__init__()

        # More efficient 1d conv - 492k params per encoder layer
        self.adapter_down = nn.Linear(768, 32)  # equivalent to 1 * 1 Conv
        self.adapter_gelu = nn.GELU()
        self.adapter_1d_cnn = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding='same', groups=1, bias=True, padding_mode='zeros')
        self.adapter_up = nn.Linear(32, 768)  # equivalent to 1 * 1 Conv

        self.adapter_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

    def forward(self, x):
        down = self.adapter_gelu(self.adapter_down(x))  # shape = [batch_size, 64, 128]
        down = down.permute(0, 2, 1)  # shape = [batch_size, 128, 64]
        conv = self.adapter_gelu(self.adapter_1d_cnn(down))  # shape = [batch_size, 128, 64]
        conv = conv.permute(0, 2, 1)  # shape = [batch_size, 64, 128]
        up = self.adapter_gelu(self.adapter_up(conv))  # shape = [batch_size, 64, 768]

        out = self.adapter_norm(up + x)  # shape = [batch_size, 64, 768]
        return out


class w2v2_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(w2v2_adapter_conv, self).__init__()

        # Attention Layers
        # note that for W2V2 encoder layer_norm is after MHSA while for ViT encoder layer_norm is before
        self.attention = nn.Sequential(
            transformer_encoder.attention,
            transformer_encoder.dropout,
            transformer_encoder.layer_norm,
        )

        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.feed_forward,
            transformer_encoder.final_layer_norm,
        )

    def forward(self, x):

        # shape of x = [batch_size, 64, 768]
        mhsa = self.attention(x) + x + self.mhsa_conv_pass(x)
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class vit_adapter_conv(nn.Module):
    def __init__(self, transformer_encoder):
        super(vit_adapter_conv, self).__init__()

        # Attention Layers. refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        self.ln1 = transformer_encoder.ln_1
        self.attention = transformer_encoder.self_attention
        self.drop = transformer_encoder.dropout

        # conv layer in adapters
        self.mhsa_conv_pass = Efficient_Conv_Pass()
        self.ffn_conv_pass = Efficient_Conv_Pass()

        # norm required after conv pass
        self.adapter_norm1 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        self.adapter_norm2 = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)

        # Feed Forward Layers
        self.feed_forward = nn.Sequential(
            transformer_encoder.ln_2,
            transformer_encoder.mlp,
        )

    def forward(self, x):
        # shape of x = [batch_size, 64, 768]

        conv_pass = self.mhsa_conv_pass(x)

        # Attention need to be performed individually. Refer EncoderBlock() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        norm_x = self.ln1(x)
        attention, _ = self.attention(query=norm_x, key=norm_x, value=norm_x, need_weights=False)
        attention = self.drop(attention)

        mhsa = attention + x + conv_pass
        mhsa = self.adapter_norm1(mhsa)

        ffn = self.feed_forward(mhsa) + mhsa + self.ffn_conv_pass(mhsa)
        ffn = self.adapter_norm2(ffn)

        return ffn


class mvit_adapter_conv(nn.Module):
    def __init__(self, transformer_block, token_size=(8, 8)):
        super(mvit_adapter_conv, self).__init__()

        self.norm1 = transformer_block.norm1
        self.attn = transformer_block.attn
        self.norm2 = transformer_block.norm2
        self.mlp = transformer_block.mlp
        self.drop_path = transformer_block.drop_path

        self.token_size = token_size  # spatial layout HxW
        dim = transformer_block.norm1.normalized_shape[0]

        # Conv paths
        self.mhsa_conv_pass = Efficient_Conv_Pass(dim)
        self.ffn_conv_pass = Efficient_Conv_Pass(dim)

        self.adapter_norm1 = nn.LayerNorm(dim)
        self.adapter_norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, D = x.shape
        H, W = self.token_size

        # [B, N, D] → [B, D, H, W] for conv pass
        x_reshaped = x.transpose(1, 2).reshape(B, D, H, W)
        conv_pass_mhsa = self.mhsa_conv_pass(x_reshaped).reshape(B, D, -1).transpose(1, 2)

        # Attention block
        res = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = x + res + conv_pass_mhsa
        x = self.adapter_norm1(x)

        # FFN
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)

        conv_pass_ffn = self.ffn_conv_pass(x.transpose(1, 2).reshape(B, D, H, W)).reshape(B, D, -1).transpose(1, 2)
        x = x + res + conv_pass_ffn
        x = self.adapter_norm2(x)

        return x
