import torch
import torch.nn as nn
import torchvision
from models.adapter import mvit_adapter_conv, mvit_adapter_nlp
from torchvision.models.video.mvit import PositionalEncoding



class conv3d_block(nn.Module):
    def __init__(self, in_channels, out_channels, pad='same', k=3, s=1):
        super(conv3d_block, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, padding=pad, stride=s, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class cnn_face(nn.Module):
    def __init__(self, ):
        super(cnn_face, self).__init__()

        self.conv1 = conv3d_block(3, 64, k=7, pad=3, s=2)
        self.layer1 = nn.Sequential(
            conv3d_block(64, 64),
            conv3d_block(64, 64),
        )

        self.conv2 = conv3d_block(64, 128, k=3, pad=3, s=2)
        self.layer2 = nn.Sequential(
            conv3d_block(128, 128),
            conv3d_block(128, 128),
        )

        self.conv3 = conv3d_block(128, 256, k=3, pad=(1, 1, 1), s=2)
        self.layer3 = nn.Sequential(
            conv3d_block(256, 256),
            conv3d_block(256, 256),
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x) + x
        x = self.conv2(x)
        x = self.layer2(x) + x
        x = self.conv3(x)
        x = self.layer3(x) + x

        return self.avg_pool(x)


class MViT_model(nn.Module):
    def __init__(self, num_encoders=4, adapter=True, adapter_type='nlp'):
        super(MViT_model, self).__init__()

        self.adapter = adapter
        self.adapter_type = adapter_type

        # Feature extractor (same CNN as before)
        self.cnn_feature_extractor = cnn_face()
        self.projection = nn.Sequential(
            nn.Linear(256, 96),  # Match MViT expected input dim
            nn.ReLU(),
        )

        # Load pretrained MViT
        base_mvit = torchvision.models.video.mvit_v2_s(weights="DEFAULT")
        # for p in base_mvit.parameters(): p.requires_grad = False
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, 96))  # 96 dim for MViT input
        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=96,
            spatial_size=(4, 4),
            temporal_size=4,
            rel_pos_embed=True,
        )

        # Build custom encoder with adapters
        layers = []
        for i in range(num_encoders):
            block = base_mvit.blocks[i]
            if adapter:
                if adapter_type == 'nlp':
                    layers.append(mvit_adapter_nlp(transformer_block=block))
                else:
                    layers.append(mvit_adapter_conv(transformer_block=block))
            else:
                for p in block.parameters():
                    p.requires_grad = True
                layers.append(block)

        # layers.append(base_mvit.norm)
        self.MViT_Encoder = nn.Sequential(*layers)

        # Output head
        self.classifier = nn.Linear(384, 2)

    def forward(self, x):
        b_s, n_f, C, H, W = x.shape
        # print(f"x shape {x.shape}")
        x = x.transpose(1, 2)
        # x = x.view(b_s * n_f, C, H, W)  # TODO: this seems to make it 4d, why?
        features = self.cnn_feature_extractor(x)
        # print(f"x shape after cnn: {features.shape}")
        features = features.flatten(2).transpose(1, 2)
        # print(f"x shape after flatten: {features.shape}")
        # print(f"features shape: {features.shape}")
        # features = features.view(b_s, n_f, 256)

        features = self.projection(features)
        # print(f"x shape after projection: {features.shape}")
        # x = x + self.pos_embedding

        # add positional encoding
        x = self.pos_encoding(features)
        # print(f"x after pos encoding {x.shape}")

        # pass patches through the encoder
        # thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size

        # print(f"x shape after pos embedding: {x.shape}")
        # x = self.MViT_Encoder(x)
        # thw = (n_f, H, W)
        thw = (4, 4, 4)

        for block in self.MViT_Encoder:
            if hasattr(block, 'forward'):
                # b_s, n_f, d = x.shape
                # thw = (64 - 1, 1, 1)
                # print(f"x shape {x.shape}")

                # print(f"x.shape: {x.shape}, thw shape: {thw}")
                x, thw = block(x, thw)
                # print("AFTER THW")
                # print(thw)
                # print("finished block")
            else:
                x = block(x)

        logits = self.classifier(x)
        return torch.mean(logits, 1)