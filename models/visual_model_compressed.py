import torch
import torch.nn as nn
import torchvision
from models.adapter import vit_adapter_conv, vit_adapter_nlp



class conv2d_block(nn.Module):
    def __init__(self, in_channels, out_channels, pad='same', k=3, s=1):
        super(conv2d_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=pad, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class cnn_face(nn.Module):
    def __init__(self, ):
        super(cnn_face, self).__init__()

        self.conv1 = conv2d_block(1, 64, k=3, pad=(1, 1), s=1)
        self.layer1 = nn.Sequential(
            conv2d_block(64, 64),
            conv2d_block(64, 64),
        )

        self.conv2 = conv2d_block(64, 128, k=3, pad=(1, 1), s=1)
        self.layer2 = nn.Sequential(
            conv2d_block(128, 128),
            conv2d_block(128, 128),
        )

        self.conv3 = conv2d_block(128, 256, k=3, pad=(1, 1), s=1)
        self.layer3 = nn.Sequential(
            conv2d_block(256, 256),
            conv2d_block(256, 256),
        )

        self.conv4 = conv2d_block(256, 512, k=3, pad=(1, 1), s=1)
        self.layer4 = nn.Sequential(
            conv2d_block(512, 512),
            conv2d_block(512, 512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x) + x
        x = self.conv2(x)
        x = self.layer2(x) + x
        x = self.conv3(x)
        x = self.layer3(x) + x
        x = self.conv4(x)
        x = self.layer4(x) + x

        return self.avg_pool(x)


class ViT_compressed_model(nn.Module):
    def __init__(self, num_encoders, adapter, adapter_type):
        super(ViT_compressed_model, self).__init__()

        self.num_encoders = num_encoders
        self.adapter = adapter
        self.adapter_type = adapter_type

        self.projection = nn.Sequential(
            nn.Linear(784, 768),
            nn.ReLU(),
        )

        # Load imagenet ViT Base 16
        vit_b_16 = torchvision.models.vit_b_16(pretrained=False)
        # extract encoder alone and discard CNN (patchify + linear projection) feature extractor, classifer head
        # Refer Encoder() class in https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
        vit = vit_b_16.encoder
        # add learnable positional embedding for 64 tokens (dim=768). Original ViT uses 196+1 tokens for position embedding
        self.pos_embedding = nn.Parameter(torch.empty(1, 64, 768).normal_(std=0.02))

        # start building ViT encoder layers
        layer_list = []

        for i in range(self.num_encoders):
            for p in vit.layers[i].parameters(): p.requires_grad = True
            layer_list.append(vit.layers[i])

        # add final encoder layer norm
        layer_list.append(nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True))

        # assign models for forward pass
        self.cnn_feature_extractor = cnn_face()
        self.ViT_Encoder = nn.Sequential(*layer_list)

        # classsification (deception detection) head
        self.classifier = nn.Sequential(
            nn.Linear(768, 2),
        )

    def forward(self, x):

        # Feature extraction for all 64 images
        b_s, no_of_frames, C, H, W = x.shape
        features = torch.reshape(x, (b_s * no_of_frames, C, H, W))
        # features = self.cnn_feature_extractor(features)
        features = torch.reshape(features, (b_s, no_of_frames, 784))

        # projection for vit encoder + position embedding
        # print(f"features before {features.shape}")
        projections = self.projection(features) + self.pos_embedding
        # feature extraction with transformer encoders
        vit_features = self.ViT_Encoder(projections)
        # deception detection
        logits = self.classifier(vit_features)

        # average predictions for every token
        return torch.mean(logits, 1)
