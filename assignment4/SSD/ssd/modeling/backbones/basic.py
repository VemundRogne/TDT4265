import torch
from torch import nn
from torch import functional as F
from typing import Tuple, List


class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        conv_args = {"padding": 1, "kernel_size": 3}

        # Initialize network layers
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, stride=1, **conv_args),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, stride=1, **conv_args),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, stride=1, **conv_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=self.out_channels[0], stride=2, **conv_args),
            nn.ReLU()
        )

        self.conv_layer2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[0], out_channels=128, stride=1, **conv_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.out_channels[1], stride=2, **conv_args),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[1], out_channels=256, stride=1, **conv_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=self.out_channels[2], stride=2, **conv_args),
            nn.ReLU()
        )
        self.conv_layer4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[2], out_channels=128, stride=1, **conv_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.out_channels[3], stride=2, **conv_args),
            nn.ReLU()
        )
        self.conv_layer5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[3], out_channels=128, stride=1, **conv_args),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.out_channels[4], stride=2, **conv_args),
            nn.ReLU()
        )
        self.conv_layer6 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels[4], out_channels=128, stride=1, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=self.out_channels[5], stride=1, padding=0, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        The forward function should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out1 = self.conv_layer1(x)
        out2 = self.conv_layer2(out1)
        out3 = self.conv_layer3(out2)
        out4 = self.conv_layer4(out3)
        out5 = self.conv_layer5(out4)
        out6 = self.conv_layer6(out5)

        
        
        out_features = [
            out1,
            out2,
            out3,
            out4,
            out5,
            out6,
        ]

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

if __name__ == "__main__":
    # Check that the network can take a forward pass w/o crashing
    pass
