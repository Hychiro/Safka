import numpy as np
import torch.nn as nn
from layers.Conv2dWithConstraint import Conv2dWithConstraint
from layers.LinearWithConstraint import LinearWithConstraint
from modules.SklearnStructure import SklearnStructure
from einops.layers.torch import Rearrange

class EegnetModel(nn.Module):
    def __init__(self, feature_extraction, head):
        super().__init__()

        self.feature_extraction = feature_extraction
        self.head = head

    def forward(self, x, targets):
        feature_extraction_result = self.feature_extraction(x)
        logits = self.head(feature_extraction_result)

        if targets is None:
            loss = None
        else:
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss


class Eegnet(SklearnStructure):
    def __init__(self,
                 n_convs=[8, 2, 16],
                 temporal_conv_size=32, temporal_conv_padding='same',
                 separable_conv_size=16, separable_conv_padding='same',
                 polling_size=[4, 8], pooling_kind=['avg', 'avg'], dropout_rate=0.5):
        super().__init__()

        self.n_convs = n_convs
        self.temporal_conv_size = temporal_conv_size
        self.temporal_conv_padding = temporal_conv_padding
        self.separable_conv_size = separable_conv_size
        self.separable_conv_padding = separable_conv_padding
        self.polling_size = polling_size
        self.pooling_kind = pooling_kind
        self.dropout_rate = dropout_rate

    def fit(self, X, y,
            lr='auto', iterations=1000, batchsize=64,
            device='cpu', validation=0, track=None, verbose=True):  # cpu

        pooling_kind_dict = {'avg': nn.AvgPool2d, 'max': nn.MaxPool2d}
        X = X.reshape(X.shape[0], X.shape[2], X.shape[3])  # Reshape to (batch_size, n_channels, n_times)
        n_channels = len(X[0])
        n_times = len(X[0, 0])
        self.temporal_conv_size = n_times // 4 if self.temporal_conv_size == 'auto' else self.temporal_conv_size

        n_classes = len(np.unique(np.array(y)))
        self.n_features = n_times

        # Temporal Conv
        self.rearrange = Rearrange('b c t -> b 1 c t')
        self.conv1 = nn.Conv2d(1, self.n_convs[0], (1, self.temporal_conv_size), bias=False,
                               padding=self.temporal_conv_padding)
        self.batch1 = nn.BatchNorm2d(self.n_convs[0])

        self.n_features = self.n_features - self.temporal_conv_size + 1 if self.temporal_conv_padding == 'valid' else self.n_features

        # Spatial Conv
        self.conv2 = Conv2dWithConstraint(self.n_convs[0], self.n_convs[0] * self.n_convs[1],
                                          (n_channels, 1), bias=False, groups=self.n_convs[0], max_norm=1)
        self.batch2 = nn.BatchNorm2d(self.n_convs[0] * self.n_convs[1])
        self.activation1 = nn.ELU()
        self.pool1 = pooling_kind_dict[self.pooling_kind[0]]((1, self.polling_size[0]))
        self.dropout1 = nn.Dropout(self.dropout_rate)

        self.n_features = self.n_features // self.polling_size[0]

        # Separable Conv
        self.separable_conv_size = self.n_features // 4 if self.separable_conv_size == 'auto' else self.separable_conv_size
        self.conv3 = nn.Conv2d(self.n_convs[0] * self.n_convs[1], self.n_convs[0] * self.n_convs[1],
                               (1, self.separable_conv_size), bias=False, padding=self.separable_conv_padding,
                               groups=self.n_convs[0] * self.n_convs[1])
        self.conv4 = nn.Conv2d(self.n_convs[0] * self.n_convs[1], self.n_convs[2], 1, bias=False)
        self.batch3 = nn.BatchNorm2d(self.n_convs[2])
        self.activation2 = nn.ELU()
        self.pool2 = pooling_kind_dict[self.pooling_kind[1]]((1, self.polling_size[1]))
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.n_features = self.n_features - self.separable_conv_size + 1 if self.separable_conv_padding == 'valid' else self.n_features
        self.n_features = self.n_features // self.polling_size[1]

        self.n_features = self.n_convs[2] * self.n_features

        self.temporal = nn.Sequential(self.rearrange, self.conv1, self.batch1)
        self.spatial = nn.Sequential(self.conv2, self.batch2, self.activation1, self.pool1, self.dropout1)
        self.separable = nn.Sequential(self.conv3, self.conv4, self.batch3, self.activation2, self.pool2, self.dropout2)

        self.feature_extraction = nn.Sequential(self.temporal, self.spatial, self.separable, nn.Flatten())
        self.head = nn.Sequential(
            LinearWithConstraint(in_features=self.n_features, out_features=n_classes, max_norm=0.25))

        self.model = EegnetModel(self.feature_extraction, self.head)

        self.train_model(X, y, batchsize=batchsize, lr=lr, iterations=iterations, device=device,
                                validation=validation, track=track, verbose=verbose)
        return self