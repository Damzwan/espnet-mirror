# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Style encoder of GST-Tacotron."""
from typeguard import check_argument_types
from typing import Sequence, Tuple, Any

import torch

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention as BaseMultiHeadedAttention,  # NOQA
)


class VAE(torch.nn.Module):
    def __init__(
            self,
            idim: int = 80,
            conv_layers: int = 6,
            conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
            conv_kernel_size: int = 3,
            conv_stride: int = 2,
            gru_layers: int = 1,
            gru_units: int = 128,
            vae_z_dim=16,
            hidden_size=256
    ):
        """Initilize global style encoder module."""
        assert check_argument_types()
        super(VAE, self).__init__()
        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )

        self.FC_h = torch.nn.Linear(gru_units, hidden_size)
        self.FC1 = torch.nn.Linear(hidden_size, vae_z_dim)
        self.FC2 = torch.nn.Linear(hidden_size, vae_z_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, speech: torch.Tensor) -> Tuple[Any, Any, Any]:
        ref_embs = self.ref_enc(speech)
        ref_embs = ref_embs.unsqueeze(1)

        hidden = self.relu(self.FC_h(ref_embs))
        mu, log_var = self.FC1(hidden), self.FC2(hidden)

        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_var


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
            self,
            idim=80,
            conv_layers: int = 6,
            conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
            conv_kernel_size: int = 3,
            conv_stride: int = 2,
            gru_layers: int = 1,
            gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        assert check_argument_types()
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
                len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                                   gru_in_units - conv_kernel_size + 2 * padding
                           ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs
