import torch.nn as nn
from pytorch_wavelets import DWT1DForward
import torch
from torch.autograd import Function
import pywt
from pytorch_wavelets.dwt import lowlevel


class DWT1DFor2DForward(nn.Module):
    """Performs a 1d DWT Forward decomposition of an 2d image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave="db1", mode="zero", dim=3):
        super().__init__()
        self.dwt = DWT1DForward(wave=wave, mode=mode, J=1)
        self.dim = dim

    def forward(self, x):
        # B, C, H, W
        if self.dim == 2:
            x = x.permute(0, 1, 3, 2)
        s = x.shape
        x = x.reshape(s[0], s[1] * s[2], s[3])
        low, [high] = self.dwt(x)
        low = low.reshape(s[0], s[1], s[2], -1)
        high = high.reshape(s[0], s[1], s[2], -1)
        if self.dim == 2:
            low = low.permute(0, 1, 3, 2)
            high = high.permute(0, 1, 3, 2)
        return low, high


class AFB1DFor2D(Function):
    """Does a single level 1d wavelet decomposition of an input.

    Needs to have the tensors in the right form. Because this function defines
    its own backward pass, saves on memory by not having to save the input
    tensors.

    Inputs:
        x (torch.Tensor): Input to decompose
        h0: lowpass
        h1: highpass
        mode (int): use mode_to_int to get the int code here

    We encode the mode as an integer rather than a string as gradcheck causes an
    error when a string is provided.

    Returns:
        x0: Tensor of shape (N, C, L') - lowpass
        x1: Tensor of shape (N, C, L') - highpass
    """

    @staticmethod
    def forward(ctx, x, h0, h1, mode, dim=3):
        mode = lowlevel.int_to_mode(mode)
        # Save for backwards
        ctx.save_for_backward(h0, h1)
        ctx.shape = x.shape[-2:]
        ctx.mode = mode
        ctx.dim = dim

        lohi = lowlevel.afb1d(x, h0, h1, mode=mode, dim=dim)
        s = lohi.shape
        lohi = lohi.reshape(s[0], -1, 2, s[-2], s[-1])
        lo, hi = torch.unbind(lohi, dim=2)
        return lo.contiguous(), hi.contiguous()

    @staticmethod
    def backward(ctx, dx0, dx1):
        dx = None
        if ctx.needs_input_grad[0]:
            mode = ctx.mode
            h0, h1 = ctx.saved_tensors
            dim = ctx.dim

            dx = lowlevel.sfb1d(dx0, dx1, h0, h1, mode=mode, dim=dim)

            # Check for odd input
            dx = dx[..., : ctx.shape[-2], : ctx.shape[-1]]  # type: ignore

        return dx, None, None, None, None, None


class DWT1DFor2DForwardFast(nn.Module):
    """Performs a 1d DWT Forward decomposition of an 2d image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays (h0, h1)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
    """

    def __init__(self, wave="db1", mode="zero", dim=3):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)  # type: ignore
        if isinstance(wave, pywt.Wavelet):  # type: ignore
            h0, h1 = wave.dec_lo, wave.dec_hi
        elif len(wave) == 2:
            h0, h1 = wave[0], wave[1]

        # Prepare the filters
        filts = lowlevel.prep_filt_afb1d(h0, h1)
        self.register_buffer("h0", filts[0])
        self.register_buffer("h1", filts[1])
        self.mode = mode
        self.dim = dim

    def forward(self, x):
        """Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        mode = lowlevel.mode_to_int(self.mode)

        yl, yh = AFB1DFor2D.apply(x, self.h0, self.h1, mode, self.dim)  # type: ignore

        return yl, yh


class DWTDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, dim=3, J=1):
        super().__init__()
        self.J = J
        self.wt = DWT1DFor2DForward(wave="haar", mode="zero", dim=dim)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 2**J, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        for _ in range(self.J):
            yl, yh = self.wt(x)
            x = torch.cat([yl, yh], dim=1)
        x = self.conv_bn_relu(x)
        return x
