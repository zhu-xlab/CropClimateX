import torch
import numpy as np
import torch.nn.functional as F
_ESP = 1e-10

class BaseTransform():
    def get_sample_dims(self, val, sample, channel_dim=None):
        """Get the dimensions of the sample tensor."""
        _dims = None
        if isinstance(self.min, (float, int)):
            pass
        elif val.dim() != sample.dim() and sample.dim() > 0 and any(torch.tensor(val.shape) > 1):
            if channel_dim:
                # if channel_dim set then use it to set the dims
                if channel_dim < 0:
                    channel_dim = sample.dim() + channel_dim
                    if channel_dim < 0:
                        raise ValueError(f"Channel dimension {channel_dim} is not valid for input tensor with shape {sample.shape}.")
                _dims = [1 if i != channel_dim else sample.shape[i] for i in range(sample.dim())]
            else:
                # if not take last dimension with same length as min
                _dims = [1 for _ in range(sample.dim() )]
                for i in range(sample.dim()-1, -1, -1):
                    if sample.shape[i] in val.shape:
                        _dims[i] = sample.shape[i]
                        break
        else:
            _dims = list(self.min.shape)
        return _dims


class MinMaxScaler():
    """Scale the sample tensor to the range [scale_min, scale_max]."""
    def __init__(self, data_min, data_max, scale_min, scale_max, channel_dim=-3) -> None:
        """data_min and data_max are the minimum and maximum values of the input tensor, scale_min and scale_max are the minimum and maximum values of the output tensor.
        channel_dim is the dimension of the channels.
        """
        if isinstance(scale_min, (int, float)):
            scale_min = torch.tensor(scale_min)
        if isinstance(scale_max, (int, float)):
            scale_max = torch.tensor(scale_max)
        if isinstance(scale_min, list) or isinstance(scale_min, np.ndarray):
            scale_min = torch.tensor(scale_min)
        if isinstance(scale_max, list) or isinstance(scale_max, np.ndarray):
            scale_max = torch.tensor(scale_max)

        if isinstance(data_min, (int, float)):
            data_min = torch.tensor(data_min)
        if isinstance(data_max, (int, float)):
            data_max = torch.tensor(data_max)
        if isinstance(data_min, list) or isinstance(data_min, np.ndarray):
            data_min = torch.tensor(data_min)
        if isinstance(data_max, list) or isinstance(data_max, np.ndarray):
            data_max = torch.tensor(data_max)
        if torch.eq(data_min, data_max).any():
            raise ValueError("data_min and data_max must not be equal.")
        self.data_min = data_min
        self.data_max = data_max
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.channel_dim = channel_dim

    def set_dims(self, sample):
        if not hasattr(self, "_dims"):
            # create dims with ones: [1,...,c,...,1]
            if self.channel_dim < 0:
                self.channel_dim = sample.dim() + self.channel_dim
                if self.channel_dim < 0:
                    raise ValueError(f"Channel dimension {self.channel_dim} is not valid for input tensor with shape {sample.shape}.")
            self._dims = [1 if i != self.channel_dim else sample.shape[i] for i in range(sample.dim())]

            if len(self._dims) > 0:
                # adjust the shape of the tensors
                if self.data_min.dim() != sample.dim() and self.data_min.dim() > 0:
                    self.data_min = self.data_min.view(*self._dims)
                if self.data_max.dim() != sample.dim() and self.data_max.dim() > 0:
                    self.data_max = self.data_max.view(*self._dims)
                
                if self.scale_min.dim() != sample.dim() and self.scale_min.dim() > 0:
                    self.scale_min = self.scale_min.view(*self._dims)
                if self.scale_max.dim() != sample.dim() and self.scale_max.dim() > 0:
                    self.scale_max = self.scale_max.view(*self._dims)

    def __call__(self, sample):
        # bring class values to the right shape
        self.set_dims(sample)
        res = ((sample - self.data_min) / (self.data_max - self.data_min)) * (self.scale_max - self.scale_min) + self.scale_min
        return res.float()

class Clip(BaseTransform):
    """Clip the sample tensor to the range [min, max] by using threshold or z-scores."""
    def __init__(self, min, max, method, channel_dim=-3, channels:list=[], **kwargs) -> None:
        # values (e.g. categorical values) can be skipped by defining them as inf
        self.channel_dim = channel_dim
        self.channels = channels # list of channels to be clipped
        if isinstance(min, list) or isinstance(min, np.ndarray):
            min = torch.tensor(min)
        if isinstance(max, list) or isinstance(max, np.ndarray):
            max = torch.tensor(max)
        if method == 'threshold':
            self.min = min
            self.max = max
        elif method == 'z-score': # compute threshold values assuming min, max are z-scores from a Gaussian
            if 'mean' not in kwargs or 'std' not in kwargs:
                raise ValueError('mean and std need to be given for z-score clipping')
            mean, std = kwargs['mean'], kwargs['std']
            self.min = min * std + mean
            self.max = max * std + mean
        else:
            raise ValueError(f'Clipper: unknown method {method}')

    def set_dims(self, sample):
        if not hasattr(self, "_dims"):
            self._dims = self.get_sample_dims(self.min, sample, self.channel_dim)
            if self._dims:
                if len(self.channels) > 0:
                    # replace the channels with the given ones
                    self._dims[self.channel_dim] = len(self.channels)
                self.min = self.min.view(*self._dims)
                self.max = self.max.view(*self._dims)

    def __call__(self, sample):
        self.set_dims(sample)
        # select the channels to be clipped
        if len(self.channels) > 0:
            clipped = torch.index_select(sample, self.channel_dim, torch.tensor(self.channels))
            clipped = torch.clamp(clipped, min=self.min, max=self.max) # clip to min and max for each channel
            # replace the channels with the clipped ones
            return torch.index_copy(sample, self.channel_dim, torch.tensor(self.channels), clipped)
        else:
            return torch.clamp(sample, min=self.min, max=self.max) # clip to min and max for each channel

class Binarize():
    def __init__(self, threshold=1) -> None:
        self.threshold = threshold

    def __call__(self, sample):
        return torch.where(sample >= self.threshold, 1, 0)

class ReplaceValue():
    def __init__(self, value='nan', replace_value=0) -> None:
        if isinstance(value, (int, float)):
            value = list(value)
        self.value = value
        self.replace_value = replace_value
    
    def __call__(self, sample):
        if self.value == 'nan':
            return torch.where(torch.isnan(sample), self.replace_value, sample)
        elif 'inf' in self.value:
            return torch.where(float(self.value), self.replace_value, sample)
        else:
            mask = torch.isin(sample, self.value)
            return torch.where(mask, self.replace_value, sample)

class ToTensor():
    """Convert the sample to a tensor."""
    def __call__(self, sample):
        return torch.tensor(sample)

class ToDType():
    """Convert the sample tensor to the specified dtype."""
    def __init__(self, dtype) -> None:
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype

    def __call__(self, sample):
        return sample.to(dtype=self.dtype)

class PermuteDims():
    """Permute the dimensions of the sample tensor."""
    def __init__(self, dims) -> None:
        self.dims = dims

    def __call__(self, sample):
        return sample.permute(self.dims)

class Norm():
    def __init__(self, p=2, dim=1) -> None:
        self.p = p
        self.dim = dim

    def __call__(self, sample):
        sample = torch.nn.functional.normalize(sample, self.p, self.dim)
        return sample

class NormalizedDifferenceIndex():
    """Normalized Difference Index (NDI) = (BAND1 - BAND2) / (BAND1 + BAND2)"""

    def __init__(self, band1: int, band2: int, dim: int = -3) -> None:
        """Band1 and band2 are the indices of the bands in the sample tensor in the dim dimension,
        dim -3 is the default, assumes [..., channel, H, W] tensors."""
        self.band1 = band1
        self.band2 = band2
        self.dim = dim

    def __call__(self, sample):
        channel1 = torch.index_select(sample, self.dim, torch.tensor(self.band1))
        channel2 = torch.index_select(sample, self.dim, torch.tensor(self.band2))
        return (channel1 - channel2) / (channel1 + channel2 + _ESP)

class KernelIndex():
    """Kernel Index = tanh(index)**2,
    e.g. kNDVI=tanh((BAND1 - BAND2) / (BAND1 + BAND2))**2"""

    def __init__(self, index) -> None:
        """index is a callable that takes a tensor as input and returns a tensor."""
        self.index = index

    def __call__(self, sample):
        index = self.index(sample)
        return torch.tanh(index)**2

class EnhancedVegetationIndex():
    """Enhanced Vegetation Index (EVI) = 2.5 * (BAND1 - BAND2) / (BAND1 + 6 * BAND2 - 7.5 * BAND3 + 1)"""
    def __init__(self, band1: int, band2: int, band3: int, g: float=2.5, c1: float=6, c2: float=7.5, l: float=1, dim: int=-3) -> None:
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        self.dim = dim
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.l = l
    
    def __call__(self, sample):
        channel1 = torch.index_select(sample, self.dim, torch.tensor(self.band1))
        channel2 = torch.index_select(sample, self.dim, torch.tensor(self.band2))
        channel3 = torch.index_select(sample, self.dim, torch.tensor(self.band3))
        return self.g * (channel1 - channel2) / (channel1 + self.c1 * channel2 - self.c2 * channel3 + self.l + _ESP)

class AppendTransform():
    """Append a transformation to the sample tensor."""

    def __init__(self, transform, dim=-3) -> None:
        """Transform is a callable that takes a tensor as input and returns a tensor, dim -3 is the
        default, assumes [..., channel, H, W] tensors."""
        self.transform = transform
        self.dim = dim

    def __call__(self, sample):
        channel = self.transform(sample)
        return torch.cat([sample, channel], dim=self.dim)

class ReplaceTransform():
    """Replace multiple channels with a transform."""

    def __init__(self, transform, channels, dim=-3) -> None:
        """Transform is a callable that takes a tensor as input and returns a tensor, dim -3 is the
        default, assumes [..., channel, H, W] tensors."""
        self.transform = transform # the transform to be applied
        self.channels = channels # the channels to be replaced
        self.dim = dim # the dimension of the channels

    def __call__(self, sample):
        channels = torch.index_select(sample, self.dim, torch.tensor(self.channels))
        new_channel = self.transform(sample)
        sample_dims = [i for i in range(sample.size(self.dim)) if i not in self.channels]
        if len(sample_dims) == 0:
            # input tensor has only the channels to be replaced
            return new_channel
        sample = torch.index_select(sample, self.dim, torch.tensor(sample_dims))
        return torch.cat([sample, new_channel], dim=self.dim)

class Flatten():
    """Flatten the sample tensor."""
    def __init__(self, start_dim=0, end_dim=-1) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim
    
    def __call__(self, sample):
        # Flatten the input tensor
        flattened_sample = torch.flatten(sample, self.start_dim, self.end_dim)
        return flattened_sample
    
class Hist():
    """create the histogram of the input tensor over the dimensions dim and return a normalized histogram."""
    def __init__(self, bins=256, range=(0, 1), dim=(-2, -1), normalize=True) -> None:
        self.bins = bins
        self.range = range
        if not isinstance(dim, (list, tuple)):
            dim = [dim]
        self.dim = dim
        self.normalize = normalize

    def __call__(self, sample):
        min, max = self.range
        bins = self.bins

        # get the dimension if negative
        dim = [d if d >= 0 else d + len(sample.shape) for d in self.dim]

        new_dims = [h for h in range(len(sample.shape)) if h not in dim]
        if len(new_dims) > 0:
            # reshape the tensor to [channels, -1], all dimensions not in dim are channels
            start_dim = len(new_dims)
            new_dims.extend(dim)
            sample = sample.permute(new_dims).flatten(start_dim=start_dim)
            org_shape = sample.shape[:-1]
            sample = sample.view(np.prod(sample.shape[:start_dim]), sample.shape[-1])
            # bring the values in the right format
            if not isinstance(bins, (list, tuple)):
                bins = [bins] * sample.shape[0]
            if not isinstance(min, (list, tuple)):
                min = [min] * sample.shape[0]
            if not isinstance(max, (list, tuple)):
                max = [max] * sample.shape[0]
            d_hist_list = []
            # compute the histogram for each channel
            for j in range(sample.shape[0]):
                hist = torch.histc(sample[j][~torch.isnan(sample[j])], bins=bins[j], min=min[j], max=max[j])
                if self.normalize:
                    hist = hist / (hist.sum() + _ESP)
                d_hist_list.append(hist)
            # bring back to wished shape
            hist = torch.stack(d_hist_list, axis=0)
            hist = hist.view(*org_shape, self.bins)

        else:
            sample = sample.flatten()
            hist = torch.histc(sample[~torch.isnan(sample)], bins=bins, min=min, max=max)
            hist = torch.unsqueeze(hist,0) # add dimension for channel, so shape is same as with channels
            if self.normalize:
                hist = hist / (hist.sum() + _ESP)

        return hist

class Mean():
    """calculate the mean of the input tensor along the dimensions dim."""
    def __init__(self, dim=(-2, -1)) -> None:
        self.dim = dim
    
    def __call__(self, sample):
        mean = torch.mean(sample, dim=self.dim)
        return mean
    
class Median():
    """calculate the median of the input tensor along the dimensions dim."""
    def __init__(self, dim=(-2, -1)) -> None:
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
    
    def set_dims(self, sample):
        if not hasattr(self, "_dims"):
            # if dim is negative then add the length of the tensor
            self.dim = [d if d >= 0 else d + len(sample.shape) for d in self.dim]
            self._dims = [i for i in range(sample.dim()) if i not in self.dim] + list(self.dim)

        return self._dims
            
    def __call__(self, sample):
        new_dims = self.set_dims(sample)
        sample = sample.permute(new_dims).flatten(start_dim=-len(self.dim))
        return torch.median(sample, dim=-1)[0]

class Mode():
    """calculate the mode of the input tensor along the dimensions dim."""
    def __init__(self, start_dim=-2) -> None:
        self.start_dim = start_dim
    
    def __call__(self, sample):
        flat_sample = sample.flatten(start_dim=self.start_dim)
        mode, _ = torch.mode(flat_sample, dim=-1)
        return mode.squeeze(-1)

class Variance():
    """calculate the variance of the input tensor along the dimensions dim."""
    def __init__(self, dim=(-2, -1)) -> None:
        self.dim = dim
    
    def __call__(self, sample):
        return torch.var(sample, dim=self.dim)

class Stack():
    """Stack multiple transformations in dimension dim."""
    def __init__(self, transforms: list, dim=-1) -> None:
        self.transforms = transforms
        self.dim = dim

    def __call__(self, sample):
        res = [transform(sample) for transform in self.transforms]
        return torch.stack(res, dim=self.dim)

class ToOneHot():
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes

    def __call__(self, sample):
        return F.one_hot(sample.long(), num_classes=self.num_classes).squeeze().float()

class Sample():
    """choose samples randomly along the last dimensions starting with start_dim.
    Resulting tensor last dimensions have shape n. If not enough values are available, then the values drawn with replacement.
    Numpy is used to select a random sample, so the same sample can be selected multiple times
    and invalid values can be excluded."""
    def __init__(self, n, start_dim=-2, include_only_valid=True, invalid_value=None) -> None:
        if not isinstance(n, (tuple, list)):
            n = [n]
        self.n = n
        self.start_dim = start_dim
        self.include_only_valid = include_only_valid
        self.invalid_value = invalid_value
    
    def select_elements_per_row(self, row):
        """random choice per row, only valid values are considered. If all values are invalid, then invalid are returned."""
        mask = (np.isnan(row)) | (row == self.invalid_value)
        probs = np.ones_like(row, dtype=np.float32)
        probs[mask] = 0 # assign 0 probability to invalid values
        if probs.sum() == 0: # if there are only invalid, then same probability for all
            probs = np.ones_like(row, dtype=np.float32)
        probs /= probs.sum()
        return np.random.choice(row, self.n, p=probs, replace=(np.prod(self.n) > (~mask).sum()))

    def __call__(self, sample):
        nr_samples = np.prod(self.n)
        sample = sample.flatten(start_dim=self.start_dim)
        if self.include_only_valid:
            sample = torch.Tensor(np.apply_along_axis(self.select_elements_per_row, axis=-1, arr=sample))
        else:
            indices = np.random.choice(sample.shape[-1], nr_samples, replace=(nr_samples > sample.shape[-1]))
            sample = sample[..., indices]
            sample = sample.view(*sample.shape[:-1], *self.n)
        return sample

class MaskChannel():
    """choose samples across channel according to values from channel_nr channel and drop this channel.
    The other channels will be filled with nans.
    """
    def __init__(self, dim=-1, channel_nr=-1, values=[], fill_value=torch.nan) -> None:
        self.dim = dim
        self.channel_nr = channel_nr
        if isinstance(values, (int, float)):
            values = [values]
        assert len(values) > 0, "At least one value must be provided."
        self.values = values
        self.fill_value = fill_value

    def __call__(self, sample):
        # select the variables and create mask
        type_values = sample.select(dim=self.dim, index=self.channel_nr)
        mask = torch.zeros_like(type_values, dtype=torch.bool)
        for value in self.values:
            mask = mask | torch.eq(type_values, value)
        mask = mask.unsqueeze(self.dim)
        # remove channel which holds the value dimension
        keep_mask = torch.ones(sample.size(self.dim), dtype=bool)
        keep_mask[self.channel_nr] = False
        sample = sample[:, keep_mask]
        # fill with fill_value
        sample = sample.masked_fill(~mask, self.fill_value)
            
        return sample