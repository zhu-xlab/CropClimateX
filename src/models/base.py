import torch.nn as nn

class TimeDistributed(nn.Module):
    """Wrap a module to apply it on every time step."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.size()[2:])  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        y = y.contiguous().view(x.size(0), x.size(1), *y.size()[1:])  # (org_axis1, org_axis2, output_size)

        return y  

def time_distribute(x, time_size=None):
    """Function to reshape the input tensor to apply a module on every time step and then reshape it back."""
    if time_size:
        return x.contiguous().view(x.size(0)//time_size, time_size, *x.size()[1:])
    else:
        return x.contiguous().view(-1, *x.size()[2:])

class SequentialNet(nn.Module):
    def __init__(self, blocks: list[nn.Module]|dict[str, nn.Module]):
        super().__init__()

        if isinstance(blocks, list):
            self.block = nn.Sequential(*blocks)
        elif isinstance(blocks, dict):
            self.block = nn.Sequential(*list(blocks.values()))
        else:
            raise ValueError('blocks must be a list or dict of nn.Modules')
            
    def forward(self,x):
        x = self.block(x)
        return x
