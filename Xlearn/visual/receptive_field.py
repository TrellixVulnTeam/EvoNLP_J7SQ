import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['NumericRF']

def get_bounds(gradient):
    
    coords = gradient.nonzero(as_tuple=True)[0] # get non-zero coords
    mini, maxi = coords.min().item(), coords.max().item()
    return {'bounds': (mini, maxi), 'range': maxi - mini}

def visual(gradient, oshape, out_pos, coords, ishape):
    fig = plt.figure(figsize=(8,4))
    gs = fig.add_gridspec(2,1, hspace=0.01, wspace=0.0, height_ratios=[0.9,0.1])
    inp_ax = fig.add_subplot(gs[0,0])
    inp_ax.set_title("Input shape %s"%(list(ishape)))
    inp_ax.get_yaxis().set_visible(False)
    height = len(gradient)>>4
    inp_ax.imshow(gradient.repeat(height,1), cmap='copper', interpolation='nearest')
    inp_ax.add_patch(plt.Rectangle(
        (coords[0]-0.5, 0-0.5), coords[1]+1, height+1, fill=False, edgecolor='cyan'
    ))
    # Plot channel mean of output
    out_ax = fig.add_subplot(gs[1,0])
    out_ax.set_title("Output shape %s"%(list(oshape)))
    out_ax.get_yaxis().set_visible(False)
    out = torch.zeros(*oshape).mean([0,1])
    out_ax.imshow(out.repeat(1,1), cmap='binary', interpolation='nearest')
    out_ax.add_patch(plt.Rectangle((out_pos-0.5, 0-0.5), 1, 1, color='cyan'))
    plt.tight_layout()
    plt.show()

class NumericRF:

    def __init__(self, model, input_shape):

        if not isinstance(input_shape, list):
            input_shape = list(input_shape)
        self.model = model.eval()
        self.input_shape = input_shape

    def _remove_bias(self):
        for conv in self.model:
            conv.bias.data.fill_(0)
            conv.bias.requires_grad = False
        
    def get_rf_coords(self):
        return self._info['bounds'][0], self._info['range']

    def heatmap(self, pos):
        self.pos = pos
        # Step 1: build computational graph
        self.inp = torch.zeros(self.input_shape, requires_grad=True)
        self.out = self.model(self.inp)
        self.oshape = list(self.out.shape)

        # Step 2: zero out gradient tensor
        grad = torch.zeros_like(self.out)

        # Step 3: this could be any non-zero value
        grad[..., pos] = 1.0

        # Step 4: propagate tensor backward
        self.out.backward(gradient=grad)

        # Step 5: average signal over batch and channel + we only care about magnitute of signal
        self.grad_data = self.inp.grad.mean([0, 1]).abs().data
        self._info = get_bounds(self.grad_data)
        return self._info

    @property
    def info(self):
        return self._info

    def plot(self):
        visual(
            gradient=self.grad_data, oshape=self.oshape, out_pos=self.pos, coords=self.get_rf_coords(), ishape=self.input_shape 
        )
            