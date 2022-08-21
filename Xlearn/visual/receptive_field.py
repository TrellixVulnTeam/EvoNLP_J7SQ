import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['NumericRF']

def get_bounds(gradient):
    
    coords = gradient.nonzero(as_tuple=True) # get non-zero coords
    names = ['h', 'w']
    ret = {}
    
    for ind in [0, 1]:
        mini, maxi = coords[ind].min().item(), coords[ind].max().item()
        ret[names[ind]] = {'bounds' : (mini, maxi), 'range': maxi - mini}
    return ret


def plot_input_output(image, gradient, output_tensor, out_pos, coords, ishape, fname=None, add_text=False, use_out=None):
    fig = plt.figure(figsize=(13,8))
    ax = [plt.subplot2grid(shape=(4, 1), loc=(0, 0), rowspan=3),
    plt.subplot2grid(shape=(4, 1), loc=(3, 0))]

    # Plot RF
    oshape = output_tensor.squeeze().shape

    ax[0].set_title("Input shape %s"%(list(ishape)))
    
    if image is not None:
        if image.ndim == 4:
            image = image[0]
        
        if not isinstance(image, np.ndarray):
            print(image.shape)
            image = image.permute(1, 2, 0).numpy()

        ax[0].imshow(image)
    else:
        ax[0].imshow(gradient, cmap='copper', interpolation='nearest')

    # Draw RF bounds
    h0, w0, h, w = coords
    ax[0].add_patch(plt.Rectangle((w0-0.5, h0-0.5), w+1, h+1, fill=False, edgecolor='cyan'))

    # Plot channel mean of output
    ax[1].set_title("Output shape %s"%(list(oshape)))

    if use_out is not None:
        out = use_out
    else:
        out = np.random.rand(*oshape)

    ax[1].imshow(out.mean(1), cmap='binary', interpolation='nearest')
    ax[1].add_patch(plt.Rectangle((out_pos[1]-0.5, out_pos[0]-0.5), 1, 1, color='cyan'))

    if add_text:
        ax[0].text(w0+w+2, h0, 'Receptive Field', size=17, color='cyan', weight='bold')
        ax[1].text(out_pos[1]+1, out_pos[0], f'{list(out_pos)}', size=19, color='cyan', weight='bold')

    plt.tight_layout()

    if fname is not None:
        plt.savefig(fname, format='png')
        plt.close()
    else:
        plt.show()

class NumericRF:

    def __init__(self, model, input_shape):

        if not isinstance(input_shape, list):
            input_shape = list(input_shape)

        self.model = model.eval()

        if len(input_shape) == 3:
            input_shape = [1] + input_shape

        assert len(input_shape) == 4
        self.input_shape = input_shape

    def _remove_bias(self):
        for conv in self.model:
            conv.bias.data.fill_(0)
            conv.bias.requires_grad = False
        
    def get_rf_coords(self):
        h0, w0 = [self._info[k]['bounds'][0] for k in ['h', 'w']]
        h, w = [self._info[k]['range'] for k in ['h', 'w']]
        return h0, w0, h, w

    def heatmap(self, pos):
        self.pos = pos
        # Step 1: build computational graph
        self.inp = torch.zeros(self.input_shape, requires_grad=True)

        self.out = self.model(self.inp)

        # Step 2: zero out gradient tensor
        grad = torch.zeros_like(self.out)

        # Step 3: this could be any non-zero value
        grad[..., pos[0], pos[1]] = 1.0

        # Step 4: propagate tensor backward
        self.out.backward(gradient=grad)

        # Step 5: average signal over batch and channel + we only care about magnitute of signal
        self.grad_data = self.inp.grad.mean([0, 1]).abs().data
        
        self._info = get_bounds(self.grad_data)

        return self._info

    def info(self):
        return self._info

    def plot(self, image=None, fname=None, add_text=False, use_out=None):
        plot_input_output(image=image,
                          gradient=self.grad_data,
                          output_tensor=self.out,
                          out_pos=self.pos,
                          coords=self.get_rf_coords(),
                          ishape=self.input_shape,
                          fname=fname,
                          add_text=add_text,
                          use_out=use_out
            )
            