import torch as T
import torch.nn as nn


class RF:
    """Receptive Field
    Args:
        model(callable)     network to estimate receptive field from
        input_shape(tuple)  can be (C,W) or (C,W,H)
        n_sample(int)       number of sample to estimate gradient from
    """
    def __init__(self, model, input_shape, n_sample=10):
        pass

def get_name(module):
    return module.__class__.__name__


if __name__ == '__main__':
    from sparsemax import Sparsemax
    import torch
    import torch.nn as nn

    sparsemax = Sparsemax(dim=-1)
    softmax = torch.nn.Softmax(dim=-1)

    logits = torch.randn(2, 3, 5)
    logits.requires_grad = True
    print("\nLogits")
    print(logits)

    softmax_probs = softmax(logits)
    print("\nSoftmax probabilities")
    print(softmax_probs)

    sparsemax_probs = sparsemax(logits)
    print("\nSparsemax probabilities")
    print(sparsemax_probs)