# iPoLNG

An unsupervised model for the integrative analysis of single-cell multiomics data, coded in the deep universal probabilistic program [Pyro](https://pyro.ai/).

## Dependency

This package relies on [PyTorch](https://pytorch.org/) to run. Please install the correct CUDA version that matches your Operating System on [the official website](https://pytorch.org/get-started/locally/). The exact version of the dependent packages can be found in ``requirements.txt``.

## Installation

Please install the ``iPoLNG`` package using the following command in the command line:

```{Shell}
pip install iPoLNG
```

## Usage

```{Python}
import iPoLNG
import torch
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor")
# torch.set_default_tensor_type("torch.FloatTensor")
W = iPoLNG.load_example_data()
model = iPoLNG.iPoLNG(W, num_topics=20, num_epochs=300)
result = model.Run(warmup_epochs=300, verbose=True)
```

To be completed.

## Reference

To be completed.