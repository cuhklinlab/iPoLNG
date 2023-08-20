# iPoLNG

This is an unsupervised model for the integrative analysis of single-cell multiomics data, coded in the deep universal probabilistic program [Pyro](https://pyro.ai/).

## Dependency

This package relies on [PyTorch](https://pytorch.org/) to run. Please install the correct CUDA version that matches your Operating System on [the official website](https://pytorch.org/get-started/locally/). The exact version of the dependent packages can be found in ``requirements.txt``.

## Installation

Please install the ``iPoLNG`` package using the following command in the command line:

```{Shell}
pip install iPoLNG
```

## Usage

The main function in this package is ``iPoLNG.iPoLNG``. The input consists of the following:

``W``: a dictionary with each value being ``torch.tensor``. Each value represents the count matrix for one data modality, with rows being cells and columns being features.

``num_topics``: the number of topics or latent factors.

``alpha_k``: The shape parameter for the inverse gamma distribution in the prior. The default value is ``1``.

``integrated_epochs``: The number of epochs to run in the stochastic variational inference algorithm for iPoLNG. The default value is ``3000``.

``warmup_epochs``: The number of epochs to run in the stochastic variational inference algorithm for PoLNG (as a warm-up step). The default value is ``3000``.

``lr``: The learning rate in the stochastic variational inference algorithm. The default value is ``0.1``. You may adjust it according to the characteristics of the data.

``seed``: The random seed used in the iPoLNG model. The default value is ``42``. 

``verbose``: A boolean value. Set to ``True`` if you would like to show the learning progress in the model. The default value is ``True``.

Here's the example code for a quick start:

```{Python}
from iPoLNG import iPoLNG
import torch
torch.set_default_tensor_type("torch.cuda.FloatTensor" if torch.cuda.is_available() else "torch.FloatTensor") # enable GPU acceleration if possible
W = iPoLNG.load_example_data()
model = iPoLNG.iPoLNG(W, num_topics=20, integrated_epochs=300, warmup_epochs=500, seed=42, verbose=True)
result = model.Run()
```

``result`` is a dictionary consisting of the following keys:

``L_est``: a cell by latent factor matrix, interpreted as the low-dimensional representation of the cells.

``Ls_est``: a dictionary consisting of several cell by latent factor matries, each interpreted as the low-dimensional representation of the cells in the corresponding individual data modality.

``Thetas_est``: a dictionary consisting of several latent factor by feature matries, each interpreted as the feature scores in the corresponding individual data modality.

``time``: the running time for the algorithm (including the warm-up step) in seconds.

``loss``: a dictionary consisting of the loss (i.e. negative ELBO in the stochastic variational inference algorithm) for individual data modality in the warm-up step and integrated data in the iPoLNG model.

``alpha0s``: a dictionary consisting of the hyperparameters that control the levels of noise across different data modalities in the iPoLNG model.

Please refer to [the vignette](https://github.com/cuhklinlab/iPoLNG/blob/main/vignette/vignette.ipynb) for the outputs of the example code.

## Reference

Zhang W and Lin Z (2023) iPoLNG — An unsupervised model for the integrative analysis of single-cell multiomics data. Front. Genet. 14:998504. doi: 10.3389/fgene.2023.998504

## Bug reports

If you encounter any problem when using this package, please feel free to open an [issue](https://github.com/cuhklinlab/iPoLNG/issues).
