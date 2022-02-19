# unsupervised_cells_supn
Experiments, evaluation, and models for "Cell Anomaly Localisation using Structured Uncertainty Prediction Networks"

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

### Training

This repository can be used to reproduce the results in the paper, but it can
also be used for experimenting on different data.

The training is done in two stages, first run:

```shell
python train.py --config example_train_mean.yaml
```

to train the encoder and mean-decoder. The paper parameters are in the default 
config yaml file. The training will take a few hours.

Once the encoder and mean decoder are trained, train the supn decoder:

```shell
python train.py --config example_train_supn.yaml
```

The scheduler parameters are not used in the supn training, because the
learning rate does not change.
