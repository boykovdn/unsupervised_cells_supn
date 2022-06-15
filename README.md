# unsupervised_cells_supn
Experiments, evaluation, and models for "Cell Anomaly Localisation using Structured Uncertainty Prediction Networks"

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

## Setup

Use the environment file:

```shell
conda env create -f environment.yml
```

You might have to manually specify the cudatoolkit version, if you are using 
a newer Nvidia GPU (such as RTX3090).

## Training

This repository can be used to reproduce the results in the paper, but it can
also be used for experimenting on different data.

The training is done in two stages, first run:

```shell
python train.py --config example_train_mean.yaml
```

to train the encoder and mean-decoder. The paper parameters are in the default 
config yaml file. The training will take a few hours. Sometimes the VAE gets
stuck in a local minimum, and the decoded mean is a single constant value
across the image, so it is good to inspect the model training:

```shell
tensorboard --logdir ./experiments/mean_training --port 8008
```

Then check the "Images" tab in tensorboard by opening your browser and
navigating to localhost:8008 (or whichever port you have specified). The mean
should look right immediately after starting the training (there should be some
variation across the images).

Once the encoder and mean decoder are trained, train the supn decoder:

```shell
python train.py --config example_train_supn.yaml
```

The scheduler parameters are not used in the supn training, because the
learning rate does not change.

## Evaluation

Here we show how to compute the evaluation table from the paper. We will apply
the previously trained SUPN model to the fluorescent dataset to get the relevant
scores, and we will also apply the leading MVTec methods to the fluorescent
dataset, and first make sure that they achieve good performance on the MVTec
dataset itself.

### VAE on Fluorescent cells

To reproduce the performance of our model on the fluorescent dataset, run the
evaluate_fl.py script pointing to the supn and diag training configs:

```shell
python evaluate_fl.py --path_supn example_train_supn.yaml --path_diag example_train_diag.yaml
```

### MVTec-related

To work with the MVTec leading methods we use the implementation from 
https://github.com/rvorias/ind_knn_ad.

```shell
git clone https://github.com/rvorias/ind_knn_ad
```

Then ensure that the 'indad' folder is in your python packages path, you can
easily do this by creating a symlink to it as follows:

```shell
cd <path_to_environment>/lib/python3.8/site-packages
ln -s <path_to_repos>/ind_knn_ad/indad
```

You can check that you can access the models by opening the Python interpreter
and typing the following as an example.

```python
from indad.models import *
```

Now that the MVTec models are available, we can check that they score high on
the MVTec dataset itself.

To get the MVTec dataset, run the following commands in a directory of your
preference:

```shell
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz -C mvtec
```

The wget link was copied from the MVTec website, but you can replace it with
another mirror if it is broken, or simply download it from the website. The tar
command will unpack the archive contents into a new folder called 'mvtec'.

You can now run the mvtec evaluation script:

```shell
python evaluate_mvtec.py --mvpath ./mvtec --calc_mode mv_reproduce
python evaluate_mvtec.py --mvpath ./mvtec --calc_mode fl_eval
```

You will need to replace './mvtec' to the folder containing the MVTec dataset,
our script will automatically load the 'good' subfolders of every class as 
training or testing data, depending on the folder structure. The calc_mode arg
will determine whether to run the reproduction of the mvtec results for the 
mvtec methods (mv_reproduce), or the evaluation of the mvtec methods'
performance on the fluorescent cell dataset (fl_eval). The results will be 
printed in the console.

To evaluate our autoencoder on the MVTec dataset is a time-consuming task, as
a different mean and supn model has to be trained on every class. Some code for
the evaluation part exists in the evaluate_mvtec.py script, but is not complete.
