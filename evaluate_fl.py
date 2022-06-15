import torch
import yaml
from dataset import FullFrames, MvtecClass
import pandas as pd
from utils import (
        retile_mask,
        L2_model_wrap,
        Diag_model_wrap,
        Restoration_model_wrap,
        conn_to_nonzeros,
        load_model,
        get_log_prob_from_sparse_L_precision)
from tqdm.auto import tqdm
import numpy as np
from transforms import (rescale_to, SigmoidScaleShift)
import torchvision.transforms as tv_transforms
from utils import (apply_sparse_chol_rhs_matmul, 
        mahalanobis_dist)
from sklearn.metrics import (
        roc_curve, 
        roc_auc_score,
        auc,
        precision_recall_curve)
from model_blocks import (DiagChannelActivation, 
        IPE_autoencoder_mu_l)
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def scoring_table(true_labels, predicted_labels, plot_curves=False, model_label=None):
    r"""
    Reports a couple of evaluation metrics.

    Args:
        :true_labels: np.array, [N,] all values are either 0 or 1, the ground
            truth labels.
        :predicted_labels: np.array, [N,], vals are either 0 or 1, the
            predicted labels.
    """
    metrics = {}

    # AUPRC
    precision, recall, threshs = precision_recall_curve(true_labels, 
            predicted_labels)
    auprc = auc(recall, precision)

    # AUROC
    auroc = roc_auc_score(true_labels, predicted_labels)

    metrics['auprc'] = auprc
    metrics['auroc'] = auroc

    if plot_curves:
        fpr, tpr, thresh = roc_curve(true_labels, predicted_labels)
        #precision, recall, thresh = precision_recall_curve(true_labels, predicted_labels)
        plt.plot(fpr, tpr, color="blue", label=model_label)
        #plt.plot(precision, recall, color="orange")

    return metrics

def vae_dataset_likelihood(dset, model, device=0):
    r"""
    Calculate the likelihood of the dataset under the model. The likelihood of
    X is the sum of the likelihoods of the individual datapoints.

    Inputs:
        :dset: Dataset object
        :model: VAE generative model outputs distribution parameters.
        :device: int
    """
    total_log_l = 0

    for dpoint,mask in tqdm(dset, desc="Calculating dataset likelihood..."):
        dpoint = dpoint[None,].to(device)

        x_mu, x_chol, _, _ = model(dpoint)

        total_log_l += get_log_prob_from_sparse_L_precision(
                dpoint,
                x_mu,
                model.connectivity,
                x_chol[:,0,None,],
                x_chol[:,1:]).detach().item() / len(dset)

    return total_log_l

def vae_model_scoring(dset_n, dset_a, model, scoring_func, device=0,
        samples=100, model_label=None):
    r"""
    Calculates the auroc metric, or how well the model can detect anomalies, or
    classify class "normal" from class "anomalous".

    Args:
        :samples: int, how many samples from q_z to average in order to get the
            estimate of the expected score.
    """
    def get_scores(dset, model, scoring_func, plot_convergence=False):
        scores = []
        deltas_cumul = [] # TODO Remove
        for dpoint, mask in tqdm(dset):
            dpoint = dpoint[None,].to(device) # [B,1,H,W]
            mask = mask[None]

            score_samples = []
            for sid in tqdm(range(samples)):
                x_mu, x_chol, _, _ = model(dpoint)
                # In range [0,1], binary classification task between in or out of
                # distribution.
                score_sample = scoring_func(x_mu, x_chol, dpoint, mask, model.connectivity)
                score_samples.append(score_sample)

            score = np.array(score_samples).mean()
            scores.append(score)
            #deltas_cumul.append(torch.Tensor(model.deltas)) # TODO Remove

        # TODO Temporary, to get convergence graphs
        if plot_convergence:
            deltas_cumul = torch.stack(deltas_cumul) # [DP, Optim_steps]
            deltas_means = deltas_cumul.mean(0)
            deltas_stds = deltas_cumul.std(0)
            plt.fill_between(range(len(deltas_stds)), deltas_means - deltas_stds/2, 
                    deltas_means + deltas_stds/2, alpha=0.5)
            plt.plot(range(len(deltas_means)), deltas_means, linestyle="dotted", label=model_label)

        return scores

    anom_labels = np.ones(len(dset_a))
    anom_scores = np.array(get_scores(dset_a, model, scoring_func, plot_convergence=False)) # TODO
    norm_labels = np.zeros(len(dset_n))
    norm_scores = np.array(get_scores(dset_n, model, scoring_func))

    labels = np.concatenate((anom_labels, norm_labels))
    scores = np.concatenate((anom_scores, norm_scores))

    metrics = scoring_table(labels, scores, model_label=model_label)

    return metrics

def scoring_func_outliers(x_mu, x_chol, x, mask, TILE_FACTOR, saliency_map=None, conn=1):
    r"""
    This function tiles the image and returns the highest score, which would
    correspond to the outlier if there is one.
    """
    tile_size = 2**TILE_FACTOR

    B,_,H,W = x.shape
    assert B == 1, "Works only for batch 1 currently..."

    _, tile_coords = retile_mask(mask.float(), TILE_FACTOR)

    tile_scores = []
    with torch.no_grad():
        for tile_idx in range(tile_coords.shape[-1]):

            coord_h, coord_w, _ = tile_coords[:,tile_idx].long()
            selection = torch.zeros_like(x).bool()
            slice_h = slice(coord_h, coord_h + tile_size)
            slice_w = slice(coord_w, coord_w + tile_size)
            selection[..., slice_h, slice_w] = True

            whitened_mah = mahalanobis_dist(x, x_mu, x_chol, conn=conn)
            whitened_mah = whitened_mah[selection].sum()

            tile_scores.append([ coord_h.item(), coord_w.item(), 
                whitened_mah.item() ])

    tile_scores = np.stack(tile_scores) # [N, 3]

    return tile_scores[:,-1].max()

def main():
    PATH_HEALTHY_RAW = "./fl_dataset/healthy_test/raw"
    PATH_HEALTHY_GT = "./fl_dataset/healthy_test/mask"

    PATH_INFECTED_RAW = "./fl_dataset/infected/raw"
    PATH_INFECTED_GT = "./fl_dataset/infected/mask"

    PATH_SUPN = "./example_train_supn.yaml"
    PATH_DIAG = "./example_train_diag.yaml"
    # TILE_FACTOR is basically the size of a 2dpool-ing neighbourhood.
    TILE_FACTOR = 2
    
    FL_FIXED_VAR = 0.11
    
    DEVICE='0'
    device_name = 'cuda:{}'.format(DEVICE) if type(DEVICE) is int else 'cpu'

    parser = argparse.ArgumentParser(description="Script to evaluate SUPN, Diag, L2 performance on fluo data.")
    parser.add_argument("--path_supn", type=str, help="Path to SUPN training config", required=True)
    parser.add_argument("--path_diag", type=str, help="Path to DIAG training config")
    args = parser.parse_args()

    #### TODO To load data function
    tforms = tv_transforms.Compose([
        lambda x : rescale_to(x, to=(-1,1)),
        tv_transforms.GaussianBlur(5, sigma=2.0)
        ])
    tforms_joint = None

    # Test data
    dset_healthy = FullFrames(PATH_HEALTHY_RAW, PATH_HEALTHY_GT,
            raw_transforms=tforms,
            joint_transforms=tforms_joint,
            apply_joint_first=False)
    dset_infected = FullFrames(PATH_INFECTED_RAW, PATH_INFECTED_GT,
            raw_transforms=tforms,
            joint_transforms=tforms_joint,
            apply_joint_first=False)
    ####

    #### TODO To load model type function, parametrized for resto somehow
    model_diag = load_model(args.path_diag, map_location=torch.device(device_name), 
            pretrained_mean=False)
    # Wrap around the diag model to change the diagonal vector prediciton into
    # a valid Cholesky decomposition for the downstream evaluation (so it can
    # be shared with SUPN).
    model_diag_fwd = Diag_model_wrap(model_diag)
    model_supn = load_model(args.path_supn, map_location=torch.device(device_name), 
            pretrained_mean=False)
    # Does not matter which model is taken as the L2 model, both have equally
    # good mean predictors, and the covariance is an identity.
    model_l2_fwd = L2_model_wrap(FL_FIXED_VAR, model_supn)

    scoring_func = lambda x_mu, x_chol, x, mask, conn : scoring_func_outliers(
            x_mu, x_chol, x, mask, TILE_FACTOR, conn=conn)

    model_resto_supn = Restoration_model_wrap(model_supn)
    model_resto_diag = Restoration_model_wrap(model_diag_fwd)
    model_resto_l2 = Restoration_model_wrap(model_l2_fwd)
    ####

    # For restoration-based methods there is no sampling because the z becomes
    # an optimized point estimate (samples = 1).
    metrics_resto_supn = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_supn, scoring_func, device=device_name, samples=1, model_label="SUPN (Restoration)")
    metrics_resto_l2 = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_l2, scoring_func, device=device_name, samples=1, model_label="L2 (Restoration)")
    metrics_resto_diag = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_diag, scoring_func, device=device_name, samples=1, model_label="Diag (Restoration)")
    ############## /TODO

    # TODO Check if below is necessary
    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    # Sampling is done for non-restorative methods (samples > 1).
    metrics_supn = vae_model_scoring(dset_healthy, dset_infected, 
            model_supn, scoring_func, device=device_name, samples=100)
    metrics_diag = vae_model_scoring(dset_healthy, dset_infected, 
            model_diag_fwd, scoring_func, device=device_name, samples=100)
    metrics_l2 = vae_model_scoring(dset_healthy, dset_infected, 
            model_l2_fwd, scoring_func, device=device_name, samples=100)

    metrics = {
            'resto_supn' : metrics_resto_supn,
            'resto_diag' : metrics_resto_diag,
            'resto_l2' : metrics_resto_l2,
            'supn' : metrics_supn,
            'diag' : metrics_diag,
            'l2' : metrics_l2,
            }

    df_columns = {
            'model_name' : [],
            'auprc' : [],
            'auroc' : [],
            }
    for name, metric_table in metrics.items():
        auprc = metric_table['auprc']
        auroc = metric_table['auroc']

        df_columns['model_name'].append(name)
        df_columns['auprc'].append(auprc)
        df_columns['auroc'].append(auroc)

    df_results = pd.DataFrame(df_columns)
    print(df_results)
    #with open("./latex_table_results.txt", "w+") as fout:
    #    fout.write(df_results.to_latex(index=False))

if __name__ == "__main__":
    main()
