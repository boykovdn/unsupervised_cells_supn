import torch
import yaml
from dataset import FullFrames, MvtecClass
import pandas as pd
from structured_uncertainty.utils import (
        retile_mask,
        get_combi_metric)
from tqdm.auto import tqdm
import numpy as np
from transforms import (rescale_to, SigmoidScaleShift)
import torchvision.transforms as tv_transforms
from utils import apply_sparse_chol_rhs_matmul
from sklearn.metrics import (
        roc_curve, 
        roc_auc_score,
        auc,
        precision_recall_curve)
from model_blocks import (DiagChannelActivation, 
        IPE_autoencoder_mu_l)
import argparse
from pathlib import Path

def mahalanoubis_dist(x,mu,sparseL, conn=1):
    r"""
    r.T cov-1 r
    r.T (L.T L) r
    (Lr).T (Lr)
    """
    Lr = apply_sparse_chol_rhs_matmul((x-mu),
            log_diag_weights=sparseL[:,0].unsqueeze(1),
            off_diag_weights=sparseL[:,1:],
            local_connection_dist=conn,
            use_transpose=True)

    return Lr.square()

def scoring_table(true_labels, predicted_labels):
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

    return metrics

def vae_model_scoring(dset_n, dset_a, model, scoring_func, device=0):
    r"""
    Calculates the auroc metric, or how well the model can detect anomalies, or
    classify class "normal" from class "anomalous".
    """
    def get_scores(dset, model, scoring_func):
        scores = []
        for dpoint, mask in tqdm(dset):
            dpoint = dpoint[None,].to(device) # [B,1,H,W]
            mask = mask[None]
            x_mu, x_chol, _, _ = model(dpoint)
            # In range [0,1], binary classification task between in or out of
            # distribution.
            score = scoring_func(x_mu, x_chol, dpoint, mask)
            scores.append(score)

        return scores

    anom_labels = np.ones(len(dset_a))
    anom_scores = np.array(get_scores(dset_a, model, scoring_func))
    norm_labels = np.zeros(len(dset_n))
    norm_scores = np.array(get_scores(dset_n, model, scoring_func))

    labels = np.concatenate((anom_labels, norm_labels))
    scores = np.concatenate((anom_scores, norm_scores))

    metrics = scoring_table(labels, scores)

    return metrics

def scoring_func_outliers(x_mu, x_chol, x, mask, TILE_FACTOR, saliency_map=None):
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

            whitened_mah = mahalanoubis_dist(x, x_mu, x_chol)
            whitened_mah = whitened_mah[selection].sum()

            tile_scores.append([ coord_h.item(), coord_w.item(), 
                whitened_mah.item() ])

    tile_scores = np.stack(tile_scores) # [N, 3]

    return tile_scores[:,-1].max()

def conn_to_nonzeros(conn):
    neighbourhood_size = 2*conn + 1
    nonzeros = (neighbourhood_size**2) // 2 + 1

    return nonzeros

class Restoration_model_wrap:

    def __init__(self, model, step=0.01, n_steps=100):
        self.model = model
        self.connectivity = model.connectivity
        self.step = step
        self.n_steps = n_steps

    def zero_grad(self):
        self.model_diag.zero_grad()

    def optimize_z(self, dpoint, model, step=0.01, n_steps=100):
        r"""
        """
        conn = self.connectivity

        x_mu, x_chol, z_mu, z_logvar = model(dpoint)

        model.zero_grad()
        z_mu = z_mu.detach()
        z_logvar = z_logvar.detach()

        # Sample from dist for the initialization point.
        z_ = z_mu + z_logvar.exp() * torch.randn_like(z_logvar)
        z_.requires_grad = True

        optimizer = torch.optim.Adam([z_], lr=step)
        deltas = []
        for step_idx in range(n_steps):
            x_mu = model.mu_decoder(z_)
            x_chol = model.var_decoder(z_)
            mah = mahalanoubis_dist(dpoint, x_mu, x_chol, conn=conn)
            mah.sum().backward()
            optimizer.step()
            with torch.no_grad():
                deltas.append((z_.grad * step).square().sum())
            optimizer.zero_grad()
            #deltas.append(delta_z.detach().square().sum())
            #z_ = z_ + delta_z

        return z_.detach()

    def __call__(self, x):
        r"""
        Optimize the latent representation z to give the best image
        reconstruction via gradient descent.
        """
        model = self.model
        conn = self.connectivity

        z_optim = self.optimize_z(x, model)
        x_mu = model.mu_decoder(z_optim)
        x_chol = model.var_decoder(z_optim)

        return x_mu, x_chol, z_optim, None

class Diag_model_wrap:

    def __init__(self, model_diag):
        self.model_diag = model_diag
        self.connectivity = model_diag.connectivity

    def zero_grad(self):
        self.model_diag.zero_grad()

    def mu_decoder(self, z):
        return self.model_diag.mu_decoder(z)

    def var_decoder(self, z):
        # For diagonal model, output is a vector representing the log-diagonal 
        # of the covariance matrix.
        x_logvar = self.model_diag.var_decoder(z)
        device = z.device
        B,_,H,W = x_logvar.shape
        nonzeros = conn_to_nonzeros(self.connectivity)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)
        x_chol[:,0] = -0.5*x_logvar[:,0]

        return x_chol

    def __call__(self, x):
        r"""
        Output of the diagonal model has only one channel and is interpreted as
        the log-diagonal of the covariance matrix. In order to be interpreted
        as the log-diag of the Cholesky of the precision matrix, we need to
        multiply by -0.5. The minus is for inverting the value (to precision) and 
        the 0.5 for taking the square-root (Cholesky of a diagonal precision).
        """
        model_diag = self.model_diag
        conn = self.connectivity

        x_mu, x_logvar, z_mu, z_logvar = model_diag(x)
        B,_,H,W = x_mu.shape

        device = x_mu.device
        nonzeros = conn_to_nonzeros(conn)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)
        x_chol[:,0] = -0.5*x_logvar[:,0]

        return x_mu, x_chol, z_mu, z_logvar

class L2_model_wrap:

    def __init__(self, model_l2):
        self.model_l2 = model_l2
        self.connectivity = model_l2.connectivity

    def zero_grad(self):
        self.model_l2.zero_grad()

    def mu_decoder(self, z):
        return self.model_l2.mu_decoder(z)

    def var_decoder(self, z, H=128, W=128):
        conn = self.connectivity
        B,_ = z.shape
        device = z.device
        nonzeros = conn_to_nonzeros(conn)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)

        # Here x_logvar is completely discarded. It doesn't matter if the model was
        # trained to predict something meaningful there or not.
        return x_chol

    def __call__(self, x):
        r"""
        Output of the L2 model is really only the mean, but it is interpreted as
        a spherical standard MVN (identity matrix as covariance). To reinterpret it
        as a cholesky output, we do the same as in 'diag_model_wrap', but it turns
        out that the cholesky of the precision is also an identity matrix. Since 
        the evaluation functions expect a Cholesky output with log-diagonal values,
        we pass a matrix of zeros, where the off-diagonals are interpreted as
        actual zeros, but the diagonal will be exponentiated later to have ones.
        """
        model_l2 = self.model_l2
        conn = self.connectivity

        x_mu, x_logvar, z_mu, z_logvar = model_l2(x)
        B,_,H,W = x_mu.shape

        device = x_mu.device
        nonzeros = conn_to_nonzeros(conn)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)

        # Here x_logvar is completely discarded. It doesn't matter if the model was
        # trained to predict something meaningful there or not.
        return x_mu, x_chol, z_mu, z_logvar

def load_model(config_path, input_size=(1,128,128), map_location='cuda:0'):
    r"""
    Initialize model architecture from config file, then load the trained state
    dict.

    Args:
        :config_path: Path or str, to the config file specifying the model params.
        :input_size: tuple, (c,h,w) what size the input images are, channels
            including (even though at this point model only works with
            grayscale).
        :map_location: str or torch.device, where to put the model.
    """
    yaml_dict = {}
    with open(config_path, "r") as yaml_config:
        yaml_dict = yaml.load(yaml_config, Loader=yaml.Loader)

    # Find the model state dict from the config file.
    experiment_dir = Path(yaml_dict["EXPERIMENT_DIR"])
    experiment_folder = Path(yaml_dict["EXPERIMENT_FOLDER"])
    model_name = yaml_dict["MODEL_NAME"]
    path_state_dict = experiment_dir / experiment_folder / "{}.state".format(model_name)

    depth = yaml_dict["DEPTH"]
    connectivity = yaml_dict["MODEL_CONNECTIVITY"]
    encoding_dim = yaml_dict["ENCODING_DIMENSION"]
    dim_h = yaml_dict["MODEL_DIM_H"]
    encoder_kernel_size = yaml_dict["ENCODER_KERNEL_SIZE"]
    batch_size = yaml_dict["BATCH_SIZE"]
    sigmoid_scale = yaml_dict["SIGMOID_SCALE"]
    sigmoid_shift = yaml_dict["SIGMOID_SHIFT"]

    model = IPE_autoencoder_mu_l(
            (batch_size, *input_size),
            encoding_dim,
            connectivity=connectivity,
            depth=depth,
            dim_h=dim_h,
            final_mu_activation=None,
            final_var_activation=(lambda :
                    DiagChannelActivation(
                        activation_maker=(
                            lambda : SigmoidScaleShift(
                                scale=sigmoid_scale,
                                shift=sigmoid_shift)),
                        diag_channel_idx=0)),
                encoder_kernel_size=encoder_kernel_size
            )

    model = model.to(map_location)

    state_dict = torch.load(path_state_dict)
    model.load_state_dict(state_dict)

    return model

def main():
    PATH_HEALTHY_RAW = "./fl_dataset/healthy_test/raw"
    PATH_HEALTHY_GT = "./fl_dataset/healthy_test/mask"

    PATH_INFECTED_RAW = "./fl_dataset/infected/raw"
    PATH_INFECTED_GT = "./fl_dataset/infected/mask"

    PATH_SUPN = "./example_train_supn.yaml"
    PATH_DIAG = "./example_train_diag.yaml"
    # TILE_FACTOR is basically the size of a 2dpool-ing neighbourhood.
    TILE_FACTOR = 2
    
    DEVICE='cpu'
    device_name = 'cuda:{}'.format(DEVICE) if type(DEVICE) is int else 'cpu'

    parser = argparse.ArgumentParser(description="Script to evaluate SUPN, Diag, L2 performance on fluo data.")
    parser.add_argument("--path_supn", type=str, help="Path to SUPN training config", required=True)
    parser.add_argument("--path_diag", type=str, help="Path to DIAG training config")
    args = parser.parse_args()

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

    model_diag = load_model(args.path_diag, map_location=torch.device(device_name))
    # Wrap around the diag model to change the diagonal vector prediciton into
    # a valid Cholesky decomposition for the downstream evaluation (so it can
    # be shared with SUPN).
    model_diag_fwd = Diag_model_wrap(model_diag)
    model_supn = load_model(args.path_supn, map_location=torch.device(device_name))
    # Does not matter which model is taken as the L2 model, both have equally
    # good mean predictors, and the covariance is an identity.
    model_l2_fwd = L2_model_wrap(model_supn)

    scoring_func = lambda x_mu, x_chol, x, mask : scoring_func_outliers(
            x_mu, x_chol, x, mask, TILE_FACTOR)

    model_resto_supn = Restoration_model_wrap(model_supn)
    model_resto_diag = Restoration_model_wrap(model_diag_fwd)
    model_resto_l2 = Restoration_model_wrap(model_l2_fwd)

    metrics_resto_supn = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_supn, scoring_func, device=device_name)
    metrics_resto_diag = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_diag, scoring_func, device=device_name)
    metrics_resto_l2 = vae_model_scoring(dset_healthy, dset_infected, 
                model_resto_l2, scoring_func, device=device_name)

    metrics_supn = vae_model_scoring(dset_healthy, dset_infected, 
            model_supn, scoring_func, device=device_name)
    metrics_diag = vae_model_scoring(dset_healthy, dset_infected, 
            model_diag_fwd, scoring_func, device=device_name)
    metrics_l2 = vae_model_scoring(dset_healthy, dset_infected, 
            model_l2_fwd, scoring_func, device=device_name)

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
