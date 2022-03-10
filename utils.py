import argparse
from datetime import datetime
import yaml
import os
import time
from pathlib import Path

import torch
import torchvision.transforms as tv_transforms
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_triangular
from scipy.ndimage.morphology import distance_transform_edt

from torch.nn import functional as F

from transforms import SigmoidScaleShift
from model_blocks import (DiagChannelActivation,
        IPE_autoencoder_mu_l)

def get_timestamp_string():
    dtime = datetime.now()
    outp_ = "{}_{:02d}_{}_{}{}{}" \
            .format(dtime.year,
                    dtime.month,
                    dtime.day,
                    dtime.hour,
                    dtime.minute,
                    dtime.second)

    return outp_

def load_config_file(config_filepath):
    """
    Helper function that reads a yaml file and returns its contents as a dict.
    Args:
        :param config_filepath: str, a path pointing to the yaml config.
    """
    with open(config_filepath, "r") as yaml_config:
        yaml_dict = yaml.load(yaml_config, Loader=yaml.Loader)
        return yaml_dict

def parse_config_dict(description, config_arg_help):
    """
    Helper function which requires the user to submit a yaml config file before 
    running the rest of the code following it. It will then load the contents 
    of the config file and return them as a dict. 
    
    Passing a single yaml config file will be needed in a couple of places 
    throughout the algorithm (training and inference).
    Args:
        :param description: str, the program description that will be shown to
            the user.
    Returns:
        :argparse.ArgumentParser: Will prompt the user for a --config argument, 
            followed by a path to a .yaml configuration file.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, help=config_arg_help, required=True)
    args = parser.parse_args()

    return load_config_file(args.config)

def make_dirs_if_absent(DIR_LIST):
    for DIR in DIR_LIST:
        if not os.path.exists(DIR):
            print("creating dir {}".format(DIR))
            os.makedirs(DIR)

def get_coordinate_mask(mask):
    r"""
    Get coordinates of True elements.

    Args:
        :mask: [B,1,H,W]
    """
    B,_,H,W = mask.shape

    hs = torch.arange(H)
    ws = torch.arange(W)
    bs = torch.arange(B)

    coords = torch.zeros(3,B,H,W)
    coords[0] = hs[None,:,None].expand(B,-1,W).long()
    coords[1] = ws[None,None,:].expand(B,H,-1).long()
    coords[2] = bs[:,None,None].expand(-1,H,W).long()

    return coords[:,mask[:,0].bool()]

def retile_mask(mask, tile_factor):
    r"""
    Gets a smooth mask and returns a mask covered by the tiles.
    """
    tiled = mask
    for _ in range(tile_factor):
        tiled = torch.nn.functional.max_pool2d(tiled, 2)

    coords = get_coordinate_mask(tiled) 
    coords[:-1] = coords[:-1] * 2**tile_factor

    for _ in range(tile_factor):
        tiled = torch.nn.functional.interpolate(tiled, scale_factor=2, mode='nearest')

    return tiled, coords

def lhs_inverse_LT_matmul(image, mean, chols, conn=1, use_transpose=True, device='cpu', frame=1):

    if frame != 0:
        # Ignore some of the pixels near the edges
        image = image[..., frame:-frame, frame:-frame]
        mean = mean[..., frame:-frame, frame:-frame]
        chols = chols[..., frame:-frame, frame:-frame]

    B, nonzero, H, W = chols.shape

    assert B == 1, "Batch > 1 not implemented"
    assert conn == 1, "Calculation only implemented for connectivity 1"
    device = chols.device

    idxs = torch.arange(H*W)

    idxs_diag = torch.stack([idxs, idxs], dim=0)
    vals_diag = chols[:,0].reshape(B, H*W).exp() # [B,H,W] -> [B,H*W]

    idxs_right = torch.stack([idxs+1, idxs], dim=0)[:, :-1]
    vals_right = chols[:,1].reshape(B, H*W)[:, :-1]

    idxs_botleft = torch.stack([idxs + W - 1, idxs], dim=0)[:, :(H*W - W + 1)]
    vals_botleft = chols[:,2].reshape(B, H*W)[:, :(H*W - W + 1)]

    idxs_bot = torch.stack([idxs + W, idxs], dim=0)[:, :(H*W - W)] # [2, H*W]
    vals_bot = chols[:,3].reshape(B, H*W)[:, :(H*W - W)]

    idxs_botright = torch.stack([idxs + W + 1, idxs], dim=0)[:, :(H*W - W - 1)]
    vals_botright = chols[:,4].reshape(B, H*W)[:, :(H*W - W - 1)]

    idxs_out = torch.cat([idxs_diag, idxs_right, idxs_botleft, idxs_bot, idxs_botright], dim=1)
    vals_out = torch.cat([vals_diag, vals_right, vals_botleft, vals_bot, vals_botright], dim=1)

    Chol = torch.sparse_coo_tensor(idxs_out.to(device), vals_out[0].to(device), (H*W, H*W))
    Chol_T_inv = torch.linalg.inv(Chol.transpose(0,1).to_dense())
     
    LT_inv_x = Chol_T_inv @ mean[0,0].reshape(-1)

    return LT_inv_x.reshape(H,W)

def build_off_diag_filters(local_connection_dist, use_transpose=True, device=None, dtype=torch.float):
    """Create the conv2d filter weights for the off-diagonal components of the sparse chol.

    NOTE: Important to specify device if things might run under cuda since constants are created and need to be
        on the correct device.

    Parameters:
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True - usually what we want for the jacobi sampling.
        device: Specify the device to create the constants on (i.e. cpu vs gpu).

    Returns:
        tri_off_diag_filters(tensor): [num_off_diag_weights x 1 x filter_size x filter_size] Conv2d kernel filters.
    """
    filter_size = 2 * local_connection_dist + 1
    filter_size_sq = filter_size * filter_size
    filter_size_sq_2 = filter_size_sq // 2

    if use_transpose:
        tri_off_diag_filters = torch.cat((torch.zeros(filter_size_sq_2, (filter_size_sq_2 + 1),
                                                      device=device, dtype=dtype),
                                          torch.eye(filter_size_sq_2,
                                                    device=device, dtype=dtype)), dim=1)
    else:
        tri_off_diag_filters = torch.cat((torch.fliplr(torch.eye(filter_size_sq_2,
                                                                 device=device, dtype=dtype)),
                                          torch.zeros(filter_size_sq_2, (filter_size_sq_2 + 1),
                                                      device=device, dtype=dtype)), dim=1)

    tri_off_diag_filters = torch.reshape(tri_off_diag_filters, (filter_size_sq_2, 1, filter_size, filter_size))

    return tri_off_diag_filters


def apply_sparse_chol_rhs_matmul(dense_input, log_diag_weights, off_diag_weights,
                                 local_connection_dist, use_transpose=True):
    """Apply the sparse chol matrix to a dense input on the rhs i.e. result^T = input^T L  (standard matrix mulitply).

    IMPORTANT: Only valid for a single channel at the moment.

    Parameters:
        dense_input(tensor): [BATCH x 1 x W x H] Input matrix (must be single channel).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        use_transpose(bool): Defaults to True.

    Returns:
        product(tensor): [BATCH x 1 x W x H] Result of (L dense_input) or (L^T dense_input).
    """
    assert dense_input.ndim == 4
    assert log_diag_weights.ndim == 4
    assert off_diag_weights.ndim == 4

    device = dense_input.device

    assert dense_input.shape[1] == 1

    tri_off_diag_filters = build_off_diag_filters(local_connection_dist=local_connection_dist,
                                                  use_transpose=use_transpose,
                                                  device=device)

    MIN_DIAG_VALUE = 1.0e-10
    diag_values = torch.exp(log_diag_weights) + MIN_DIAG_VALUE

    interim = F.conv2d(dense_input, tri_off_diag_filters, padding=local_connection_dist, stride=1)

    after_weights = torch.einsum('bfwh, bfwh->bwh' if off_diag_weights.shape[0] > 1 else 'bfwh, xfwh->bwh',
                                 interim, off_diag_weights)

    result = diag_values * dense_input + after_weights.view(*dense_input.shape)

    return result

def get_log_prob_from_sparse_L_precision(x,
                                         mean,
                                         local_connection_dist,
                                         log_diag_weights,
                                         off_diag_weights,
                                         use_transpose=True,
                                         mask=None,
                                         pixelwise=False):
    """Efficient calculation of the log probability of x under the sparse chol precision matrix.

    IMPORTANT: Only valid for a single channel at the moment.

    Parameters:
        x(tensor): [BATCH x 1 x W x H] Data to evaluate the likelihood of (must be single channel).
        mean(tensor): [BATCH x 1 x W x H] Mean (must be single channel).
        local_connection_dist(int): Positive integer specifying local pixel distance (e.g. 1 => 3x3, 2 => 5x5, etc..).
        log_diag_weights(tensor): [B ? 1 x 1 x W x H] log of the diagonal terms (mapped through exp).
        off_diag_weights(tensor): [B ? 1 x F x W x H] off-diagonal terms. F = get_num_off_diag_weights(local_connection_dist)
        use_transpose(bool): Defaults to True.
        mask(tensor,bool or None): [B x 1 x W x H] If not none, use to select which pixels are used to compute the prob, and which are ignored.

    Returns:
        log_prob(tensor): [B] The log probability of x.
    """
    assert log_diag_weights.ndim == 4
    assert off_diag_weights.ndim == 4

    device = x.device

    # assert log_diag_weights.shape[0] == 1
    assert log_diag_weights.shape[1] == 1
    im_size_w = log_diag_weights.shape[-2]
    im_size_h = log_diag_weights.shape[-1]

    # Might need to do something more clever here:
    x_minus_mu = x - mean

    fitting_term = apply_sparse_chol_rhs_matmul(x_minus_mu,
                                                log_diag_weights=log_diag_weights,
                                                off_diag_weights=off_diag_weights,
                                                local_connection_dist=local_connection_dist,
                                                use_transpose=use_transpose)

    constant_term = im_size_w * im_size_h * torch.log(torch.Tensor([2.0 * np.pi]))
    constant_term = constant_term.to(device)

    if mask is not None:
        log_diag_weights[(-(mask-1)).bool()] *= 0.
        fitting_term[(-(mask-1)).bool()] *= 0.

    if not pixelwise:
        log_det_term = 2.0 * torch.sum(log_diag_weights, dim=(1,2,3,)) # Note these are precision NOT covariance L
    else:
        log_det_term = 2.0 * log_diag_weights # Note these are precision NOT covariance L

    if not pixelwise:
        log_prob = -0.5 * constant_term -0.5 * torch.sum(torch.square(fitting_term), dim=(1,2,3,)) \
               +0.5 * log_det_term # Note positive since precision..
    else:
        log_prob = -0.5 * constant_term -0.5 * torch.square(fitting_term) \
               +0.5 * log_det_term # Note positive since precision..

    return log_prob

def load_model(config_path, input_size=(1,128,128), map_location='cuda:0', 
        dict_passed=None, pretrained_mean=True):
    r"""
    Initialize model architecture from config file, then load the trained state
    dict.

    Args:
        :config_path: Path or str, to the config file specifying the model params.
        :input_size: tuple, (c,h,w) what size the input images are, channels
            including (even though at this point model only works with
            grayscale).
        :map_location: str or torch.device, where to put the model.
        :dict_passed: dict, if not None, will use this as the already loaded 
            yaml dict rather than look for the file at :config_path:.
    """
    if dict_passed is None:
        yaml_dict = {}
        with open(config_path, "r") as yaml_config:
            yaml_dict = yaml.load(yaml_config, Loader=yaml.Loader)
    else:
        yaml_dict = dict_passed

    # Find the model state dict from the config file.
    experiment_dir = Path(yaml_dict["EXPERIMENT_DIR"])
    experiment_folder = Path(yaml_dict["EXPERIMENT_FOLDER"])
    model_name = yaml_dict["MODEL_NAME"]

    if pretrained_mean:
        path_state_dict = yaml_dict["PRETRAINED_MODEL_PATH"]
    else:
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
    print("loaded model {}".format(path_state_dict))

    return model

def get_scores(images, model, scoring_func):
    r"""
    Args:
        :images: [N,1,H,W] torch.Tensor
        :model: torch.Module
        :scoring_func: callable

    """
    scores = []
    for dpoint, mask in tqdm(dset):
        dpoint = dpoint[None,].to(device) # [B,1,H,W]
        mask = mask[None]

        score_samples = []
        for sid in tqdm(range(samples)):
            x_mu, x_chol, _, _ = model(dpoint)
            # In range [0,1], binary classification task between in or out of
            # distribution.
            score_sample = scoring_func(x_mu, x_chol, dpoint, mask)
            score_samples.append(score_sample)

        score = np.array(score_samples).mean()
        scores.append(score)

    return scores

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
            mah = mahalanobis_dist(dpoint, x_mu, x_chol, conn=conn)
            mah.sum().backward()
            optimizer.step()
            with torch.no_grad():
                deltas.append((z_.grad * step).square().sum())
            optimizer.zero_grad()

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

class ModelWrapper:
    r"""
    Abstract class for a model wrapper.
    """
    def __init__(self, model):
        self.model = model
        self.connectivity = model.connectivity

    def zero_grad(self):
        self.model.zero_grad()

    def encoder(self, x):
        return self.model.encoder(x)

    def mu_decoder(self, z):
        raise NotImplementedError

    def var_decoder(self, z):
        raise NotImplementedError
    
    def select_z(self, z_mu, z_logvar):
        raise NotImplementedError

    def __call__(self, x):
        r"""
        Common calculations for all models:

        z_mu, z_logvar <--- Encoder(x)
        z ~ N( z_mu, z_logvar )
        x_mu, x_chol <--- Decoder(z)

        But they could vary, e.g. for the restorative approach we do not sample
        z, but rather select it using gradient descent for the ELBO.
        """

        z_mu, z_logvar = self.encoder( x )
        z = self.select_z( z_mu, z_logvar )
        x_mu = self.mu_decoder( z )
        x_chol = self.var_decoder( z )

        return x_mu, x_chol, z_mu, z_logvar

class Diag_model_wrap(ModelWrapper):

    def zero_grad(self):
        self.model.zero_grad()

    def mu_decoder(self, z):
        return self.model.mu_decoder(z)

    def select_z(self, z_mu, z_logvar):
        r"""
        Sample from MVN defined by the z predictions, much like the model
        during training.
        """
        return self.model.reparametrize(z_mu, z_logvar.exp())

    def var_decoder(self, z):
        r"""
        Output of the diagonal model has only one channel and is interpreted as
        the log-diagonal of the covariance matrix. In order to be interpreted
        as the log-diag of the Cholesky of the precision matrix, we need to
        multiply by -0.5. The minus is for inverting the value (to precision) and 
        the 0.5 for taking the square-root (Cholesky of a diagonal precision).
        """
        # For diagonal model, output is a vector representing the log-diagonal 
        # of the covariance matrix.
        x_logvar = self.model.var_decoder(z)
        device = z.device
        B,_,H,W = x_logvar.shape
        nonzeros = conn_to_nonzeros(self.connectivity)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)
        x_chol[:,0] = -0.5*x_logvar[:,0]

        return x_chol

class L2_model_wrap(ModelWrapper):

    def mu_decoder(self, z):
        return self.model.mu_decoder(z)

    def select_z(self, z_mu, z_logvar):
        r"""
        Sample from MVN defined by the z predictions, much like the model
        during training.
        """
        return self.model.reparametrize(z_mu, z_logvar.exp())

    def var_decoder(self, z, H=128, W=128):
        r"""
        Output of the L2 model is really only the mean, but it is interpreted as
        a spherical standard MVN (identity matrix as covariance). To reinterpret it
        as a cholesky output, we do the same as in 'diag_model_wrap', but it turns
        out that the cholesky of the precision is also an identity matrix. Since 
        the evaluation functions expect a Cholesky output with log-diagonal values,
        we pass a matrix of zeros, where the off-diagonals are interpreted as
        actual zeros, but the diagonal will be exponentiated later to have ones.
        """
        conn = self.connectivity
        B,_ = z.shape
        device = z.device
        nonzeros = conn_to_nonzeros(conn)
        x_chol = torch.zeros(B, nonzeros, H, W).to(device)

        # Here x_logvar is completely discarded. It doesn't matter if the model was
        # trained to predict something meaningful there or not.
        return x_chol

def run_through_vae_restoration(input_images, model_l2, model_diag, model_chol,
        fixed_var, DEVICE=0, return_means=False, return_chols=False, 
        return_diag_logvar=False):
    r"""
    Calculate the Mahalanobis distances using a restoration-based method.

    Args:
        :input_images: torch.Tensor [TP, 1, H, W]
        :masks: [TP,1,H,W]
    """
    assert len(input_images.shape) == 4,\
            "Got {} need [TP, 1, H, W]".format(input_images.shape)

    conn = model_chol.connectivity
    TEST_POINTS, _, H, W = input_images.shape
    # TP (Test Points along the x axis) is the batch dimension.
    inp_ = input_images.to(DEVICE)

    # In: [TP,1,H,W], Out: [TP,*,H,W]
    x_mu_l2_, x_var_l2_, _, _ = model_l2(inp_) # Wrapped model.
    x_mu_diag_, x_var_diag_, _, _ = model_diag(inp_)
    x_mu_chol_, x_var_chol_, _, _ = model_chol(inp_)

    chols = x_var_chol_
    means = (x_mu_diag_ + x_mu_chol_)/2 # diag and chol should have the same decoder
    diag_logvars = x_var_diag_
    L2 = mahalanobis_dist(inp_, x_mu_l2_, x_var_l2_)
    Diag = mahalanobis_dist(inp_, x_mu_diag_, x_var_diag_)
    Supn = mahalanobis_dist(inp_, x_mu_chol_, x_var_chol_)

    if return_diag_logvar:
        return L2.detach(), Diag.detach(), Supn.detach(), means.detach(), chols.detach(), diag_logvars.detach()
    if return_chols:
        return L2.detach(), Diag.detach(), Supn.detach(), means.detach(), chols.detach()
    if return_means:
        return L2.detach(), Diag.detach(), Supn.detach(), means.detach()
    else:
        return L2.detach(), Diag.detach(), Supn.detach()

def mahalanobis_dist(x,mu,sparseL, conn=1):
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

def get_ellipsoid_mask(out_shape, center, a=10., b=5., angle=0.3):
    cos_ = torch.cos(torch.Tensor([angle])).item()
    sin_ = torch.sin(torch.Tensor([angle])).item()
    rot_ = torch.Tensor([[cos_, -sin_],
            [sin_, cos_]]) # (2,2)

    mask = torch.tensor(np.indices(out_shape))
    dist_ = (mask - center[:,None,None])
    dist_ = torch.einsum("ij,jkl->ikl", rot_,dist_)
    dist_[0] /= a
    dist_[1] /= b
    dist_ = dist_.square().sum(0).sqrt()

    mask = (dist_ < 1).bool()

    return mask

def get_ellipsoid_pattern(out_shape, center, a=10., b=5., angle=0.3):
    r"""
    """
    mask = get_ellipsoid_mask(out_shape, center, a=a, b=b, angle=angle)
    dt = distance_transform_edt(mask.numpy())
    pattern = rescale_to(dt, to=(0,1))

    return pattern

def get_ellipsoid_noise(out_shape, center, a=10., b=5., angle=0.3, 
        stdev=0.5):
    r"""
    """
    mask = get_ellipsoid_mask(out_shape, center, a=a, b=b, angle=angle)
    random_sample = torch.randn(mask.sum()) * stdev
    pattern = torch.zeros(*out_shape)
    pattern[mask] = random_sample

    return pattern
