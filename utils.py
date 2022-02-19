import argparse
from datetime import datetime
import yaml
import os
import time

import torch
import torchvision.transforms as tv_transforms
import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_triangular
from scipy.ndimage.morphology import distance_transform_edt

from structured_uncertainty.losses import kl_divergence_unit_normal

from torch.nn import functional as F

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

def mahalanoubis_dist(x,mu,sparseL):
    r"""
    r.T cov-1 r
    r.T (L.T L) r
    (Lr).T (Lr)
    """
    Lr = apply_sparse_chol_rhs_matmul((x-mu),
            log_diag_weights=sparseL[:,0].unsqueeze(1),
            off_diag_weights=sparseL[:,1:],
            local_connection_dist=1,
            use_transpose=True)

    return Lr.square()

def conditional_mahalanobis(image, mean, chols, selection, conn=1, use_transpose=True):
    r"""
    args:
        :image: [b,1,h,w]
        :mean: [b,1,h,w]
        :selection: [b,1,h,w] bool mask, the selected pixels which will be
            the new random variables, conditioned on the rest of the pixels.
        :chols: [b, nonzero ,h,w], nonzero is the number of nonzero elements
            predicted in the cholesky matrix.
        :conn: int, connectivity parameter.
        :use_transpose: bool, default true, used upstream?

    output:
        conditional mahalanobis distance.
    """
    b, nonzero, h, w = chols.shape

    assert b == 1, "batch > 1 not implemented"
    assert conn == 1, "calculation only implemented for connectivity 1"
    device = chols.device

    idxs = torch.arange(h*w)

    idxs_diag = torch.stack([idxs, idxs], dim=0)
    vals_diag = chols[:,0].reshape(b, h*w).exp() # [b,h,w] -> [b,h*w]

    idxs_right = torch.stack([idxs+1, idxs], dim=0)[:, :-1]
    vals_right = chols[:,1].reshape(b, h*w)[:, :-1]

    idxs_botleft = torch.stack([idxs + w - 1, idxs], dim=0)[:, :(h*w - w + 1)]
    vals_botleft = chols[:,2].reshape(b, h*w)[:, :(h*w - w + 1)]

    idxs_bot = torch.stack([idxs + w, idxs], dim=0)[:, :(h*w - w)] # [2, h*w]
    vals_bot = chols[:,3].reshape(b, h*w)[:, :(h*w - w)]

    idxs_botright = torch.stack([idxs + w + 1, idxs], dim=0)[:, :(h*w - w - 1)]
    vals_botright = chols[:,4].reshape(b, h*w)[:, :(h*w - w - 1)]

    idxs_out = torch.cat([idxs_diag, idxs_right, idxs_botleft, idxs_bot, idxs_botright], dim=1)
    vals_out = torch.cat([vals_diag, vals_right, vals_botleft, vals_bot, vals_botright], dim=1)

    ##### more efficient approach
    # create coordinates array
    hs_selection = torch.arange(h)
    ws_selection = torch.arange(w)
    idxs_selection = torch.zeros(2,h,w)
    idxs_selection[0] = hs_selection.unsqueeze(1).expand(-1,w)
    idxs_selection[1] = ws_selection.unsqueeze(0).expand(h,-1)

    # select the hw coordinates from the passed selection
    x_hw_selection = idxs_selection[:, selection[0,0]].long()
    y_hw_selection = idxs_selection[:, ~selection[0,0]].long()

    # build the roi cholesky matrix row subset
    tri_off_diag_filters = build_off_diag_filters(
            local_connection_dist=conn,
            use_transpose=use_transpose,
            device=device)

    offdiag_num_ = tri_off_diag_filters.shape[0] # number of offdiagonal elements
    ks_ = tri_off_diag_filters.shape[-1] # kernel size
    flipped_ = torch.flip(tri_off_diag_filters, [2,3])
    transpose_filters = torch.zeros(offdiag_num_, offdiag_num_, ks_, ks_)

    for pos_ in range(offdiag_num_):
        transpose_filters[pos_, pos_] = flipped_[pos_, 0]

    chol_offdiag_transpose = f.conv2d(chols[:,1:], transpose_filters, padding=conn, stride=1)

    # todo is this correct? -- seems to agree with the slow method when converted to dense.
    # flip along the channels, so that the diagonal becomes the last entry.
    chol_transpose = torch.flip(chols.clone(), [1])
    # replace the entries up to the last entry with the transposed entries.
    # also flip the transposed entries because the order along ch has to be reversed.
    chol_transpose[:,:-1] = torch.flip(chol_offdiag_transpose, [1])
    
    # Select x rows from L; the L representation has been made into 'row format',
    # so that the vector at each (h,w) now represent rows of the L matrix rather
    # than columns as the model originally predicts.
    Lx_vals = chol_transpose[:, :, x_hw_selection[0], x_hw_selection[1]] # [B,offd, |x|]
    Ly_vals = chol_transpose[:, :, y_hw_selection[0], y_hw_selection[1]] # [B,offd, |y|]

    transp_enum = torch.flip(torch.Tensor([0, -1, -W+1, -W, -W-1]), [0]) # displacement from diagonal along rows
    Lh_positions = (idxs_selection[0] * W + idxs_selection[1]).unsqueeze(0).expand(nonzero,-1,-1) # [nonzero, H,W]
    Lw_positions = Lh_positions + transp_enum[:,None,None].expand(-1,H,W) # [nonzero,H,W]

    select_x_Lh_positions = Lh_positions[:, x_hw_selection[0], x_hw_selection[1]] # [offd, |x|]
    select_x_Lw_positions = Lw_positions[:, x_hw_selection[0], x_hw_selection[1]] # [offd, |x|]
    select_y_Lh_positions = Lh_positions[:, y_hw_selection[0], y_hw_selection[1]] # [offd, |y|]
    select_y_Lw_positions = Lw_positions[:, y_hw_selection[0], y_hw_selection[1]] # [offd, |y|]

    re_enum_x = torch.arange(select_x_Lh_positions.shape[-1])[None,].expand(nonzero, -1)
    re_enum_y = torch.arange(select_y_Lh_positions.shape[-1])[None,].expand(nonzero, -1)

    tmp_x_unravel = torch.stack([
        re_enum_x[None,None,].expand(B,-1,-1,-1),
        select_x_Lw_positions[None,None,].expand(B,-1,-1,-1), 
        Lx_vals[None,]], dim=1)[0,:,0].reshape(3,-1)
    tmp_y_unravel = torch.stack([
        re_enum_y[None,None,].expand(B,-1,-1,-1),
        select_y_Lw_positions[None,None,].expand(B,-1,-1,-1), 
        Ly_vals[None,]], dim=1)[0,:,0].reshape(3,-1)

    tmp_x = tmp_x_unravel[:, torch.logical_and(tmp_x_unravel[0] >= 0, tmp_x_unravel[1] >= 0)]
    tmp_y = tmp_y_unravel[:, torch.logical_and(tmp_y_unravel[0] >= 0, tmp_y_unravel[1] >= 0)]
    Chol_x = torch.sparse_coo_tensor(tmp_x[0:2], tmp_x[2], (selection.sum(), H*W))
    Chol_y = torch.sparse_coo_tensor(tmp_y[0:2], tmp_y[2], ((~selection).sum(), H*W))

    Lambda_xx = torch.sparse.mm(Chol_x, Chol_x.transpose(0,1))
    Lambda_xx_inv = torch.linalg.inv(Lambda_xx.to_dense())
    Lambda_xy = torch.sparse.mm(Chol_x, Chol_y.transpose(0,1))

    x_rows = idxs[selection[0,0].reshape(-1)].long()
    y_rows = idxs[~selection[0,0].reshape(-1)].long()
    residual = (image - mean)[0,0].reshape(-1)
    rx, ry = residual[x_rows], residual[y_rows]

    Lxy = torch.sparse.mm(Lambda_xy, ry.unsqueeze(-1)) # LxLy^T ry
    Lxx = torch.sparse.mm(Chol_x.transpose(0,1), rx.unsqueeze(-1)) # (hw,)
    Lyy = torch.sparse.mm(Chol_y.transpose(0,1), ry.unsqueeze(-1)) # (hw,)

    mah_term1 = Lxx.square().sum()
    mah_term2 = -2 * (Lxx * Lyy).sum()
    mah_term3 = Lxy.T @ Lambda_xx_inv @ Lxy

    ##### Approach directly with sparse matrices
    #sparse_chol = torch.sparse_coo_tensor(idxs_out, vals_out[0], (H*W, H*W))
    #sparse_Lambda = torch.sparse.mm(sparse_chol, sparse_chol.transpose(0,1))


    #Lambda_xx_xy = sparse_Lambda.index_select(0, xx_rows)
    #Lambda_xx = Lambda_xx_xy.index_select(1, xx_rows)
    #Lambda_xy = Lambda_xx_xy.index_select(1, xy_rows)
    #
    #sch = sparse_chol.index_select(0, xx_rows)

    #
    #Lambda_xy_ry = torch.sparse.mm(Lambda_xy, ry.unsqueeze(-1))
    #Lambda_xx_inv = torch.linalg.inv(Lambda_xx.to_dense())

    ##### Calculate Mahalanobis distance #####

    ## Mah Term 1 is the equivalent of the unconditional Mah dist for these pixels.
    #mah_term1 = rx * torch.sparse.mm(Lambda_xx_fast, rx.unsqueeze(-1))[:,0]
    #mah_term2 = 2 * (rx * Lambda_xy_ry)[:,0]
    ## Mah Term 3 requires the inverse of the Lambda_xx matrix (dense).
    #mah_term3 = (Lambda_xy_ry * (Lambda_xx_inv @ Lambda_xy_ry))[:,0]

    #mah_term2 = -2 * Lxx

    #import matplotlib.pyplot as plt
    #fig,axes = plt.subplots(1,3)
    #axes[0].imshow(mah_term1.reshape(10,10))
    #axes[1].imshow(mah_term2.reshape(10,10))
    #axes[2].imshow(mah_term3.reshape(10,10))
    #plt.show()

    conditional_mahalanobis = (mah_term1 + mah_term2 + mah_term3).sum()

    #k_dim = selection.sum()
    #logprob = -0.5 * conditional_mahalanobis - 0.5 * torch.logdet(Lambda_xx_inv) - 0.5 * k_dim * torch.log(torch.Tensor([2 * np.pi]))
    #import pdb; pdb.set_trace()

    return conditional_mahalanobis

def conditional_nll(image, mean, chols, selection, conn=1, use_transpose=True):
    r"""
    Args:
        :image: [B,1,H,W]
        :mean: [B,1,H,W]
        :selection: [B,1,H,W] bool mask, the selected pixels which will be
            the new random variables, conditioned on the rest of the pixels.
        :chols: [B, nonzero ,H,W], nonzero is the number of nonzero elements
            predicted in the cholesky matrix.
        :conn: int, connectivity parameter.
        :use_transpose: bool, default True, used upstream?

    Output:
        Conditional Mahalanobis distance.
    """
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

    ##### More efficient approach
    # Create coordinates array
    hs_selection = torch.arange(H)
    ws_selection = torch.arange(W)
    idxs_selection = torch.zeros(2,H,W)
    idxs_selection[0] = hs_selection.unsqueeze(1).expand(-1,W)
    idxs_selection[1] = ws_selection.unsqueeze(0).expand(H,-1)

    # Select the hw coordinates from the passed selection
    x_hw_selection = idxs_selection[:, selection[0,0]].long()
    y_hw_selection = idxs_selection[:, ~selection[0,0]].long()

    # Build the ROI Cholesky matrix row subset
    tri_off_diag_filters = build_off_diag_filters(
            local_connection_dist=conn,
            use_transpose=use_transpose,
            device=device)

    offdiag_num_ = tri_off_diag_filters.shape[0] # number of offdiagonal elements
    ks_ = tri_off_diag_filters.shape[-1] # kernel size
    flipped_ = torch.flip(tri_off_diag_filters, [2,3])
    transpose_filters = torch.zeros(offdiag_num_, offdiag_num_, ks_, ks_)

    for pos_ in range(offdiag_num_):
        transpose_filters[pos_, pos_] = flipped_[pos_, 0]

    chol_offdiag_transpose = F.conv2d(chols[:,1:], transpose_filters, padding=conn, stride=1)

    # TODO Is this correct? -- Seems to agree with the slow method when converted to dense.
    # Flip along the channels, so that the diagonal becomes the last entry.
    chol_transpose = torch.flip(chols.clone(), [1])
    # Replace the entries up to the last entry with the transposed entries.
    # Also flip the transposed entries because the order along ch has to be reversed.
    chol_transpose[:,:-1] = torch.flip(chol_offdiag_transpose, [1])
    
    # Select x rows from L; the L representation has been made into 'row format',
    # so that the vector at each (h,w) now represent rows of the L matrix rather
    # than columns as the model originally predicts.
    Lx_vals = chol_transpose[:, :, x_hw_selection[0], x_hw_selection[1]] # [B,offd, |x|]
    Ly_vals = chol_transpose[:, :, y_hw_selection[0], y_hw_selection[1]] # [B,offd, |y|]

    transp_enum = torch.flip(torch.Tensor([0, -1, -W+1, -W, -W-1]), [0]) # displacement from diagonal along rows
    Lh_positions = (idxs_selection[0] * W + idxs_selection[1]).unsqueeze(0).expand(nonzero,-1,-1) # [nonzero, H,W]
    Lw_positions = Lh_positions + transp_enum[:,None,None].expand(-1,H,W) # [nonzero,H,W]

    select_x_Lh_positions = Lh_positions[:, x_hw_selection[0], x_hw_selection[1]] # [offd, |x|]
    select_x_Lw_positions = Lw_positions[:, x_hw_selection[0], x_hw_selection[1]] # [offd, |x|]
    select_y_Lh_positions = Lh_positions[:, y_hw_selection[0], y_hw_selection[1]] # [offd, |y|]
    select_y_Lw_positions = Lw_positions[:, y_hw_selection[0], y_hw_selection[1]] # [offd, |y|]

    re_enum_x = torch.arange(select_x_Lh_positions.shape[-1])[None,].expand(nonzero, -1)
    re_enum_y = torch.arange(select_y_Lh_positions.shape[-1])[None,].expand(nonzero, -1)

    tmp_x_unravel = torch.stack([
        re_enum_x[None,None,].expand(B,-1,-1,-1),
        select_x_Lw_positions[None,None,].expand(B,-1,-1,-1), 
        Lx_vals[None,]], dim=1)[0,:,0].reshape(3,-1)
    tmp_y_unravel = torch.stack([
        re_enum_y[None,None,].expand(B,-1,-1,-1),
        select_y_Lw_positions[None,None,].expand(B,-1,-1,-1), 
        Ly_vals[None,]], dim=1)[0,:,0].reshape(3,-1)

    tmp_x = tmp_x_unravel[:, torch.logical_and(tmp_x_unravel[0] >= 0, tmp_x_unravel[1] >= 0)]
    tmp_y = tmp_y_unravel[:, torch.logical_and(tmp_y_unravel[0] >= 0, tmp_y_unravel[1] >= 0)]
    Chol_x = torch.sparse_coo_tensor(tmp_x[0:2], tmp_x[2], (selection.sum(), H*W))
    Chol_y = torch.sparse_coo_tensor(tmp_y[0:2], tmp_y[2], ((~selection).sum(), H*W))

    Lambda_xx = torch.sparse.mm(Chol_x, Chol_x.transpose(0,1))
    Lambda_xx_inv = torch.linalg.inv(Lambda_xx.to_dense())
    Lambda_xy = torch.sparse.mm(Chol_x, Chol_y.transpose(0,1))

    x_rows = idxs[selection[0,0].reshape(-1)].long()
    y_rows = idxs[~selection[0,0].reshape(-1)].long()
    residual = (image - mean)[0,0].reshape(-1)
    rx, ry = residual[x_rows], residual[y_rows]

    Lxy = torch.sparse.mm(Lambda_xy, ry.unsqueeze(-1)) # LxLy^T ry
    Lxx = torch.sparse.mm(Chol_x.transpose(0,1), rx.unsqueeze(-1)) # (hw,)
    Lyy = torch.sparse.mm(Chol_y.transpose(0,1), ry.unsqueeze(-1)) # (hw,)

    mah_term1 = Lxx.square().sum()
    mah_term2 = -2 * (Lxx * Lyy).sum()
    mah_term3 = Lxy.T @ Lambda_xx_inv @ Lxy

    conditional_mahalanobis = (mah_term1 + mah_term2 + mah_term3).sum()

    k_dim = selection.sum()
    # TODO Glitchy?
    conditional_nll = 0.5 * conditional_mahalanobis - 0.5 * torch.logdet(Lambda_xx_inv) + 0.5 * k_dim * torch.log(torch.Tensor([2 * np.pi]))

    return conditional_nll

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

def nonmax_suppression_tiles(scores, threshold=0.4, tile_side=4, bb_size=1):
    r"""
    Performs the nonmax suppression algorithm and outputs the filtered out 
    bounding boxes, where the boxes are expanded tiles (iou can be calculated
    from the coordinates).

    Args:
        :scores: [N,4], each box (N) holds (n, coord_h, coord_w, score),
            where n is one of the image stack.
        :threshold: float, parameter of the nonmax suppression, percent overlap,
            if the overlap is greater than this, then the overlapping cells with
            lower score will be filtered out.
        :tile_side: int, how big the tiles are (squares).
        :bb_size: int, how many times bigger the bounding box sizes are than
            the small tiles.

    Returns:
        np.ndarray [Q, ]
    """
    images_indices = np.unique(scores[:,0])
    bb_side = tile_side * bb_size

    output_boxes = []
    
    # Calculate the nonmax suppression per frame:
    for image_idx in images_indices:

        # TODO Check why filtering not correct.
        # Select the boxes of this frame from the input.
        boxes_idx_mask = scores[:,0] == image_idx # [N,]
        scores_idx_ = scores[boxes_idx_mask] # [M,4], M <= N
        is_not_processed = np.ones(boxes_idx_mask.sum()).astype(bool) # [M,], init all to True

        # Iterate until all boxes were either selected or suppressed.
        while is_not_processed.sum() != 0:

            # Get the max score box and the coordinate offset to the rest of the boxes.
            max_score_idx = scores_idx_[:,-1].argmax() # int for indexing M
            output_boxes.append(scores_idx_[max_score_idx]) # Appends [n,h,w,score] array.
            delta_coords = scores_idx_[:, 1:-1] - scores_idx_[max_score_idx, 1:-1] # [M, 2]

            # Mark the boxes in the neighbourhood which are to be suppressed.
            # Intersection area is calculated by multiplying the intersection in
            # each coordinate axis.
            intersection = np.maximum(bb_side - np.abs(delta_coords), 0) # [M,2]
            intersection = intersection[:,0] * intersection[:,1] # [M,]
            union = 2 * bb_side**2 - intersection
            iou_ = intersection / union # [M, 1] IOU between 'image_idx' and the rest.
            suppression_mask = iou_ > threshold # [M,1] bool mask of boxes which will be suppressed.

            # Only cells which were not suppressed in this iteration and have not
            # been 'processed' previously, will be candidates in future iterations.
            is_not_processed = np.logical_and(is_not_processed, ~suppression_mask)

            # Set the scores of the 'processed' boxes (all selected or suppressed
            # boxes so far) to 0, so that argmax ignores them (scores >= 0).
            scores_idx_[~is_not_processed, -1] = 0.0

    output_boxes = np.stack(output_boxes, axis=0)

    return output_boxes

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

def get_combi_metric(x, model, conn=1, kl_blur=3, kl_kernel=5):
    r"""
    The Combi metric used to get good results in the following work:

    Unsupervised Anomaly Localization using Variational Auto-Encoders

    Combi = | dKL/dx | * Rec

    Where KL is the KL-divergence in the latent space and Rex is the
    reconstruction loss.
    """
    gaussian_blur = tv_transforms.GaussianBlur(kl_kernel, sigma=kl_blur)

    model.zero_grad()
    x = x.detach()
    x.requires_grad = True

    x_mu, x_chol, z_mu, z_logvar = model(x)

    kl = kl_divergence_unit_normal(z_mu, z_logvar)

    nll = - get_log_prob_from_sparse_L_precision(
                x.detach(), x_mu.detach(), conn,
                x_chol[:,0,...].unsqueeze(1).detach(),
                x_chol[:,1:,...].detach(),
                pixelwise=True) # (B,)
 
    kl.backward() # Compute gradients.
    # xgrad is the derivative of the KL divergence wrt each pixel in the input,
    # and it acts as a saliency map - emphasizing on the more 'important' parts
    # of the image.
    xgrad = gaussian_blur(x.grad.detach().abs())

    model.zero_grad()

    return xgrad * nll

def get_kl_saliency_map(x, model, conn=1, kl_blur=3, kl_kernel=5):
    r"""
    """
    gaussian_blur = tv_transforms.GaussianBlur(kl_kernel, sigma=kl_blur)

    model.zero_grad()
    x = x.detach()
    x.requires_grad = True

    _, _, z_mu, z_logvar = model(x)

    kl = kl_divergence_unit_normal(z_mu, z_logvar)

    kl.backward() # Compute gradients.
    xgrad = gaussian_blur(x.grad.detach())

    model.zero_grad()

    # xgrad is a saliency map (not normalized!)
    return xgrad


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
