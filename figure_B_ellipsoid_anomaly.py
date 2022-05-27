import torch
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import tqdm
import torchvision.transforms as tv_transforms
import dataset
from transforms import rescale_to

from utils import (
        load_model,
        mahalanobis_dist,
        run_through_vae_restoration,
        run_through_vae_qzsampled,
        L2_model_wrap,
        Diag_model_wrap,
        Restoration_model_wrap,
        conn_to_nonzeros,
        apply_sparse_chol_rhs_matmul,
        get_ellipsoid_noise,
        get_ellipsoid_pattern)

def plot_dpoint_lines(Lr_square, Diag_square, L2_square, x_ticks,
        supn_label='SUPN', diag_label='Diag', l2_label='L2', 
        alpha=None, linewidth=1.5, passed_axes=None, colors=None,
        labels=None, xlabel=None, ylabel=None):
    r"""
    Plot a line for each datapoint test sequence.

    Args:
        :Lr_square: torch.Tensor [Test points, datapoints]
        :Diag_square: torch.Tensor [TP, DP]
        :L2_square: torch.Tensor [TP, DP]
        :x_ticks: torch.Tensor [TP,] x axis values.
    """
    TP, DP = Lr_square.shape
    if alpha is None:
        alpha = 1 / DP

    VALUES = {
        0 : L2_square,
        1 : Diag_square,
        2 : Lr_square
    }

    if labels is not None:
        LABELS = labels
    else:
        LABELS = {
            0 : l2_label,
            1 : diag_label,
            2 : supn_label
        }

    if colors is not None:
        COLORS = colors
    else:
        COLORS = {
            0 : 'blue',
            1 : 'orange',
            2 : 'teal'
        }

    if passed_axes is not None:
        axes = passed_axes
    else:
        fig, axes = plt.subplots(1,3)

    for lidx, vals in VALUES.items():

        for dpidx in range(DP):
            
            axes[lidx].plot(x_ticks, vals[:,dpidx], 
                    color=COLORS[lidx], alpha=alpha,
                    linewidth=linewidth)

        label = LABELS[lidx]
        axes[lidx].plot(x_ticks, vals.mean(1), 
                color=COLORS[lidx], alpha=1.,label=label,
                linewidth=linewidth, linestyle="--")
        axes[lidx].legend()

    return axes

def main():
    RAW_PATH = "/u/homes/biv20/datasets/Kate_BF_healthy_sparse_crops/test_raw"
    GT_PATH = "/u/homes/biv20/datasets/Kate_BF_healthy_sparse_crops/test_mask"
    REAL_WORLD_PATH = "/u/homes/biv20/datasets/special_image.png"

    # TODO Change to state dicts and new loading function.
    CHOL_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/oldsupn_bf.yaml"
    #CHOL_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/example_train_bf_supn.yaml"
    DIAG_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/example_train_bf_diag.yaml"

    DEVICE = 0
    DSET_INPUT_IDX = 0
    N_DPOINTS = 100#5
    TEST_POINTS = 10#4
    ALPHA_TICK = 1 / (TEST_POINTS-1)
    PATTERN_LOC = (60,60)
    FIXED_VAR = 0.06

    sampled = True

    tforms = tv_transforms.Compose([
        lambda x : rescale_to(x, to=(-1,1)),
        tv_transforms.GaussianBlur(3, sigma=2.0)
        ])

    dset = dataset.FullFrames(RAW_PATH, GT_PATH, raw_transforms=tforms, debug=N_DPOINTS)

    # Model load and wrap
    #model_chol = Restoration_model_wrap(
    #        load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False))
    #model_diag = Restoration_model_wrap(
    #        Diag_model_wrap(
    #            load_model(DIAG_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False)
    #            ))
    #model_l2 = Restoration_model_wrap( L2_model_wrap(FIXED_VAR, 
    #    load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False) ))

    model_chol = load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False)
    model_diag = Diag_model_wrap(
                load_model(DIAG_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False)
                )
    model_l2 = L2_model_wrap(FIXED_VAR, 
        load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False) )

    tmp_, _ = dset[DSET_INPUT_IDX] # (1,H,W)
    _,H,W = tmp_.shape
    input_imgs = torch.zeros(N_DPOINTS,1,H,W)
    masks_ = torch.zeros(N_DPOINTS,1,H,W)
    for idx in range(N_DPOINTS):
        raw_, mask_ = dset[idx]
        input_imgs[idx] = raw_
        masks_[idx] = mask_

    # Load real BF parasite image
    input_real_rbc = torch.from_numpy(rescale_to(imageio.imread(REAL_WORLD_PATH), to=(-1,1))).unsqueeze(0).unsqueeze(0).unsqueeze(0).float()
    temp_tform = tv_transforms.GaussianBlur(3, sigma=2.0)
    input_real_rbc[0] = temp_tform(input_real_rbc[0])

    # Add pattern
    input_corrupted_noise = torch.zeros(TEST_POINTS, N_DPOINTS, 1, H,W)
    input_corrupted_pattern = torch.zeros(TEST_POINTS, N_DPOINTS, 1, H,W)
    noise = get_ellipsoid_noise((H,W), torch.Tensor(PATTERN_LOC), a=10.,b=4., angle=-0.7)
    pattern = get_ellipsoid_pattern((H,W), torch.Tensor(PATTERN_LOC), a=10.,b=4., angle=-0.7)
    # Load and expand pattern to shape (DP,H,W)
    noise = noise.unsqueeze(0).unsqueeze(0).expand(N_DPOINTS, -1,-1,-1)
    pattern = torch.from_numpy(pattern).unsqueeze(0).unsqueeze(0).expand(N_DPOINTS, -1,-1,-1) 
    for tpidx in tqdm(range(TEST_POINTS), desc="corrupting images.."):
        a_ = ALPHA_TICK * tpidx
        input_corrupted_noise[tpidx] = input_imgs + noise * a_
        input_corrupted_pattern[tpidx] = input_imgs + pattern * a_

    # Raw illustrations
    fig, axes = plt.subplots(4,1)
    axes[0].imshow(input_imgs[0,0], vmin=-1, vmax=1, cmap='gray')
    axes[0].set_title("Raw")
    axes[1].imshow(input_corrupted_noise[-1,0,0], vmin=-1, vmax=1, cmap='gray')
    #axes[1].set_title("Noisy anomaly")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[2].imshow(input_corrupted_pattern[-1,0,0], vmin=-1, vmax=1, cmap='gray')
    #axes[2].set_title("Smooth anomaly")
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[3].imshow(input_real_rbc[0,0,0], vmin=-1, vmax=1, cmap='gray')
    #axes[2].set_title("Smooth anomaly")
    axes[3].set_xticks([]); axes[3].set_yticks([])
    plt.show()

    # Saving to disk
    #torch.save(input_corrupted_noise.unsqueeze(1), "fig_B_input_noise.tensor")
    #torch.save(input_corrupted_pattern.unsqueeze(1), "fig_B_input_pattern.tensor")
    #torch.save(input_real_rbc.unsqueeze(1), "fig_B_input_real.tensor")

    # TODO Is this right?
    input_real_rbc = input_real_rbc[0]

    L2_noises = []
    Diag_noises = []
    Lr_noises = []
    Means_noises = []

    L2_patterns = []
    Diag_patterns = []
    Lr_patterns = []
    Means_patterns = []

    # For the real rbc
    if sampled:

        L2_real, Diag_real, Lr_real, Means_real = run_through_vae_qzsampled(
                input_real_rbc.to(DEVICE), model_l2, model_diag, model_chol, 
                DEVICE=DEVICE, return_means=True)

        for dp_idx in tqdm(range(N_DPOINTS)):

            L2_noise, Diag_noise, Lr_noise, Means_noise = run_through_vae_qzsampled(
                    input_corrupted_noise[:,dp_idx], model_l2, model_diag, model_chol,
                    DEVICE=DEVICE, return_means=True)
            L2_pattern, Diag_pattern, Lr_pattern, Means_pattern = run_through_vae_qzsampled(
                    input_corrupted_pattern[:,dp_idx], model_l2, model_diag, model_chol,
                    DEVICE=DEVICE, return_means=True)

            L2_noises.append(L2_noise)
            Diag_noises.append(Diag_noise)
            Lr_noises.append(Lr_noise)
            Means_noises.append(Means_noise)

            L2_patterns.append(L2_pattern)
            Diag_patterns.append(Diag_pattern)
            Lr_patterns.append(Lr_pattern)
            Means_patterns.append(Means_pattern)

    else:
        L2_real, Diag_real, Lr_real, Means_real = run_through_vae_restoration(
                input_real_rbc.to(DEVICE), model_l2, model_diag, model_chol, 
                DEVICE=DEVICE, return_means=True)

        for dp_idx in tqdm(range(N_DPOINTS)):

            L2_noise, Diag_noise, Lr_noise, Means_noise = run_through_vae_restoration(
                    input_corrupted_noise[:,dp_idx], model_l2, model_diag, model_chol, 
                    DEVICE=DEVICE, return_means=True)
            L2_pattern, Diag_pattern, Lr_pattern, Means_pattern = run_through_vae_restoration(
                    input_corrupted_pattern[:,dp_idx], model_l2, model_diag, model_chol, 
                    DEVICE=DEVICE, return_means=True)

            L2_noises.append(L2_noise)
            Diag_noises.append(Diag_noise)
            Lr_noises.append(Lr_noise)
            Means_noises.append(Means_noise)

            L2_patterns.append(L2_pattern)
            Diag_patterns.append(Diag_pattern)
            Lr_patterns.append(Lr_pattern)
            Means_patterns.append(Means_pattern)

    
    L2_noise = torch.stack(L2_noises, dim=1)
    Diag_noise = torch.stack(Diag_noises, dim=1)
    Lr_noise = torch.stack(Lr_noises, dim=1)
    Means_noise = torch.stack(Means_noises, dim=1)

    L2_pattern = torch.stack(L2_patterns, dim=1)
    Diag_pattern = torch.stack(Diag_patterns, dim=1)
    Lr_pattern = torch.stack(Lr_patterns, dim=1)
    Means_pattern = torch.stack(Means_patterns, dim=1)

    x_ticks = torch.arange(TEST_POINTS) * ALPHA_TICK

    fig,axes_all = plt.subplots(4,4)

    sequence = [
            (Means_noise, Means_pattern, Means_real),
            (L2_noise, L2_pattern, L2_real), 
            (Diag_noise, Diag_pattern, Diag_real), 
            (Lr_noise, Lr_pattern, Lr_real)
            ]
    titles = ["Mean", 'L2', "Diag", "SUPN"]


    #torch.save(Means_noise.unsqueeze(1), "fig_B_Means_noise.tensor")
    #torch.save(Means_pattern.unsqueeze(1), "fig_B_Means_pattern.tensor")
    #torch.save(Means_real.unsqueeze(1), "fig_B_Means_real.tensor")
    #torch.save(L2_noise.unsqueeze(1), "fig_B_L2_noise.tensor")
    #torch.save(L2_pattern.unsqueeze(1), "fig_B_L2_pattern.tensor")
    #torch.save(L2_real.unsqueeze(1), "fig_B_L2_real.tensor")
    #torch.save(Diag_noise.unsqueeze(1), "fig_B_Diag_noise.tensor")
    #torch.save(Diag_pattern.unsqueeze(1), "fig_B_Diag_pattern.tensor")
    #torch.save(Diag_real.unsqueeze(1), "fig_B_Diag_real.tensor")
    #torch.save(Lr_noise.unsqueeze(1), "fig_B_SUPN_noise.tensor")
    #torch.save(Lr_pattern.unsqueeze(1), "fig_B_SUPN_pattern.tensor")
    #torch.save(Lr_real.unsqueeze(1), "fig_B_SUPN_real.tensor")

    for col in range(4):

        dist_noise, dist_pattern, dist_real = sequence[col]
        dist_clean = dist_noise[0,0,0]
        axes = axes_all[:,col]
        title = titles[col]

        lr0 = axes[0].imshow(dist_clean.cpu(), cmap='gray')
        cb0 = fig.colorbar(lr0, ax=axes[0])
        cb0.ax.tick_params(labelsize=6)
        axes[0].set_xticks([]); axes[0].set_yticks([])

        axes[0].set_title(title)

        lr1 = axes[1].imshow(dist_noise[-1,0,0].cpu(), cmap='gray')
        cb1 = fig.colorbar(lr1, ax=axes[1])
        cb1.ax.tick_params(labelsize=6)
        axes[1].set_xticks([]); axes[1].set_yticks([])

        lr2 = axes[2].imshow(dist_pattern[-1,0,0].cpu(), cmap='gray')
        cb2 = fig.colorbar(lr2, ax=axes[2])
        cb2.ax.tick_params(labelsize=6)
        axes[2].set_xticks([]); axes[2].set_yticks([])

        lr3 = axes[3].imshow(dist_real[0,0].cpu(), cmap='gray')
        cb3 = fig.colorbar(lr3, ax=axes[3])
        cb3.ax.tick_params(labelsize=6)

    plt.show()


    labels_noise = {
        0 : "L2 noisy",
        1 : "Diag noisy",
        2 : "SUPN noisy"
    }
    colors_noise = {
        0 : "turquoise",
        1 : "red",
        2 : "gray"
    }
    axes = plot_dpoint_lines(
        Lr_noise.sum((2,3,4)).cpu().log(), 
        Diag_noise.sum((2,3,4)).cpu().log(), 
        L2_noise.sum((2,3,4)).cpu().log(), 
        x_ticks,
        alpha=0.03,
        linewidth=3,
        colors=colors_noise,
        labels=labels_noise)

    labels_pattern = {
        0 : "L2 smooth",
        1 : "Diag smooth",
        2 : "SUPN smooth"
    }
    colors_pattern = {
        0 : "blue",
        1 : "orange",
        2 : "darkcyan"
    }
    axes = plot_dpoint_lines(
        Lr_pattern.sum((2,3,4)).cpu().log(), 
        Diag_pattern.sum((2,3,4)).cpu().log(), 
        L2_pattern.sum((2,3,4)).cpu().log(), 
        x_ticks,
        alpha=0.03,
        linewidth=3,
        labels=labels_pattern,
        colors=colors_pattern,
        passed_axes=axes)

    axes[0].set_xlabel('alpha')
    axes[0].set_ylabel('log-Mahalanobis')

    common_ylim = (L2_pattern.sum((2,3,4)).log().min().item(), Lr_noise.sum((2,3,4)).mean(1).max().log().item())
    plt.setp(axes, ylim=common_ylim)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
