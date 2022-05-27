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
        L2_model_wrap,
        Diag_model_wrap,
        Restoration_model_wrap,
        conn_to_nonzeros,
        apply_sparse_chol_rhs_matmul)

def main():
    RAW_PATH = "/u/homes/biv20/datasets/Kate_BF_healthy_sparse_crops/test_raw"
    GT_PATH = "/u/homes/biv20/datasets/Kate_BF_healthy_sparse_crops/test_mask"

    CHOL_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/oldsupn_bf.yaml"
    #CHOL_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/example_train_bf_supn.yaml"
    DIAG_CONFIG = "/u/homes/biv20/repos/unsupervised_cells_supn/example_train_bf_diag.yaml"

    DEVICE = 0
    FIXED_VAR = 0.06
    DSET_INPUT_IDX = 50
    TEST_POINTS = 5
    N_SAMPLES = 100

    tforms = tv_transforms.Compose([
        lambda x : rescale_to(x, to=(-1,1)),
        tv_transforms.GaussianBlur(3, sigma=2.0)
        ])

    # Dataset load
    dset = dataset.FullFrames(RAW_PATH, GT_PATH, raw_transforms=tforms, debug=100)
    # Model load and wrap
    model_chol = Restoration_model_wrap( 
            load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False))
    model_diag = Restoration_model_wrap( 
            Diag_model_wrap( 
                load_model(DIAG_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False)
                ))
    model_l2 = Restoration_model_wrap( L2_model_wrap( FIXED_VAR, 
        load_model(CHOL_CONFIG, map_location="cuda:{}".format(DEVICE), pretrained_mean=False)))

    tmp_, _ = dset[DSET_INPUT_IDX] # (1,H,W)
    _,H,W = tmp_.shape
    input_imgs = torch.zeros(TEST_POINTS, 1,H,W)
    # Load subset of data.
    for idx in range(TEST_POINTS):
        raw_, _ = dset[DSET_INPUT_IDX + idx]
        input_imgs[idx] = raw_

    # Calculate Mahalanobis distances as images.
    L2_square, Diag_square, Lr_square, Means, Chols = run_through_vae_restoration(
            input_imgs, model_l2, model_chol, model_diag, FIXED_VAR, 
            DEVICE=DEVICE, return_means=True, return_chols=True)

    # Save intermediate results on disk.
    import pdb; pdb.set_trace()
    torch.save(input_imgs.unsqueeze(1).cpu(), "figA_Input.tensor")
    torch.save(Means.unsqueeze(1).cpu(), "figA_Means.tensor")
    torch.save(Chols.unsqueeze(1).cpu(), "figA_Chols.tensor") 
    torch.save(L2_square.unsqueeze(1).cpu(), "l2_dist.tensor")
    torch.save(Diag_square.unsqueeze(1).cpu(), "Diag_dist.tensor")
    torch.save(Lr_square.unsqueeze(1).cpu(), "SUPN_dist.tensor")

    ### Produce residuals figure here ###
    fig, axes = plt.subplots(5,TEST_POINTS)
    for axidx in range(TEST_POINTS):
        inp=axes[0,axidx].imshow(input_imgs[axidx,0].cpu(), cmap='gray',
                vmin=-1, vmax=1)
        cbar0 = fig.colorbar(inp, ax=axes[0,axidx])
        cbar0.ax.tick_params(labelsize=6)
        mean=axes[1,axidx].imshow(Means[axidx,0].cpu(), cmap='gray',
                vmin=-1, vmax=1)
        cbar1 = fig.colorbar(mean,ax=axes[1,axidx])
        cbar1.ax.tick_params(labelsize=6)
        sph=axes[2,axidx].imshow(L2_square[axidx,0].cpu(), cmap='gray')
        cbar2 = fig.colorbar(sph, ax=axes[2,axidx])
        cbar2.ax.tick_params(labelsize=6)
        diag=axes[3,axidx].imshow(Diag_square[axidx,0].cpu(), cmap='gray')
        cbar3 = fig.colorbar(diag, ax=axes[3,axidx])
        cbar3.ax.tick_params(labelsize=6)
        supn=axes[4,axidx].imshow(Lr_square[axidx,0].cpu(), cmap='gray')
        cbar4 = fig.colorbar(supn, ax=axes[4,axidx])
        cbar4.ax.tick_params(labelsize=6)
        for tickidx in range(0, 5):
            axes[tickidx,axidx].set_xticks([])
            axes[tickidx,axidx].set_yticks([])

    axes[0,0].set_ylabel("Raw")
    axes[1,0].set_ylabel("Mean")
    axes[2,0].set_ylabel("Spherical")
    axes[3,0].set_ylabel("Diagonal")
    axes[4,0].set_ylabel("SUPN")

    plt.show()

if __name__ == "__main__":
    main()
