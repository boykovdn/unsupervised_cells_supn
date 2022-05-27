import torch
import matplotlib.pyplot as plt
import imageio
from tqdm.auto import tqdm
import torchvision.transforms as tv_transforms
import dataset
from transforms import (rescale_to, 
        random_channel_select, 
        add_gaussian_noise, 
        joint_random_cell_crop)

#from supn._jacobi_sampler import jacobi_sampler
from structured_uncertainty.new_sparse_supn_sampler import apply_sparse_solve_sampling

from utils import (
        run_through_vae_restoration,
        load_model,
        Restoration_model_wrap,
        L2_model_wrap,
        Diag_model_wrap)

def main():
    ## Original MIDL SUPN model
    #MODEL_CHOL_PATH = "/u/homes/biv20/experiments/midl03_fl_supn_d100_reg1000/model00_fl_supn.model"
    #MODEL_DIAG_PATH = "/u/homes/biv20/experiments/midl03_fl_diag_d100_111221/model00_fl_diag.model"
    #MODEL_CHOL_PATH = "/u/homes/biv20/experiments/exp45_fl_supn_d100_conn2/model00_fl_supn.model"
    MODEL_DIAG_PATH = "/u/homes/biv20/repos/unsupervised_cells_supn/experiments/1603_exps_diag_1/config.yaml"
    #MODEL_CHOL_PATH = "/u/homes/biv20/repos/unsupervised_cells_supn/oldsupn.yaml"
    MODEL_CHOL_PATH = "/u/homes/biv20/repos/unsupervised_cells_supn/experiments/1603_exps_supn_l1_0_1/config.yaml"
    RAW_PATH = "/u/homes/biv20/datasets/dual_stain_infected/raw"
    GT_PATH = "/u/homes/biv20/datasets/dual_stain_infected/mask"

    DEVICE = 0
    FIXED_VAR = 0.11
    DSET_INPUT_IDX = 0
    TEST_POINTS = 5
    N_DPOINTS = 1
    N_SAMPLES = 100

    tforms = tv_transforms.Compose([
        lambda x : rescale_to(x, to=(-1,1)),
        tv_transforms.GaussianBlur(3, sigma=2.0)
        ])

    dset = dataset.FullFrames(RAW_PATH, GT_PATH, raw_transforms=tforms, debug=100)
    model_chol = Restoration_model_wrap(
                load_model(MODEL_CHOL_PATH, map_location="cuda:{}".format(DEVICE), 
            pretrained_mean=False))
    model_diag = Restoration_model_wrap(
            Diag_model_wrap(
                load_model(MODEL_DIAG_PATH, map_location="cuda:{}".format(DEVICE), 
            pretrained_mean=False)))
    model_l2 = Restoration_model_wrap( L2_model_wrap(FIXED_VAR,
                load_model(MODEL_CHOL_PATH, map_location="cuda:{}".format(DEVICE), 
            pretrained_mean=False)))

    chol_connectivity = model_chol.connectivity

    tmp_, _ = dset[DSET_INPUT_IDX] # (1,H,W)
    _,H,W = tmp_.shape
    input_imgs = torch.zeros(TEST_POINTS, N_DPOINTS,1,H,W)
    masks_ = torch.zeros(TEST_POINTS, N_DPOINTS,1,H,W)
    for idx in range(TEST_POINTS):
        raw_, mask_ = dset[DSET_INPUT_IDX + idx]
        input_imgs[idx] = raw_
        masks_[idx] = mask_

    L2_square, Diag_square, Lr_square, Means, Chols = run_through_vae_restoration(
            input_imgs[:,0], model_l2, model_diag, model_chol, FIXED_VAR,
            DEVICE=DEVICE, return_means=True, return_chols=True)

    #torch.save(input_imgs.cpu(), "figA_Input.tensor")
    #torch.save(Means.cpu(), "figA_Means.tensor")
    #torch.save(Chols.cpu(), "figA_Chols.tensor") 
    ### Produce residuals figure here ###
    fig, axes = plt.subplots(TEST_POINTS,6)
    for axidx in range(TEST_POINTS):
        inp=axes[axidx,0].imshow(input_imgs[axidx,0,0].cpu(), cmap='gray', vmin=-1,vmax=1)
        #cbar0 = fig.colorbar(inp, ax=axes[0,axidx])
        #cbar0.ax.tick_params(labelsize=6)
        mean=axes[axidx,1].imshow(Means[axidx,0].cpu(), cmap='gray', vmin=-1,vmax=1)
        #cbar1 = fig.colorbar(mean,ax=axes[1,axidx])
        #cbar1.ax.tick_params(labelsize=6)
        sph=axes[axidx,2].imshow(L2_square[axidx,0].cpu(), cmap='viridis', vmin=0, vmax=8)
        #cbar2 = fig.colorbar(sph, ax=axes[2,axidx])
        #cbar2.ax.tick_params(labelsize=6)
        diag=axes[axidx,3].imshow(Diag_square[axidx,0].cpu(), cmap='viridis', vmin=0, vmax=8)
        #cbar3 = fig.colorbar(diag, ax=axes[3,axidx])
        #cbar3.ax.tick_params(labelsize=6)
        supn=axes[axidx,4].imshow(Lr_square[axidx,0].cpu(), cmap='viridis', vmin=0, vmax=8)
        #cbar4 = fig.colorbar(supn, ax=axes[4,axidx])
        #cbar4.ax.tick_params(labelsize=6)
        tmp_ = apply_sparse_solve_sampling(Chols[axidx,0,None].unsqueeze(0).double().cpu(), Chols[axidx,None,1:].double().cpu(), chol_connectivity, 1)
        axes[axidx,5].imshow(Means[axidx,0].cpu() + tmp_[0,0,0].cpu(), cmap='gray', vmin=-1, vmax=1)

        for tickidx in range(0, 6):
            axes[axidx,tickidx].set_xticks([])
            axes[axidx,tickidx].set_yticks([])

    axes[0,0].set_title("Input")
    axes[0,1].set_title("Mean")
    axes[0,2].set_title("Spherical")
    axes[0,3].set_title("Diagonal")
    axes[0,4].set_title("SUPN")
    axes[0,5].set_title("Sample")

    plt.show()

if __name__ == "__main__":
    main()
