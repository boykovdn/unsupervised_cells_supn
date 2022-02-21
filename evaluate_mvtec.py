import torch
import yaml
import argparse
from evaluation import scoring_table
from dataset import FullFrames, MvtecClass
import pandas as pd
from structured_uncertainty.utils import retile_mask
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

def eval_spade_padim_patchcore_performance():
    r"""
    Trying out one of the well-performing MVTEC models, using implementation
    from the following repository:

    https://github.com/rvorias/ind_knn_ad

    This implementation was chosen due to the simplicity of running the
    training and evaluation. No further hyperparameter tuning was done, and 
    the default settings of the model are used.
    """
    from indad.models import SPADE, PaDiM, PatchCore

    RAW_PATH = "./fl_dataset/healthy_train/raw"
    GT_PATH = "./fl_dataset/healthy_train/mask"
    ANOM_PATH_RAW = "./fl_dataset/infected/raw"
    ANOM_PATH_GT = "./fl_dataset/infected/mask"
    NORM_PATH_RAW = "./fl_dataset/healthy_test/raw"
    NORM_PATH_GT = "./fl_dataset/healthy_test/mask"

    IMAGENET_MEAN = torch.Tensor([.485, .456, .406])
    IMAGENET_STD = torch.Tensor([.229, .224, .225])

    model_creators = {
        'padim' : lambda : PaDiM(backbone_name='wide_resnet50_2', d_reduced=250),
        'spade' : lambda : SPADE(k=50, backbone_name='wide_resnet50_2'),
        'patch' : lambda : PatchCore(backbone_name='wide_resnet50_2', f_coreset=0.1)
        }

    # Rescaling to between 0 and 1 because the ImageNet dataset is in that
    # range before the other transformations.
    tforms=tv_transforms.Compose([
        lambda x : rescale_to(x, to=(0,1)),
        lambda x : x.expand(3,-1,-1),
        tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # debug=100 means the models will fit on the first 100 images of the
    # training set. Going above that might result in running out of memory.
    dset_train = FullFrames(RAW_PATH, GT_PATH, raw_transforms=tforms, debug=100)
    dloader = torch.utils.data.DataLoader(dset_train, batch_size=1, shuffle=True)

    tforms_test=tv_transforms.Compose([
        lambda x : rescale_to(x, to=(0,1)),
        lambda x : x.expand(3,-1,-1),
        tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        lambda x : x.unsqueeze(0)])
 
    dset_anom = FullFrames(ANOM_PATH_RAW, ANOM_PATH_GT, raw_transforms=tforms_test)
    dset_norm = FullFrames(NORM_PATH_RAW, NORM_PATH_GT, raw_transforms=tforms_test)

    def get_metrics(dset_anom, dset_norm, model):
        anom_scores = []
        for dpoint, _ in dset_anom:
            anom_scores.append(model.predict(dpoint)[0].item())

        norm_scores = []
        for dpoint, _ in dset_norm:
            norm_scores.append(model.predict(dpoint)[0].item())

        true_labels = np.array([1] * len(dset_anom) + [0] * len(dset_norm))
        pred_scores = np.concatenate((anom_scores, norm_scores))

        return scoring_table(true_labels, pred_scores)

    # Padim crashes, too much memory requirement.
    names = ('spade', 'padim', 'patch')
    metrics = []
    for name in names:
        model = model_creators[name]()
        print("Fitting {}...".format(name))
        model.fit(dloader)

        metrics.append(get_metrics(dset_anom, dset_norm, model))

        del model

    return tuple(metrics)

def evaluate_mvtec_class_performance(dir_norm, dir_anom, model, dset_tforms, device=0):
    r"""
    """
    # Non-recursive loading of the 'good' test dataset.
    dset_norm = MvtecClass(dir_norm, recursive_except=None, ext=".png",
            transforms=dset_tforms, load_with_rgb=True)
    dset_anom = MvtecClass(dir_anom, recursive_except=['good'], ext=".png",
            transforms=dset_tforms, load_with_rgb=True)

    def get_scores(dset, model):
        r"""
        Dset here can load rgb images as well.
        """
        scores = []
        for dpoint in dset:
            dpoint = dpoint[:,None].to(device) # Channels turn into batch
            x_mu, x_chol, _, _ = model(dpoint)
            mahs = mahalanoubis_dist(dpoint, x_mu, x_chol, conn=model.connectivity).detach()
            mahs = mahs.sum((1,2,3))
            scores.append(max(mahs))

        return scores

    anom_labels = np.ones(len(dset_anom))
    anom_scores = np.array(get_scores(dset_anom, model))
    norm_labels = np.zeros(len(dset_norm))
    norm_scores = np.array(get_scores(dset_norm, model))

    labels = np.concatenate((anom_labels, norm_labels))
    scores = np.concatenate((anom_scores, norm_scores))

    metrics = scoring_table(labels, scores)

    return metrics

def reproduce_spade_padim_patchcore_performance(mvtec_path):
    r"""
    Calculates the metrics for the three leading MVTec models.

    Args:
        :mvtec_path: str or Path
    """
    from indad.models import SPADE, PaDiM, PatchCore

    IMAGENET_MEAN = torch.Tensor([.485, .456, .406])
    IMAGENET_STD = torch.Tensor([.229, .224, .225])

    MVTEC_CLASSES = [        
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    model_creators = {
        'padim' : lambda : PaDiM(backbone_name='wide_resnet50_2', d_reduced=250),
        'spade' : lambda : SPADE(k=50, backbone_name='wide_resnet50_2'),
        'patch' : lambda : PatchCore(backbone_name='wide_resnet50_2', f_coreset=0.1)
        }

    TRAIN_PATH = mvtec_path + "/{}/train/good"
    ANOM_PATH_RAW = mvtec_path + "/{}/test"
    NORM_PATH_RAW = mvtec_path + "/{}/test/good"

    tforms=tv_transforms.Compose([
        lambda x : rescale_to(x, to=(0,1)),
        tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    # Test transforms same as above, but with added batch dimension as final
    # transform.
    tforms_test=tv_transforms.Compose([
        lambda x : rescale_to(x, to=(0,1)),
        tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
        tv_transforms.CenterCrop(224),
        tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        lambda x : x.unsqueeze(0)])

    def get_metrics(dset_anom, dset_norm, model):
        anom_scores = []
        for dpoint, _ in dset_anom:
            anom_scores.append(model.predict(dpoint)[0].item())

        norm_scores = []
        for dpoint, _ in dset_norm:
            norm_scores.append(model.predict(dpoint)[0].item())

        true_labels = np.array([1] * len(dset_anom) + [0] * len(dset_norm))
        pred_scores = np.concatenate((anom_scores, norm_scores))

        return scoring_table(true_labels, pred_scores)

    model_names = ('patch', 'padim', 'spade')
    auprc_col = []
    auroc_col = []
    names_col = []
    class_col = []
    for mvtec_class in MVTEC_CLASSES:

        dset_anom = MvtecClass(ANOM_PATH_RAW.format(mvtec_class), 
                transforms=tforms_test, recursive_except=['good'], 
                fake_gt=True, load_with_rgb=True)
        dset_norm = MvtecClass(NORM_PATH_RAW.format(mvtec_class), 
                transforms=tforms_test, fake_gt=True, load_with_rgb=True)

        dset_train = MvtecClass(TRAIN_PATH.format(mvtec_class), 
                transforms=tforms, fake_gt=True, load_with_rgb=True)
        dloader = torch.utils.data.DataLoader(dset_train, batch_size=32, 
                shuffle=True)

        for model_name in model_names:

            print("Loading {}...".format(model_name))
            model = model_creators[model_name]()

            print("Fitting {}...".format(model_name))
            model.fit(dloader)

            print("Evaluating {}...".format(model_name))
            model_metrics = get_metrics(dset_anom, dset_norm, model)

            names_col.append(model_name)
            class_col.append(mvtec_class)
            auprc_col.append(model_metrics['auprc'])
            auroc_col.append(model_metrics['auroc'])

            del model

    metrics_df = pd.DataFrame({
        'model' : names_col,
        'class' : class_col,
        'auprc' : auprc_col,
        'auroc' : auroc_col
        })

    df_auprc = metrics_df.pivot_table(columns='model', values='auprc', index='class')
    df_auroc = metrics_df.pivot_table(columns='model', values='auroc', index='class')

    return {'auroc' : df_auroc, 'auprc' : df_auprc}

def main():

    SCRIPT_DESCRIPTION = "MVTec reproduction scripts. Require a local copy of the MVTec dataset"

    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument("--mvpath", type=str, help="Path to the MVTec dataset folder which contains all the\
            subclasses.")
    parser.add_argument("--calc_mode", type=str, help="Which part of the script to run {mv_reproduce | fl_eval}.")
    args = parser.parse_args()

    ### Reproduction script for MVtec methods on Mvtec
    if args.calc_mode == "mv_reproduce":
        spp_mvtec_metrics = reproduce_spade_padim_patchcore_performance(args.mvpath)
        print("PaDiM, PatchCore, SPADE MVTec reproduction")
        print(spp_mvtec_metrics)

    ### Evaluation script for Mvtec methods on cells without modification (but
    # transforms applied as closely as possible).
    elif args.calc_mode == "fl_eval":
        metrics_spade, metrics_padim, metrics_patch = \
                eval_spade_padim_patchcore_performance()
        print("SPADE")
        print(metrics_spade)
        print("PaDiM")
        print(metrics_padim)
        print("PatchCore")
        print(metrics_patch)

    # TODO
    ### Mvtec and our autoencoder.
    #MVTEC_NAMES = ['capsule', 'grid', 'hazelnut', 'metal_nut', 'screw',
    #        'toothbrush', 'transistor', 'zipper', 'leather',
    #        'carpet']
    #mvtec_cols = {
    #        'dataset' : [],
    #        'auprc' : [],
    #        'auroc' : [],
    #        'resto_dataset' : [],
    #        'resto_auprc' : [],
    #        'resto_auroc' : []
    #        }
    #for subset_name in tqdm(MVTEC_NAMES, desc='Mvtec evals'):
    #    MVTEC_NORM = "./mvtec/{}/test/good".format(subset_name)
    #    MVTEC_ANOM = "./mvtec/{}/test".format(subset_name)
    #    MVTEC_MODEL = "./experiments/mvtec_{}_supn/model00.model".format(subset_name)
    #    mvtec_model = torch.load(MVTEC_MODEL, map_location=torch.device(device_name))
    #    mvtec_resto_model = Restoration_model_wrap(mvtec_model)

    #    mv_metrics_ = evaluate_mvtec_class_performance(MVTEC_NORM, MVTEC_ANOM, mvtec_model, tv_transforms.Compose([lambda x : rescale_to(x, to=(-1,1))]), device=device_name)
    #    mv_metrics_resto = evaluate_mvtec_class_performance(MVTEC_NORM, MVTEC_ANOM, mvtec_resto_model, tv_transforms.Compose([lambda x : rescale_to(x, to=(-1,1))]), device=device_name)

    #    mvtec_cols['dataset'].append(subset_name)
    #    mvtec_cols['auprc'].append(mv_metrics_['auprc'])
    #    mvtec_cols['auroc'].append(mv_metrics_['auroc'])
    #    mvtec_cols['resto_dataset'].append(subset_name)
    #    mvtec_cols['resto_auprc'].append(mv_metrics_resto['auprc'])
    #    mvtec_cols['resto_auroc'].append(mv_metrics_resto['auroc'])

if __name__ == "__main__":
    main()
