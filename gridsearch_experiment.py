from utils import (
        make_dirs_if_absent, 
        get_timestamp_string, 
        load_model,
        L2_model_wrap,
        Diag_model_wrap,
        Restoration_model_wrap,
        mahalanobis_dist,
        get_log_prob_from_sparse_L_precision)
from models import run, TRAINING_STOP_REASONS, get_random_latent_sample
from tqdm.auto import tqdm
from sklearn.model_selection import ParameterGrid
import argparse
import logging
import yaml
import os
import torchvision.transforms as tv_transforms
from dataset import FullFrames
from evaluate_fl import (
        vae_model_scoring, 
        vae_dataset_likelihood, 
        scoring_func_outliers)
from transforms import rescale_to
import pandas as pd
import imageio
import torch
import matplotlib.pyplot as plt

from new_sparse_supn_sampler import apply_sparse_solve_sampling
from pylatex import Document, NoEscape, Figure

params = {
        "L1_REG_WEIGHT" : float,
        "DEPTH" : int,
        "ENCODING_DIMENSION": int,
        "LEARNING_RATE" : float,
        "FIXED_VAR" : float,
        "SLIDING_WINDOW_SIZE" : int
        }

params_eval = {
        "healthy_test_raw" : str,
        "healthy_test_mask" : str,
        "infected_raw" : str,
        "infected_mask" : str
        }

def log_and_save(conf, logger=None, training_outcome=None):

    with open("{}/{}/config.yaml".format(
            conf["EXPERIMENT_DIR"], 
            conf["EXPERIMENT_FOLDER"]), "w+") as fout:
        yaml.dump(conf, fout)

    if logger is not None:
        logger.info("Finished {} training due to {}...".format( conf["TRAINING_TYPE"], TRAINING_STOP_REASONS[training_outcome] ))

def run_sequential(conf, followed_by='supn', logger=None):
    r"""
    Trains first the mean of the model, then the supn or diag, specified in the
    function call. The second stage cannot be executed without having a trained
    mean model.
    """
    root_exp_folder = conf["EXPERIMENT_FOLDER"]

    # Run training of the mean
    conf["MODEL_NAME"] = "model_mean"
    conf["TRAINING_TYPE"] = "mean"
    conf["EXPERIMENT_FOLDER"] = "{}/{}".format(root_exp_folder, "mean")
    conf["PRETRAINED_MODEL_PATH"] = None

    print(conf)
    training_outcome_mean = None
    while training_outcome_mean not in [0,1]:
        training_outcome_mean = run(conf, logger=logger)

    log_and_save(conf, logger=logger, training_outcome=training_outcome_mean)

    # Run training of the second step
    conf["MODEL_NAME"] = "model_{}".format(followed_by)
    conf["TRAINING_TYPE"] = followed_by
    conf["EXPERIMENT_FOLDER"] = "{}/{}".format(root_exp_folder, followed_by)
    conf["PRETRAINED_MODEL_PATH"] = "{}/{}/mean/model_mean.state".format(
            conf["EXPERIMENT_DIR"], root_exp_folder)

    print(conf)
    training_outcome_supn = run(conf, logger=logger)
    # We do not expect supn to train poorly, so only outcome should be 
    # due to reaching max iterations.
    log_and_save(conf, logger=logger, training_outcome=training_outcome_supn)

def setup_logging(output_name):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.FileHandler(output_name, mode='w')
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def load_model_type(path_config, model_type, map_location='cuda:0',
        restoration=False, l2_fixed_var=0.06):
    r"""
    Loads different types of model, parametrized by the model type and whether
    to add a restoration step.
    """
    model_ = load_model(path_config, map_location=map_location,
            pretrained_mean=False)

    if model_type == 'diag':
        model_ = Diag_model_wrap(model_)

    elif model_type == 'l2':
        # Does not matter which model is taken as the L2 model, both have equally
        # good mean predictors, and the covariance is an identity. Need to pass
        # the empirical variance as a parameter along with the model.
        model_ = L2_model_wrap(l2_fixed_var, model_)

    if restoration:
        model_ = Restoration_model_wrap(model_)

    return model_

def get_healthy_infected_dsets(
        PATH_HEALTHY_RAW, PATH_HEALTHY_GT,
        PATH_INFECTED_RAW, PATH_INFECTED_GT):
    r"""
    Loads the dataset objects with some standard transformations
    """
    # Transforms
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

    return dset_healthy, dset_infected

def update_doc_with_samples(model, doc, n_samples=5, caption=None):
    r"""
    Add visualization images to the pdf doc.

    1. Random latent sample decoded into the observation space.
    """
    mu_samples, chol_samples = get_random_latent_sample(model, n=n_samples)
    device = mu_samples.device

    # noise sample
    noise_ = torch.zeros_like(mu_samples) # (n_samples,1,H,W)

    for idx in range(n_samples):
        noise_[idx] = apply_sparse_solve_sampling(
                chol_samples[idx,None,0].unsqueeze(1).double().cpu(),
                chol_samples[idx,None,1:].double().cpu(),
                model.connectivity,
                1)[-1,0]

    samples = mu_samples + noise_.to(device)

    with doc.create(Figure()) as sample_illustration:
        fig, axes = plt.subplots(1,len(samples))
        for idx, sample in enumerate(samples):
            ax = axes[idx]
            im = ax.imshow(sample[0].cpu())
            fig.colorbar(im, ax=ax)

        sample_illustration.add_plot()

        if caption is not None:
            sample_illustration.add_caption(caption)

def update_doc_with_reconstructions(
        model, dset, doc, n_images=5, 
        caption=None, device=0, dset_idx_offset=0):
    r"""
    Reconstruct a number of images and show Mahalanobis distance, diagonal of L
    and the log-likelihood.
    """
    with doc.create(Figure()) as rec_fig:

        fig, axes = plt.subplots(n_images, 4)

        with torch.no_grad():

            for idx in range(n_images):
                dpoint, mask = dset[idx + dset_idx_offset]
                dpoint = dpoint[None,].to(device)

                x_mu, x_chol, _, _ = model(dpoint)

                # [1,1,H,W]
                log_l_ = get_log_prob_from_sparse_L_precision(
                        dpoint, 
                        x_mu, 
                        model.connectivity,
                        x_chol[:,0,None],
                        x_chol[:,1:],
                        pixelwise=True)

                mah_ = mahalanobis_dist(
                        dpoint, 
                        x_mu, 
                        x_chol, 
                        conn=model.connectivity)

                ax0 = axes[idx, 0]
                im0 = ax0.imshow(dpoint[0,0].cpu()) 
                ax0.set_title("Raw")
                fig.colorbar(im0, ax=ax0)

                ax1 = axes[idx, 1]
                im1 = ax1.imshow(mah_[0,0].cpu()) 
                ax1.set_title("Mah")
                fig.colorbar(im1, ax=ax1)

                ax2 = axes[idx, 2]
                im2 = ax2.imshow(log_l_[0,0].cpu()) 
                ax2.set_title("Log l")
                fig.colorbar(im2, ax=ax2)

                ax3 = axes[idx, 3]
                im3 = ax3.imshow(x_chol[0,0].cpu()) 
                ax3.set_title("Chol diag")
                fig.colorbar(im3, ax=ax3)

        rec_fig.add_plot()
        rec_fig.add_caption(caption)

def summarize_gridsearch_performance(
        path_healthy_raw, path_healthy_gt,
        path_infected_raw, path_infected_gt,
        root_folder="./",
        model_type="supn",
        config_name="config.yaml",
        restoration=False,
        samples=10,
        device=0,
        TILE_FACTOR=2,
        report_path="./{}_summary_report"):
    r"""
    Alternative mode for the the script, aggregates information from multiple
    folders with a set gridsearch (enumerated) structure, returns a summary.
    The healthy and infected datasets used here were never seen by the model,
    so they are not present in the config file and we have to pass their paths
    as parameters.
    """
    config_paths = []

    scoring_func = lambda x_mu, x_chol, x, mask, conn : scoring_func_outliers(
        x_mu, x_chol, x, mask, TILE_FACTOR, conn=conn)

    # Search for config files, add to a list
    for root, folders, files in os.walk(root_folder):
        if config_name in files and model_type in root.split("/"):
            config_paths.append("{}/{}".format(root, config_name))

    print("Found {} {} configs...".format(len(config_paths), model_type))

    # Parameters lists (named via dict)
    grid_params = {}
    for param_name in params.keys():
        grid_params[param_name] = []
    # Metrics lists
    aurocs = []
    auprcs = []
    logl_hs = []
    logl_is = []

    doc = Document()

    for config_path in tqdm(config_paths):
        config_dict = {}
        with open(config_path) as fin:
            config_dict = yaml.load(fin, Loader=yaml.Loader)
        l2_fixed_var = config_dict["FIXED_VAR"]

        dset_healthy, dset_infected = get_healthy_infected_dsets(
                path_healthy_raw, path_healthy_gt,
                path_infected_raw, path_infected_gt)

        model = load_model_type(config_path, model_type,
                map_location="cuda:{}".format(device), restoration=restoration,
                l2_fixed_var=l2_fixed_var)

        # Add likelihood to lists for dataframe.
        dset_likelihood_healthy = vae_dataset_likelihood(dset_healthy, model)
        dset_likelihood_infected = vae_dataset_likelihood(dset_infected, model)
        logl_hs.append( dset_likelihood_healthy )
        logl_is.append( dset_likelihood_infected )

        # Classification metrics
        metrics = vae_model_scoring(dset_healthy, dset_infected,
            model, scoring_func, device=device, samples=samples,
            model_label="{}, Resto: {}".format(model_type, restoration))
        
        # Add relevant model parameters from the config file.
        for param_name in params.keys():
            grid_params[param_name].append(config_dict[param_name])

        # Record performance metrics.
        aurocs.append(metrics['auroc'])
        auprcs.append(metrics['auprc'])

        update_doc_with_samples(model, doc, n_samples=5, 
                caption="Random samples of {}".format(config_path))
        update_doc_with_reconstructions(model, dset_healthy, doc, n_images=5, 
                caption="Healthy reconstructions of {}".format(config_path))
        update_doc_with_reconstructions(model, dset_infected, doc, n_images=5, 
                caption="Infected reconstructions of {}".format(config_path))


    grid_params['auroc'] = aurocs
    grid_params['auprc'] = auprcs
    grid_params['logl_healthy'] = logl_hs
    grid_params['logl_infected'] = logl_is
    df_results = pd.DataFrame(grid_params)

    with doc.create(Figure()) as scatter_fig:
        pd.plotting.scatter_matrix(df_results[[
            "auroc", 
            "auprc", 
            "logl_healthy", 
            "logl_infected",
            "ENCODING_DIMENSION"]])

        scatter_fig.add_plot()
        scatter_fig.add_caption("Scatter plot for experiment, interested in how \
                likelihood, discriminative performance and the bottleneck size\
                affect each other.")
        
    # Evaluation seems to work sensibly.
    print(df_results)
    # Output metrics as csv table. Easier to format than latex..
    df_results.to_csv(path_or_buf="{}/result_table.csv".format(root_folder))
    # Save to the experiment root folder.
    doc.generate_pdf("{}/report".format(root_folder))

def main():

    parser = argparse.ArgumentParser(
            description="Run a grid evaluation of mean -> {supn, diag} trainings.")

    # Add the parameters that will be searched over (if multiple are passed).
    for param, ptype in params.items():
        parser.add_argument("--{}".format(param), 
                type=ptype, 
                nargs="+",
                help="(possibly empty) parameter list.", 
                required=False)

    # Add parameters for the experiment summary running mode. For example, the
    # location of the healthy holdout and infected test data, which are not in
    # the config file.
    for param, ptype in params_eval.items():
        parser.add_argument("--{}".format(param),
                type=ptype,
                help="Optional param for when running in eval mode.",
                required=False)

    # Add misc parameters (folder naming, initial yaml config, ...)
    parser.add_argument("--experiment_tag", 
            type=str, 
            help="Naming for the experiment folders", 
            required=False)
    parser.add_argument("--common_params_yaml", 
            type=str, 
            help="Path to a yaml config that sets most of the relevant parameters.", 
            required=False)

    parser.add_argument("--summarize_experiment", 
            type=str,
            help="Path to an experiment folder to summarize",
            required=False)
    parser.add_argument("--restoration", action='store_true',
            dest='restoration',
            help="If summarizing experiment, whether to use restoration")
    parser.add_argument("--no-restoration", action='store_false', 
            dest='restoration',
            help="If summarizing experiment, whether to use restoration")
    parser.add_argument("--local_minimum_detection_threshold",
            type=float,
            help="A threshold or the variance of the reconstructed samples in\
                    image space. If less than this is reached throughout the\
                    training process, then the training script assumes that the\
                    model is stuck in a local minimum and will restart the\
                    training.")

    parser.set_defaults(restoration=False)
    args = parser.parse_args()

    if args.summarize_experiment is not None:
        #PATH_HEALTHY_RAW = "/u/homes/biv20/repos/unsupervised_cells_supn/fl_dataset/healthy_test/raw"
        #PATH_HEALTHY_GT = "/u/homes/biv20/repos/unsupervised_cells_supn/fl_dataset/healthy_test/mask"
        #PATH_INFECTED_RAW = "/u/homes/biv20/repos/unsupervised_cells_supn/fl_dataset/infected/raw"
        #PATH_INFECTED_GT = "/u/homes/biv20/repos/unsupervised_cells_supn/fl_dataset/infected/mask"

        PATH_HEALTHY_RAW = args.healthy_test_raw
        PATH_HEALTHY_GT = args.healthy_test_mask
        PATH_INFECTED_RAW = args.infected_raw
        PATH_INFECTED_GT = args.infected_mask

        TILE_FACTOR=2

        summarize_gridsearch_performance(
                PATH_HEALTHY_RAW, PATH_HEALTHY_GT,
                PATH_INFECTED_RAW, PATH_INFECTED_GT,
                root_folder=args.summarize_experiment,
                restoration=args.restoration,
                TILE_FACTOR=TILE_FACTOR)

        return

    # Initialize config from the passed yaml file.
    conf_path = args.common_params_yaml
    with open(conf_path, "r") as fin:
        conf = yaml.load(fin, Loader=yaml.Loader)

    # Create a separate directory for this experiment array.
    EXPERIMENT_TAG = "{}_Experiment_Array_{}".format(
        get_timestamp_string(),
        args.experiment_tag)
    conf["EXPERIMENT_DIR"] = "{}/{}".format(
            conf["EXPERIMENT_DIR"], 
            EXPERIMENT_TAG)
    make_dirs_if_absent([conf["EXPERIMENT_DIR"]])

    # Change local minimum parameter
    conf["LOCAL_MINIMUM_DETECTION_THRESHOLD"] = \
            args.local_minimum_detection_threshold

    # Turn parameter lists (or Nones) into a grid.
    candidate_parameters = {}
    for param in params.keys():
        param_val = getattr(args, param)
        if not isinstance(param_val, list):
            param_val = [param_val]
        candidate_parameters[param] = param_val
    print("Calculate for:\n{}".format(candidate_parameters))
    param_grid = ParameterGrid(candidate_parameters)

    logger = setup_logging("{}/training.log".format(conf["EXPERIMENT_DIR"]))

    # Grid evaluation loop
    for group_idx, param_dict in enumerate(param_grid):
        logger.info("Parameter group {:03d}".format(group_idx))
        logger.info(str(param_dict))
        
        conf["EXPERIMENT_FOLDER"] = "param_group_{:03d}".format(group_idx)
        for key,val in param_dict.items():
            conf[key] = val

        try:
            run_sequential(conf, followed_by='supn', logger=logger)
        except KeyboardInterrupt:
            return
        except Exception as ex:
            logger.error(ex)
            print(ex)

if __name__ == "__main__":
    main()
