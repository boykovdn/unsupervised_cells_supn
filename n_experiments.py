from utils import make_dirs_if_absent
from models import run
from tqdm.auto import tqdm
import yaml
from contextlib import redirect_stdout
import argparse

def main():

    parser = argparse.ArgumentParser(description="Run a bunch of mean,supn,diag training loops.")
    parser.add_argument("--experiment_numbers", type=list, nargs="+", help="Numbers which will be appended to the experiment tag.")
    parser.add_argument("--experiment_tag", type=str, help="Naming for the experiment folders", required=True)
    parser.add_argument("--init_state_path", type=str, help="model state file that specifies the initial layer weights", required=True)
    parser.add_argument("--common_params_yaml", type=str, help="Path to a yaml config that sets most of the relevant parameters.", required=True)
    args = parser.parse_args()

    EXPERIMENT_TAG = args.experiment_tag
    EXP_NUMS = map(lambda x : int(x[0]), args.experiment_numbers)

    conf_path = args.common_params_yaml
    with open(conf_path, "r") as fin:
        conf = yaml.load(fin, Loader=yaml.Loader)

    conf["L1_REG_WEIGHT"] = 0 # These experiments have no L1 regularization in SUPN training mode.
    # Training loop
    for num in tqdm(EXP_NUMS):
        # Mean training
        experiment_dir = conf["EXPERIMENT_DIR"]
        exp_folder_name_mean = "{}_mean_{}".format(
                EXPERIMENT_TAG, num)

        conf["TRAINING_TYPE"] = "mean"
        conf["EXPERIMENT_FOLDER"] = exp_folder_name_mean
        conf["PRETRAINED_MODEL_PATH"] = args.init_state_path
        conf["MODEL_NAME"] = "model_mean"

        run(conf)
        with open("{}/{}/config.yaml".format(experiment_dir, exp_folder_name_mean), "w+") as fout:
            yaml.dump(conf, fout)

        # SUPN training
        exp_folder_name_supn = "{}_supn_l1_0_{}".format(
                EXPERIMENT_TAG, num)

        conf["MODEL_NAME"] = "model_supn"
        conf["PRETRAINED_MODEL_PATH"] = "{}/{}/model_mean.state".format(experiment_dir, 
                exp_folder_name_mean)
        conf["TRAINING_TYPE"] = "supn"
        conf["EXPERIMENT_FOLDER"] = exp_folder_name_supn

        run(conf)
        with open("{}/{}/config.yaml".format(experiment_dir, exp_folder_name_supn), "w+") as fout:
            yaml.dump(conf, fout)

        # Diag training
        exp_folder_name_diag = "{}_diag_{}".format(
                EXPERIMENT_TAG, num)

        conf["PRETRAINED_MODEL_PATH"] = "{}/{}/model_mean.state".format(experiment_dir, exp_folder_name_mean)
        conf["TRAINING_TYPE"] = "diag"
        conf["EXPERIMENT_FOLDER"] = exp_folder_name_diag
        conf["MODEL_NAME"] = "model_diag"

        run(conf)
        with open("{}/{}/config.yaml".format(experiment_dir, exp_folder_name_diag), "w+") as fout:
            yaml.dump(conf, fout)

if __name__ == "__main__":
    main()
