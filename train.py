from utils import (parse_config_dict,
                   make_dirs_if_absent,
                   mahalanoubis_dist)
from models import run

def main():

    description = "Pass config file path to run training with the parameters in that file!"
    config_arg_help = "Path to a config file"

    conf_ = parse_config_dict(description, config_arg_help)
    run(conf_)

if __name__ == "__main__":
    main()
