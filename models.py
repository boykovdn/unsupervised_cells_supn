from dataset import FullFrames, CellCrops
from transforms import (rescale_to, 
                        random_channel_select,
                        add_gaussian_noise,
                        joint_random_cell_crop,
                        SigmoidScaleShift)
from utils import (parse_config_dict,
                   make_dirs_if_absent,
                   mahalanobis_dist,
                   load_model)
from losses import (AnnealedDiagonalElboLoss, 
                    AnnealedElboLoss,
                    FixedStdNllLoss)
from model_blocks import (IPE_autoencoder_mu_l,
        DiagChannelActivation)
from transforms import random_gaussian_noise

import torch
from collections import OrderedDict
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as tv_transforms

import numpy as np

torch.autograd.set_detect_anomaly(True)

class TruncatedList:
    r"""
    A list that has a max length and will discard the oldest element when a new
    one is added and the max length reached. Used for keeping track of the
    latest loss values.
    """
    def __init__(self, max_len):
        super().__init__()

        self.max_len = max_len
        self.list = []

    def append(self, x):
        r"""
        Adds element to the end of the list and pops the first element (oldest
        added) if the maximum length is reached.
        """
        if len(self.list) >= self.max_len:
            self.list.pop(0) 

        self.list.append(x)

    def max_len_reached(self):
        return len(self.list) == self.max_len

    def mean(self):
        r"""
        Return the mean of the elements in the stored list, going through a np
        array in the process.
        """
        # Call .item() to return a python float instead of numpy.
        return self.numpy().mean().item()

    def numpy(self):
        r"""
        Returns a numpy array copy of the list.
        """
        return np.array(self.list)

    def torch(self):
        r"""
        Returns a torch tensor copy of the list (1D tensor).
        """
        return torch.Tensor(self.list)

TRAINING_STOP_REASONS = {
    0 : "Max iterations exceeded",
    1 : "Loss values condition is met",
    2 : "Local minimum"
}

class LoggingFunctionWrapper:

    def __init__(self, summary_writer, stopping_threshold, sliding_window_size,
            logging_dset, local_minimum_detection_threshold=1e-8, logger=None):

        super().__init__()

        self.summary_writer = summary_writer

        self.dset = logging_dset
        self.logger = logger
        self.stopping_threshold = stopping_threshold
        self.sliding_window_size = sliding_window_size
        self.losses_list =  TruncatedList(sliding_window_size)
        self.current_criterion_estimate = None
        self.variance_threshold = local_minimum_detection_threshold

    def check_for_local_minimum(self, model, n=50):
        r"""
        Assumption is that if the model is stuck in a local minimum during
        the mean training stage, the samples from p(x|z)p(z) (using ancestral
        sampling and z ~ N(0,I)) will all look the same more or less. So in
        order to check for a local minimum, we draw N samples and average the
        pixel-wise variance across them as the metric.

        Some example values of the variance for good models are between 0.1 and 
        0.04, and for bad models the variance is very low (practically 0).
        """
        mu_sample, _ = get_random_latent_sample(model, n=5) # (N,C,H,W)

        pixelwise_variance = mu_sample.std(0).square().mean()

        return pixelwise_variance < self.variance_threshold

    def check_loss_criterion(self):
        new_criterion_estimate = self.losses_list.mean()
        abs_diff_ = abs(new_criterion_estimate - self.current_criterion_estimate)
        self.current_criterion_estimate = new_criterion_estimate
        return abs_diff_ < self.stopping_threshold

    def calculate_stopping_criterion(self, model, it):
        r"""
        Return int if any stopping criterion is met, or None if not.
        """
        # Check if stopping criterion is met only once all elements in
        # the losses list have been replaced:
        proceed_with_calculation = it % self.sliding_window_size

        if not proceed_with_calculation:
            # We don't check on every iteration for efficiency
            return None

        elif self.current_criterion_estimate is None:
            # Initialize the stop criterion
            self.current_criterion_estimate = self.losses_list.mean()
            return None

        else:
            local_minimum_criterion_stop_ = self.check_for_local_minimum(model)
            loss_criterion_stop_ = self.check_loss_criterion()

            if loss_criterion_stop_:
                self.logger.info("Loss abs delta low enough.. stop: {}".format(loss_criterion_stop_))
                # Training stop reason 1 is that the loss criterion is met
                return 1 

            elif local_minimum_criterion_stop_:
                self.logger.info("Reached local minimum...")
                # Training stop reason 2 is that the model has likely reached
                # a local minimum that it likely won't escape.
                return 2

            else: return None

    def log(self, model, loss, it, log_image, device=0):
        r"""
        Called from the training loop, calls the other logging functions.
    
        :log_image: bool, whether to log images on this iteration
        :model: torch.nn.Module, the model, applied on some data to produce 
            visualizations at the current state
        :it: int, current iteration
        :loss: float, current loss value
        """
        # Log loss and show a random decoded sample
        logging_function_var_pred(model, loss, it, 
                self.summary_writer, 
                log_random_sample=log_image, 
                device=device)
    
        if model.neighbourhood_size > 3:
            return

        # Log image reconstructions
        # TODO Check if removing dset transforms has broken anything.
        if log_image:
            logging_function_rec(model, 
                    self.dset, it, self.summary_writer, device=device)

    def __call__(self, model, loss, it, log_image, device=0):
        r"""
        Log into summary_writer and return None or a dict with information why
        the training was stopped.
        """
        self.log(model, loss, it, log_image, device=device)
        # The following list won't exceed the sliding window size.
        self.losses_list.append(loss)
        return self.calculate_stopping_criterion(model, it)

def logging_function(decoders, loss, iteration, summary_writer, log_image=False):
    r"""
    To be used for the VAE, will sample from a unit normal and log the
    corresponding decoded images. Will also log the loss. Both are logged to
    the given iteration number.
    """
    summary_writer.add_scalar("Loss", loss, iteration)
    decoder = decoders[0]

    if log_image:
        latent_sample = torch.randn((5,decoder.encoding_len))
        with torch.no_grad():
            image_sample = decoder(latent_sample)
        
        summary_writer.add_images("Sampled", rescale_to(image_sample, to=(0,1)), iteration, dataformats="NCHW")

def train(model, train_loader, optimizer, loss_function, iterations, device=0, 
        scheduler=None, logging_function=None, image_logging_period=100, 
        model_backup_path=None):

    model = model.to(device)
    pbar = tqdm(desc="Training...")
    # None if the training is to continue, and a stop code 
    # if the training is to stop for some reason.
    stop_training_reason = None

    iteration = 0
    while(True):

        if stop_training_reason is not None or iteration > iterations:
            break

        for x in tqdm(train_loader):

            if stop_training_reason is not None or iteration > iterations:
                break

            x = x.to(device).float()
            model_output_ = model(x)
            loss = loss_function((x, *model_output_))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if logging_function is not None:

                with torch.no_grad():
                    log_image = iteration % image_logging_period == 0
                    stop_training_reason = logging_function(model, 
                            loss.item(), 
                            iteration, 
                            log_image, 
                            device=device)

            iteration += 1
            pbar.update()

        if scheduler is not None:
            scheduler.step()

        if model_backup_path is not None:
            torch.save(model.state_dict(), model_backup_path)

    if stop_training_reason is None:
        # 0 means that training stops due to reaching the max allowed iterations.
        stop_training_reason = 0 

    return model, stop_training_reason

def logging_function_rec(model, dset, iteration, summary_writer, device=0):
    r"""
    Visualize an input/output reconstruction in Tensorboard
    """

    # Handle the case when the dataset returns a tuple instead of single image.
    if isinstance(dset[0], tuple):
        inp_shape = dset[0][0].shape # [C,H,W]
    else:
        inp_shape = dset[0].shape

    # Select images to reconstruct and log.
    inputs = torch.zeros(5, *inp_shape)
    for i_ in range(5):
        n_rand = torch.randint(len(dset), (1,)).item()
        inputs[i_] = dset[n_rand][0]

    # Get the mean, (co)variance matrix representation and the Mahalanobis distances.
    inputs = inputs.to(device)
    with torch.no_grad():
        model_outp_ = model(inputs)
        out_mu = model_outp_[0].detach()
        out_cov = model_outp_[1].detach()
        # Mahanoubis dist internally exponentiates the diagonal, so no need to do beforehand.
        mah_dists = mahalanobis_dist(inputs, out_mu, out_cov)

    # The case when there is a Cholesky decomposition (full Gaussian).
    if out_cov.shape[1] > 1:
        out_diag = out_cov[:,0,...].unsqueeze(1).exp() # diag is constrained to positive
        out_offdiag = out_cov[:,1:,...]

        summary_writer.add_images("Rec: Out log-variance", rescale_to(out_diag, to=(0,1)), iteration, dataformats="NCHW")
        summary_writer.add_images("Rec: Out off-diagonals", rescale_to(out_offdiag.reshape(-1,*inp_shape[1:]).unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")
        summary_writer.add_images("Rec: Mahalanoubis dist b/n inp-out", rescale_to(mah_dists, to=(0,1)), iteration, dataformats="NCHW")

    # The case when the model returns a single variance vector for a diagonal Gaussian.
    else:
        # TODO This case is irrelevant for this repository, should remove.
        assert False, "This place should never be reached" # TODO

        summary_writer.add_images("Rec: Out log-variance", rescale_to(out_cov, to=(0,1)), iteration, dataformats="NCHW")

    summary_writer.add_images("Rec: Input", rescale_to(inputs[:,0].unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")
    summary_writer.add_images("Rec: Out mean", rescale_to(out_mu, to=(0,1)), iteration, dataformats="NCHW")

def get_random_latent_sample(model, n=5, device=0):
    mu_decoder, var_decoder = model.decoders
    latent_sample = torch.randn((n, mu_decoder.encoding_len)).to(device)
    with torch.no_grad():
        mu_sample = mu_decoder(latent_sample).detach()
        var_sample = var_decoder(latent_sample).detach()

    return mu_sample, var_sample

def logging_function_var_pred(model, loss, iteration, summary_writer, log_random_sample=False, device=0):
    """
    To be used for the VAE, will sample from a unit normal and log the
    corresponding decoded images. Will also log the loss. Both are logged to
    the given iteration number. Will also log the predicted pixel-wise variance.
    """
    summary_writer.add_scalar("Loss", loss, iteration)

    with torch.no_grad():
        if log_random_sample:
            image_sample, var_sample = get_random_latent_sample(model, n=5)

            # Logs are predicted along the diagonal
            var_sample_diag = var_sample[:,0,...].unsqueeze(1).exp()

            summary_writer.add_images("Random sample mean", rescale_to(image_sample, to=(0,1)), iteration, dataformats="NCHW")
            summary_writer.add_images("Random sample variance", rescale_to(var_sample_diag, to=(0,1)), iteration, dataformats="NCHW")

            if var_sample.shape[1] > 1:
                var_sample_offdiag = var_sample[:,1:,...]
                B,C,H,W = var_sample_offdiag.shape
                summary_writer.add_images("Rec: Out off-diagonals", rescale_to(var_sample_offdiag.reshape(-1,H,W).unsqueeze(1), to=(0,1)), iteration, dataformats="NCHW")

def run(conf_, logger=None):
    EXPERIMENT_DIR = conf_['EXPERIMENT_DIR']
    EXPERIMENT_FOLDER = conf_['EXPERIMENT_FOLDER']
    MODEL_NAME = conf_["MODEL_NAME"]
    RAW_PATH = conf_['RAW_PATH']

    DEPTH = conf_["DEPTH"]
    ENCODER_KERNEL_SIZE = conf_['ENCODER_KERNEL_SIZE']
    BATCH_SIZE = conf_['BATCH_SIZE']
    ITERATIONS = conf_["ITERATIONS"]
    ENCODING_DIMENSION = conf_["ENCODING_DIMENSION"]
    MODEL_CONNECTIVITY = conf_['MODEL_CONNECTIVITY']
    MODEL_DIM_H = conf_['MODEL_DIM_H']
    LEARNING_RATE = conf_['LEARNING_RATE']

    # These two scheduler params only used for training the encoder and mean.
    SCHEDULER_GAMMA = conf_['SCHEDULER_GAMMA']
    SCHEDULER_MILESTONES = conf_['SCHEDULER_MILESTONES']

    DEVICE = conf_['DEVICE']

    # L1 regularization of the covariance terms in the Cholesky matrix.
    L1_REG_WEIGHT = conf_['L1_REG_WEIGHT']

    # Variance activation (Sigmoid) parameters of the log-diagonal.
    SIGMOID_SCALE = conf_['SIGMOID_SCALE']
    SIGMOID_SHIFT = conf_['SIGMOID_SHIFT']

    RAW_TRANSFORMS = conf_['RAW_TRANSFORMS']

    PRETRAINED_MODEL_PATH = conf_['PRETRAINED_MODEL_PATH']
    TRAINING_TYPE = conf_['TRAINING_TYPE']
    FIXED_VAR = conf_['FIXED_VAR']

    STOPPING_THRESHOLD = conf_['STOPPING_THRESHOLD']
    SLIDING_WINDOW_SIZE = conf_['SLIDING_WINDOW_SIZE']
    LOCAL_MINIMUM_DETECTION_THRESHOLD = conf_['LOCAL_MINIMUM_DETECTION_THRESHOLD']

    DEBUG = False # Loads a small subset of the training data.

    # Create dir structure
    make_dirs_if_absent(["{}/{}".format(EXPERIMENT_DIR, EXPERIMENT_FOLDER)])

    ## Define transformation functions ##
    raw_transform_functions = {
            'rescale_to_-1_1' : lambda x : rescale_to(x, to=(-1,1)),
            'random_horizontal_flip' : tv_transforms.RandomHorizontalFlip(p=0.5),
            'random_vertical_flip' : tv_transforms.RandomVerticalFlip(p=0.5),
            'random_level_gaussian_blur' : tv_transforms.GaussianBlur(3,sigma=(0.1,2.0)),
            }

    ### Create Composite transform objects from list in config file.
    if RAW_TRANSFORMS is not None:
        raw_tform_list = []
        for tform_name_ in RAW_TRANSFORMS:
            raw_tform_list.append(raw_transform_functions[tform_name_])
        tforms = tv_transforms.Compose(raw_tform_list)
    else: tforms = None

    dset = CellCrops(RAW_PATH, transforms=tforms, load_to_gpu=DEVICE, debug=DEBUG)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=BATCH_SIZE,
        shuffle=True, drop_last=True)

    # Tensorboard logging
    summary_writer = SummaryWriter(log_dir="{}/{}".format(EXPERIMENT_DIR, EXPERIMENT_FOLDER))

    # Logging object adds losses to tensorboard and calculates when to stop.
    logging_object = LoggingFunctionWrapper(
            summary_writer, 
            STOPPING_THRESHOLD, 
            SLIDING_WINDOW_SIZE,
            dset,
            local_minimum_detection_threshold=LOCAL_MINIMUM_DETECTION_THRESHOLD,
            logger=logger)

    # Below, dset[0] has the batch dimension added by us to initialize the model correctly
    if PRETRAINED_MODEL_PATH is not None:
        if isinstance(DEVICE, int):
            map_loc = "cuda:{}".format(DEVICE)
        else:
            map_loc = DEVICE

        # Below we pass 'config_path' as 'None' because we will not load the
        # config from this path, but will rather pass it already loaded as the
        # final kwarg.
        model = load_model("None", map_location=map_loc, dict_passed=conf_)
        print("Loaded pre-trained model {}".format(PRETRAINED_MODEL_PATH))

    else:
        inp_size = dset._get_image_shape()
        spatial_ = inp_size[1:]
        inp_chs = inp_size[0]

        # Initialize model based in input shape (channels and H, W)
        model = IPE_autoencoder_mu_l(
                (BATCH_SIZE,inp_chs,*spatial_), 
                ENCODING_DIMENSION, 
                connectivity=MODEL_CONNECTIVITY, 
                depth=DEPTH, 
                dim_h=MODEL_DIM_H,
                final_var_activation=(lambda : 
                    DiagChannelActivation(
                        activation_maker=(
                            lambda : SigmoidScaleShift(
                                scale=SIGMOID_SCALE, 
                                shift=SIGMOID_SHIFT)),
                        diag_channel_idx=0)),
                encoder_kernel_size=ENCODER_KERNEL_SIZE
                )

        # Initialize the connectivity=2 model using pretrained architecture with conn..=1
        #from collections import OrderedDict
        #state_dict = torch.load("/u/homes/biv20/repos/unsupervised_cells_supn/experiments/mean_training/model_mean.state")
        #encoder_dict = OrderedDict()
        #mean_decoder_dict = OrderedDict()
        #for key, val in state_dict.items():
        #    if "encoder" in key:
        #        encoder_dict[key.replace("encoder.", "")] = val
        #    elif "mu_decoder" in key:
        #        mean_decoder_dict[key.replace("mu_decoder.", "")] = val
        #import pdb; pdb.set_trace()
        #model.encoder.load_state_dict(encoder_dict)
        #model.mu_decoder.load_state_dict(mean_decoder_dict)
           
    class LoggingScalarListener(object):
        r"""
        Subscribes to a loss function which returns a single number as output,
        but internally this number is the sum of multiple functions. For
        example, we might want to log KL, NLL, L2 separately, even though the
        loss output is KL + NLL + L2.
        """
        def __init__(self):
            self.it = 0
        def __call__(self, losses, from_dict=None):
            if from_dict is not None:
                assert isinstance(from_dict, dict)
                summary_writer.add_scalars('loss/moving_averages', 
                        from_dict, 
                        self.it)
            else:
                summary_writer.add_scalars('loss/moving_averages', {
                    'criterion0' : losses[0],
                    'criterion1' : losses[1],
                    'KL' : losses[2]},
                    self.it)
            self.it += 1

    if TRAINING_TYPE == 'diag':
        # Train Var decoder from the beginning, MIDL22.
        model.freeze([model.mu_decoder, model.encoder])
        model.unfreeze([model.var_decoder])
        loss = AnnealedDiagonalElboLoss(
                loss_logging_listener=LoggingScalarListener())

        optimizer = torch.optim.Adam(
                model.var_decoder.parameters(),
                lr=LEARNING_RATE)

        scheduler = None

    elif TRAINING_TYPE == 'supn':
        # Train Var decoder from the beginning, MIDL22.
        model.freeze([model.mu_decoder, model.encoder])
        model.unfreeze([model.var_decoder])

        loss = AnnealedElboLoss(
                loss_logging_listener=LoggingScalarListener(),
                l1_reg_weight=L1_REG_WEIGHT,
                connectivity=MODEL_CONNECTIVITY)

        optimizer = torch.optim.Adam(
                model.var_decoder.parameters(),
                lr=LEARNING_RATE)

        scheduler = None

    elif TRAINING_TYPE == 'mean':
        # For MIDL experiments, 07122021.
        model.freeze([model.var_decoder])
        model.unfreeze([model.mu_decoder, model.encoder])

        loss = FixedStdNllLoss(
                fixed_var=FIXED_VAR,
                loss_logging_listener=LoggingScalarListener()
                )

        optimizer = torch.optim.Adam(
            [{'params' : model.encoder.parameters()},
             {'params' : model.mu_decoder.parameters()}], 
            lr=LEARNING_RATE)

        scheduler = MultiStepLR(
                optimizer, 
                milestones=SCHEDULER_MILESTONES, 
                gamma=SCHEDULER_GAMMA)

    else:
        raise Exception("TRAINING TYPE {} not recognised, sorry! Please pass 'supn', 'mean', or 'diag'.".format(TRAINING_TYPE))

    model, training_outcome = train(
            model, train_loader, optimizer, loss, ITERATIONS, 
            device=DEVICE, logging_function=logging_object, 
            scheduler=scheduler,
            model_backup_path="{}/{}/{}_backup.state".format(EXPERIMENT_DIR, 
                EXPERIMENT_FOLDER, MODEL_NAME))

    torch.save(model.state_dict(), "{}/{}/{}.state".format(EXPERIMENT_DIR, EXPERIMENT_FOLDER, MODEL_NAME))

    return training_outcome
