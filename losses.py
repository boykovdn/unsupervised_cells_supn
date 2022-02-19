import torch
import numpy as np
from structured_uncertainty.jacobi_sampler import (
                            get_log_prob_from_sparse_L_precision,
                            apply_sparse_chol_rhs_matmul)
from torchvision.transforms import Resize

def kl_divergence_unit_normal(mu, logvar):
    r"""
    Kullback-Leibler divergence between the Gaussian with parameters ''mu'' and the
    diagonal covariance log-elements ''logvar'', and a unit normal N(0,I).
    
    Args:
        :mu: [B,C] or [B,C,H,W] for fully convolutional
        :logvar: [B,C] or [B,C,H,W]

    Outputs:
        [B,] or [B,H,W]?
    """
    var = logvar.exp()
    output_ = 0.5 * (torch.einsum("bi...,bi...->b...", mu, mu) +
                     torch.sum(var, dim=1) -
                     torch.sum(logvar, dim=1) -
                     mu.shape[1])

    if len(output_.shape) == 3:
        output_ = output_.sum(1).sum(1)

    return output_

class FixedStdNllLoss(torch.nn.Module):
    r"""
    Only works on the mean decoder (and KL), ignores any other terms.
    """
    def __init__(self, fixed_std=1.0, loss_logging_listener=None):
        r"""
        :fixed_std: float, a fixed standard deviation value. Is the diagonal of
            the covariance matrix (constant).
        :loss_logging_listener: call with list of losses to log them individually
            rather than as a sum.
        """
        super().__init__()
        self.fixed_std = fixed_std
        self.loss_logging_listener = loss_logging_listener

    def step(self):
        pass

    def exec_forward(self, x, x_mu, x_logvar, z_mu, z_logvar):
        r"""
        Only works with C=1 (grayscale) at the moment.

        :x: torch.Tensor, [B,C,H,W], input batch
        :x_mu: torch.Tensor [B,C,H,W], mean prediction
        :x_logvar: torch.Tensor [B,C,H,W], NOT USED
        :z_mu: torch.Tensor [B,D,H,W], mean of encoding, dimension D.
        :z_logvar: torch.Tensor [B,D,H,W], diagonal covariance matrix of encoding.
        """
        r = (x - x_mu).flatten(start_dim=1) # [B, H*W]
        device = r.device

        # Negative Log-Likelihood
        im_size_w = x.shape[-1]
        im_size_h = x.shape[-2]
        n_pixels = im_size_w * im_size_h
        constant_term = n_pixels * torch.log(torch.Tensor([2.0 * np.pi]))
        constant_term = constant_term.to(device)

        # Negative Log-likelihood
        nll_ = 0.5 * (torch.einsum("bi,bi->b", r, 
            ((1/self.fixed_std) * r)) + n_pixels * torch.log(torch.Tensor([self.fixed_std])).to(device)) + 0.5 * constant_term
       
        # Kullback-Leibler Div
        kl_ = kl_divergence_unit_normal(z_mu, z_logvar)

        # Mean across batches [B,] -> float
        nll_ = torch.mean(nll_)
        kl_ = torch.mean(kl_)

        # TODO Sigmoid for sane values.
        # Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:
            loss_dict = {
                    'nll' : nll_.item(),
                    'kl' : kl_.item()
                    }
            self.loss_logging_listener([], from_dict=loss_dict) 

        return nll_ + kl_

    def forward(self, inp_):
        r"""
        See exec_forward. This function just unpacks the input.
        """
        return self.exec_forward(*inp_)

class AnnealedDiagonalElboLoss(torch.nn.Module):
    r"""
    The loss from Garoe's Thesis with annealing of the L2 between input and
    mean, and diagonal covariance matrix rather than full.
    """
    def __init__(self, l2_weight=1, schedule=None, l2_gamma=1, loss_logging_listener=None,
            offdiag_passed=True, eps_=1e-7, kl_weight=1.):
        r"""
        :l2_weight: float, the weight of the L2 term, meant originally to be
            annealed away. Corresponds to the alpha parameter in Garoe's Thesis.
        :schedule: list [a,b,c,...], where a < b < c < ... . Represents the
            iterations at which the annealing takes a step.
        :l2_gamma: float, the rate of annealing of ''l2_weight'' (multiplied by it).
        :loss_logging_listener: call with list of losses to log them individually
            rather than as a sum.
        :offdiag_passed: bool, if working with model that predicts offdiagonal
            elements as well, but we want to only optimize the diagonal. Discards
            the offdiagonal elements before computing the loss.
        :kl_weight: float, optional weighting of the KL term in the loss.
            Initially introducing it to counteract the KL value scaling with
            the latent encoding dimensions.
        """
        super().__init__()

        self.schedule = schedule
        self.schedule_position = 0
        self.l2_weight = l2_weight
        self.l2_gamma = l2_gamma
        self.current_step = 0
        self.loss_logging_listener = loss_logging_listener
        self.offdiag_passed = offdiag_passed
        self.eps_ = eps_

        self.kl_weight = kl_weight

    def step(self):
        self.current_step += 1
        if self.schedule is not None and self.schedule_position < len(self.schedule):
            if self.current_step >= self.schedule[self.schedule_position]:
                self.schedule_position += 1
                self.l2_weight *= self.l2_gamma

    def exec_forward(self, x, x_mu, x_logvar, z_mu, z_logvar):
        r"""
        Only works with C=1 (grayscale) at the moment.

        :x: torch.Tensor, [B,C,H,W], input batch
        :x_mu: torch.Tensor [B,C,H,W], mean prediction
        :x_logvar: torch.Tensor [B,C,H,W], diagonal of covariance matrix.
        :z_mu: torch.Tensor [B,D,H,W], mean of encoding, dimension D.
        :z_logvar: torch.Tensor [B,D,H,W], diagonal covariance matrix of encoding.
        """
        r = (x - x_mu).flatten(start_dim=1) # [B, H*W]
        device = x.device

        if self.offdiag_passed:
            log_sigma = x_logvar[:,0].flatten(start_dim=1) # [B, H*W]
        else:
            log_sigma = x_logvar.flatten(start_dim=1) # [B, H*W]

        sigma = log_sigma.exp()
    
        # Negative Log-Likelihood
        # TODO RM prints below
        #with torch.no_grad():
        #    print("sigma min,mean,max: {},{},{}".format(sigma.min().item(),
        #        sigma.mean().item(), sigma.max().item()))
        im_size_w = x.shape[-1]
        im_size_h = x.shape[-2]
        constant_term = im_size_w * im_size_h * torch.log(torch.Tensor([2.0 * np.pi]))
        constant_term = constant_term.to(device)
        nll_ = 0.5 * (torch.einsum("bi,bi->b", r, 
            ((1/(sigma + self.eps_)) * r)) + 0.5 * log_sigma.sum(1)) + 0.5 * constant_term
       
        # Kullback-Leibler Div
        kl_ = kl_divergence_unit_normal(z_mu, z_logvar)

        # sum square differences
        l2_ = torch.linalg.norm(r, dim=1) # [B, H*W] -> [B,]

        # Mean across batches [B,] -> float
        nll_ = torch.mean(nll_)
        kl_ = torch.mean(kl_)
        l2_ = torch.mean(l2_)

        # TODO Sigmoid for sane values.
        # Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:
            loss_dict = {
                    'nll' : nll_.item(),
                    'l2' : l2_.item() * self.l2_weight,
                    'kl' : kl_.item()
                    }
            self.loss_logging_listener([], from_dict=loss_dict) 

        return nll_ + kl_ * self.kl_weight + self.l2_weight * l2_

    def forward(self, inp_):
        r"""
        See exec_forward. This function just unpacks the input.
        """
        return self.exec_forward(*inp_)

class AnnealedElboLoss(torch.nn.Module):
    r"""
    The loss function from Garoe's Thesis, complete with annealing and L1
    regularization.
    """

    def __init__(self, l2_weight=1, schedule=None, l2_gamma=1, l1_reg_weight=1, connectivity=1,
            loss_logging_listener=None, kl_weight=1., on_l2_annealing_finish=None):
        r"""
        :l2_weight: float, the weight of the L2 term, meant originally to be
            annealed away. Corresponds to the alpha parameter in Garoe's Thesis.
        :schedule: list [a,b,c,...], where a < b < c < ... . Represents the
            iterations at which the annealing takes a step.
        :l2_gamma: float, the rate of annealing of ''l2_weight'' (multiplied by it).
        :l1_reg_weight: float, the weight of the l1 regularization term.
        :connectivity: int, the connectivity (sparsity pattern) of the model. It
            is used when calculating the likelihood term using the efficient
            convolution method.
        :loss_logging_listener: callable, used to log individual components of loss.
        :kl_weight: float, optionally can weigh up or down the KL term.
        :on_l2_annealing_finish: callable, execute when l2 annealing is done.
            For example to freeze the encoder and mu-decoder.
        """
        super().__init__()

        self.connectivity = connectivity
        self.schedule = schedule
        self.schedule_position = 0
        self.l2_weight = l2_weight
        self.kl_weight = kl_weight
        self.l2_gamma = l2_gamma
        self.l1_reg_weight = l1_reg_weight
        self.current_step = 0

        self.loss_logging_listener = loss_logging_listener
        self.on_l2_annealing_finish = on_l2_annealing_finish

    def step(self):
        self.current_step += 1
        if self.schedule is not None and self.schedule_position < len(self.schedule):
            if self.current_step >= self.schedule[self.schedule_position]:
                self.schedule_position += 1
                self.l2_weight *= self.l2_gamma

            if self.on_l2_annealing_finish is not None\
                    and self.schedule_position == len(self.schedule):

                self.on_l2_annealing_finish()

    def exec_forward(self, x, x_mu, x_logvar, z_mu, z_logvar):
        r"""
        Even though x_logvar suggests that the whole Tensor is logvar, actually
        only the first dimension is log-ged (the diagonal elements).

        Args:
            :x: (B, C, W, H) input image
            :x_mu: (B, C, W, H) output reconstruction mean
            :x_logvar: (B, Cf, W, H) output weights, Cf depends on the local 
                connectivity; the 0th channel corresponds to the diagonal 
                elements.
            :z_mu: (B, Cz) encoding mean
            :z_logvar: (B, Cz) encoding variance diagonal
        """
        # Neg log-likelihood
        nll_ = - get_log_prob_from_sparse_L_precision(
                x, x_mu, self.connectivity, 
                x_logvar[:,0,...].unsqueeze(1), 
                x_logvar[:,1:,...]) # (B,)

        # KL divergence in latent space
        kl_ = kl_divergence_unit_normal(z_mu, z_logvar)

        # L2 on output.
        r = (x - x_mu).flatten(start_dim=1) # [B, H*W]
        l2_ = torch.linalg.norm(r, dim=1) # [B, H*W] -> [B,]

        # correlations l1 regularization
        l1_corr_ = x_logvar[:,1:,...].abs().mean()

        ## Mean across batches [B,] -> float
        nll_ = torch.mean(nll_)
        kl_ = torch.mean(kl_)
        l2_ = torch.mean(l2_)

        ## Log individual loss components if listener passed.
        if self.loss_logging_listener is not None:
            loss_dict = {
                    'nll' : nll_.item(),
                    'l2' : l2_.item() * self.l2_weight,
                    'kl' : kl_.item() * self.kl_weight,
                    'l1_corr' : l1_corr_.item() * self.l1_reg_weight
                    }
            self.loss_logging_listener([], from_dict=loss_dict) 

        return nll_ + self.kl_weight * kl_ + self.l2_weight * l2_ + self.l1_reg_weight * l1_corr_

    def forward(self, inp_):
        r"""
        See exec_forward. This function just unpacks the input.
        """
        return self.exec_forward(*inp_)
