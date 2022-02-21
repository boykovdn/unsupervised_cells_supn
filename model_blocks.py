import torch
from collections import OrderedDict

class EncoderConvBlock(torch.nn.Module):
    r"""
    Conv2d -> Conv2d -> Pool2D -> ReLU
    """

    def __init__(self, channels_in, channels_out, kernel_size, padding=0):

        super().__init__()

        self.conv0 = torch.nn.Conv2d(
                channels_in, 
                channels_out, 
                kernel_size, 
                padding=padding)

        self.conv1 = torch.nn.Conv2d(
                channels_out, 
                channels_out, 
                kernel_size, 
                padding=padding)

        self.pool0 = torch.nn.MaxPool2d(2)
        self.relu0 = torch.nn.ReLU()

    def forward(self, x):

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.pool0(out)
        out = self.relu0(out)

        return out


class DecoderConvBlock(torch.nn.Module):
    r"""
    Upsample -> Conv2d -> Conv2d -> activation
    """

    def __init__(self, channels_in, channels_out, kernel_size, padding=0, 
            scale_factor=2, mode='nearest', activation_creator=None,
            initialization_function=None):
        r"""
        Decoder conv block.

        Args:
            :channels_in: int, as in torch.nn.Conv2d
            :channels_out: int, as above, param for torch.nn.Conv2d
            :kernel_size: int, param for torch.nn.Conv2d
            :padding: int
            :scale_factor: int, param for torch.nn.Upscale
            :mode: str, param for torch.nn.Upscale
            :activation_creator: callable, returns an instance of the
                final layer's activation function
            :initialization_function: callable, modifies the weight
                and bias of a nn.Conv2d layer. Introduced for the very
                final layer.
        """
        
        super().__init__()

        self.upsample0 = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

        self.conv0 = torch.nn.Conv2d(
                channels_in, 
                channels_out, 
                kernel_size, 
                padding=padding)

        if initialization_function is not None:
            self.conv1 = initialization_function(torch.nn.Conv2d(
                    channels_out, 
                    channels_out, 
                    kernel_size, 
                    padding=padding))
        else:
            self.conv1 = torch.nn.Conv2d(
                    channels_out, 
                    channels_out, 
                    kernel_size, 
                    padding=padding)

        if activation_creator is not None:
            if isinstance(activation_creator, str):
                self.activation = getattr(torch.nn, activation_creator)()
            elif callable(activation_creator):
                self.activation = activation_creator()
            else:
                raise Exception("Unrecognised type {}".format(type(activation_creator)))
        else:
            self.activation = None

    def forward(self, x):

        out = self.upsample0(x)
        out = self.conv0(out)
        out = self.conv1(out)

        if self.activation is not None:
            out = self.activation(out)

        return out

class Encoder(torch.nn.Module):
    r"""
    This is a general Encoder class which contains input/output shape metadata.
    It is needed so that the autoencoders can initialize the internal shapes
    appropriately.
    """

    def __init__(self, input_shape, output_len):
        r"""
        Args:
            :input_shape: [B x C x *], where B is a batch dimension, C is
                the channel dimension, and the rest are spatial dimensions.
            :output_len: int. The Encoder outputs an encoding vector and its
                corresponding variance vector. This parameter dictates the size
                (dimension / length) of the encoding vector. 
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_len = output_len

class EncoderModule(Encoder):
    r"""
    A convolutional network which produces an encoding vector.
    """

    def __init__(self, input_shape, output_len, dim_h, output_clamp=(-100,100), depth=7, kernel_size=3):
        r"""
        Args:
            :input_shape: [B x C x *]
            :output_len: int
            :ch: int, the number of channels in the intermediate layers.
            :output_clamp: list/tuple: do not allow for values above/below these
                thresholds to be predicted so that training does not become unstable.
        """
        super().__init__(input_shape, output_len)

        self.depth = depth
        self.output_clamp = output_clamp
        self._assert_square(input_shape[2:])
        side = input_shape[2] # all spatial dims equal
        self.ks = kernel_size

        self.dim_h = dim_h
        layers, self.final_side = self.get_conv_layer_dict(side)

        self.downsampling_network = torch.nn.Sequential(layers)
        self.dense_mu = torch.nn.Linear(dim_h, output_len) 
        self.dense_logvar = torch.nn.Linear(dim_h, output_len)

        # Init the dense layers to try to aid convergence?
        self.dense_mu.weight.data.fill_(0.0); self.dense_mu.bias.data.fill_(0.0)
        self.dense_logvar.weight.data.fill_(0.0); self.dense_logvar.weight.data.fill_(0.0)       

    def get_conv_layer_dict(self, side):

        layer_dict = OrderedDict()
        ks = self.ks # Kernel size

        layer = 0
        while side >= ks:
            if layer >= self.depth:
                break
            assert side != 0, "Something went wrong, too many mod operations"
            side = side // 2
            # TODO FIx the below abomination
            conv_outp_ch_ = 2**(layer+1)
            
            if layer == 0:
                #layer_dict['conv{:02d}'.format(layer)] = torch.nn.Conv2d(self.input_shape[1], conv_outp_ch_, ks, padding=ks//2)
                layer_dict['EncoderConvBlock{:02d}'.format(layer)] = EncoderConvBlock(
                        self.input_shape[1], 
                        conv_outp_ch_, 
                        ks, 
                        padding=ks//2)
            else:
                #layer_dict['conv{:02d}'.format(layer)] = torch.nn.Conv2d(conv_outp_ch_//2, conv_outp_ch_, ks, padding=ks//2)
                layer_dict['EncoderConvBlock{:02d}'.format(layer)] = EncoderConvBlock(
                        conv_outp_ch_//2, 
                        conv_outp_ch_, 
                        ks, 
                        padding=ks//2)

            #layer_dict['pool{:02d}'.format(layer)] = torch.nn.MaxPool2d(2)
            #layer_dict['down_relu{:02d}'.format(layer)] = torch.nn.ReLU()

            layer += 1

        # Flatten [B,C,side,side] -> [B, C*side*side]
        layer_dict['enc_flatten'] = torch.nn.Flatten(start_dim=1)
        layer_dict['flatten_relu{:02d}'.format(layer)] = torch.nn.ReLU()
        layer_dict['enc_dense_final'] = torch.nn.Linear(conv_outp_ch_ * side**2, self.dim_h)

        return layer_dict, side

    def _assert_square(self, input_spatial_dims):
        r"""
        Only square inputs are supported for simplicity at the moment.
        """
        assert all(map( 
                  lambda x : x == input_spatial_dims[0], input_spatial_dims
               ))

    def forward(self, x):
        r"""
        Args:
            :x: Tensor, [B, C, *]

        Output:
            :mu: [B,C] mean
            :logvar: [B,C] Diagonal of covariance matrix, log of.
        """
        # out [B,C] spatial singleton dimensions are squeezed
        out = self.downsampling_network(x)
        if out.shape[0] == 1:
            # Make sure not to remove the batch dim, even if it is a singleton.
            out = out.squeeze().unsqueeze(0)
        else:
            out = out.squeeze()

        assert len(out.shape) == 2, \
            "Downsampling network output is {}, expected [B,C]".format(out.shape)
        
        mu = torch.clamp(self.dense_mu(out), *self.output_clamp)
        logvar = torch.clamp(self.dense_logvar(out), *self.output_clamp)

        return mu, logvar

class UnsqueezeModule(torch.nn.Module):
    r"""
    Module wrapper for the unsqueeze function.
    """

    def __init__(self, times=1, loc=-1):
        super().__init__()

        self.times = times
        self.loc=loc

    def forward(self, x):
        r"""
        Applies unsqueeze(-1) 'self.times' times.
        
        Args:
            :x: torch.Tensor [B,C]
        """
        out_ = x
        for t in range(self.times):
            out_ = out_.unsqueeze(self.loc)

        return out_


class SquareReshapeModule(torch.nn.Module):
    r"""
    Module wrapper for the reshape function.
    """

    def __init__(self, ch, side):
        super().__init__()

        self.ch = ch
        self.side = side

    def forward(self, x):
        r"""
        Simply reshapes the input to have spatial dimensions equal to ''side'',
        and channels ''ch''.
        
        Args:
            :x: torch.Tensor [B,C]
        """
        assert x.shape[1] % (self.ch * self.side**2) == 0, "Not divisible: {} / {} * {}**2".format(x.shape[1], self.ch, self.side)
        return torch.reshape(x, (x.shape[0], self.ch, self.side, self.side))

class DecoderModule(torch.nn.Module):
    r"""
    """

    def __init__(self, encoding_len, dim_h, output_ch, depth=8, 
            init_side=1, final_activation_creator=None, 
            final_initialization_function=None):
        r"""
        Args:
            :encoding_len: int, the encoding vector length (dimensions).
            :dim_h:
        """
        super().__init__()

        self.init_side = init_side
        self.dim_h = dim_h
        self.encoding_len = encoding_len
        self.upsampling_network = torch.nn.Sequential(
                self._get_conv_layer_dict(depth=depth, oc=output_ch, 
                    final_activation_creator=final_activation_creator,
                    final_initialization_function=final_initialization_function)
                )

    def _get_conv_layer_dict(self, depth=8, oc=1, final_activation_creator=None,
            final_initialization_function=None):
        r"""
        Args:
            :ch: number of channels 
            :first_layer_side: int
        """
        ks = 3

        layers = OrderedDict()

        current_dim = 2**(depth)

        layers['dec_dense_hidden'] = torch.nn.Linear(self.encoding_len, self.dim_h)
        layers["dec_hidden_relu"] = torch.nn.ReLU()
        layers['dec_dense_initial'] = torch.nn.Linear(self.dim_h, (current_dim + oc - 1) * self.init_side**2)
        layers["dec_initial_relu"] = torch.nn.ReLU()

        if self.init_side == 1:
            layers['dec_reshape_layer'] = UnsqueezeModule(times=2)
        else:
            layers['dec_reshape_layer'] = SquareReshapeModule(current_dim + oc - 1, self.init_side)

        for idx in range(depth):
            print(current_dim + oc - 1)

            if idx != depth - 1:
                # By default, ReLU in intermediate layers.
                activation_creator = torch.nn.ReLU
                # No special initialization for intermediate layers.
                initialization_function = None

            else:
                # Final activation is a factory function passed from above. If
                # None, then there is no activation applied in that block.
                activation_creator = final_activation_creator

                # Final layer initialization can be different than the rest,
                # introducing this for experimental LTX loss.
                initialization_function = final_initialization_function

            layers["DecoderConvBlock{:02d}".format(idx)] = DecoderConvBlock(
                    current_dim + oc -1, 
                    current_dim//2 + oc -1, 
                    3, 
                    padding=1,
                    scale_factor=2,
                    mode='nearest',
                    activation_creator=activation_creator,
                    initialization_function=initialization_function)

            current_dim //= 2

        return layers

    def forward(self, x):
        r"""
        Args:
            :x: [B, C], single vector is the input, its length is given by the
                encoding_len parameter.
        
        Outputs:
            torch.Tensor [B, C, *] expanded spatial dimensions, should match
                the VAE input image.
        """
        return self.upsampling_network(x)

class ReparametrizationModule(torch.nn.Module):
    r"""
    Deals with applying the reparametrization trick and providing the Decoder
    with a sampled input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu, stdev):
        r"""
        Reparametrization trick - the random sampling is done by a unit normal,
        the result multiplied by a diagonal covariance matrix and the mean is 
        added. The result is differentiable with respect to the variance and the
        mean.

        Args:
            :mu: [B,C] mean of the Gaussian
            :stdev: [B,C] diagonal entries (square-rooted) of the covariance matrix
                also known as standard deviation.
        """
        return mu + stdev * torch.randn_like(stdev)

class DiagChannelActivation(torch.nn.Module):
    def __init__(self, diag_channel_idx=0, activation_maker=None):
        r"""
        Module which applies an activation function only to one channel, made
        to work on the channel of the diagonal elements in the model that predicts
        the full L matrix.

        Args:
            :diag_channel_idx: int, the diag channel position
            :activation_maker: callable, activation function factory (instead
                of the actual function so that torch registers it as part of 
                this module).
        """
        super().__init__()

        self.activation = activation_maker() if activation_maker is not None else None
        self.diag_channel_idx = diag_channel_idx

    def forward(self, x_logvar):
        r"""
        Args:
            :x_logvar: [B,C,H,W]
        """
        if self.activation is not None:
            x_logvar[:, self.diag_channel_idx] = \
                    self.activation(x_logvar[:, self.diag_channel_idx])

        return x_logvar

class IPE_autoencoder_mu_l(torch.nn.Module):
    r"""
    IPE (Independent Parameter Estimation) VAE from Garoe Dorta's Thesis. It
    contains two decoders which share the same latent space sample, but predict
    the mean and covariance respectively. The covariance prediction follows the
    "Low Rank Sparse Cholesky Decomposition" approach. See text for details.

    This particular implementation outputs the mean pixel values and the W matrix,
    which can be transformed to the sparse cholesky decomposition matrix L.
    """

    def __init__(self, input_shape, encoding_dim, connectivity=1, depth=7, dim_h=None,
            final_mu_activation=None, final_var_activation=None, encoder_kernel_size=3,
            final_var_initialization=None):
        r"""
        Args:
            :input_shape: tuple/list, [B,C,*], assuming the spatial dims are
                equal (square image).
            :encoding_dim: int, dictates how long the encoding vector will be.
            :connectivity: int, how many correlations to predict (1 is the 
                nearest left/down pixels).
        """
        super().__init__()

        assert dim_h is not None

        self.connectivity = connectivity
        self.neighbourhood_size = 2*connectivity + 1
        self.num_nonzero_elems = (self.neighbourhood_size**2) // 2 + 1

        self.encoder = EncoderModule(input_shape, encoding_dim, dim_h, depth=depth, kernel_size=encoder_kernel_size)
        self.reparametrize = ReparametrizationModule()
        self.mu_decoder = DecoderModule(encoding_dim, dim_h, output_ch=1, 
                depth=depth, init_side=self.encoder.final_side, 
                final_activation_creator=final_mu_activation)

        self.var_decoder = DecoderModule(encoding_dim, dim_h, 
                output_ch=self.num_nonzero_elems, depth=depth, 
                init_side=self.encoder.final_side, 
                final_activation_creator=final_var_activation,
                final_initialization_function=final_var_initialization
                )

        self.decoders = (self.mu_decoder, self.var_decoder)

    def freeze_enc_mu(self):
        r"""
        Once called, this function freezes the parameters of the encoder and 
        the mu_decoder, so that the training preceeds only on the variance (or
        precision) decoder.
        """
        # Freeze encoder and mu_decoder
        self.freeze([self.encoder, self.mu_decoder])
        print("Froze encoder and mu_decoder..")

    def freeze(self, modules):
        for module_ in modules:
            for param_ in module_.parameters():
                param_.requires_grad = False

    def unfreeze(self, modules):
        for module_ in modules:
            for param_ in module_.parameters():
                param_.requires_grad = True

    def forward(self, x):
        r"""
        Args:
            :x: torch.Tensor [B,C,*], * represents spatial dims

        Outputs:
            :x_mu: [B,1,*] the mean
            :x_var: [B,N,*], 0th channel is the diagonal of the Cholesky lower
                triangular matrix, the rest represent the coupling between
                neighbouring pixels. The diagonal can be represented as the log
                value without loss of generality, but the off-diagonal terms
                can be both + or -.
            :z_mu: [B,C] the encoding mean
            :z_logvar: [B,C] the encoding log-variance
        """
        z_mu, z_logvar = self.encoder(x)
        z_sampled = self.reparametrize(z_mu, z_logvar.exp())
        x_mu = self.mu_decoder(z_sampled)
        x_var = self.var_decoder(z_sampled)

        return x_mu, x_var, z_mu, z_logvar
