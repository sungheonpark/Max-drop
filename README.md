# Max-drop
Source codes for 'Analysis on the Dropout Effect in Convolutional Neural Networks', ACCV 2016

## Installation instructions

The code is tested on Windows OS only.

1. Merge the proto file with your own caffe maintainers. Caffe.proto in this repository was forked from the BVLC Caffe in Apr 4, 2016. max_drop_param, spatial_dropout_param, stochastic_dropout_param are newly added parameters. You may need to change layer-specific ID appropriately.

2. Add the source files (.hpp, .cpp, .cu) to the project.

3. Compile and build.

## Usage

1) Max-drop

MaxDrop layer implements feature-wise max-drop and channel-wise max-drop proposed in the paper.

- prob: the probability of dropping the maximum activation to zero. (default: 0.1)

- kernel_h, kernel_w: size of dropping window (height and width). For instance, if kernel_h and kernel_w are set to 3, the activations of the 3x3 region centered at the maximum activation are dropped to 0. (default: 1)

- dim_option: Specify along which dimension the maximum should be found. (default: 1)

  - feature-wise max-drop : Find the maximum activation within each feature map (dim_option: 1)

  - channel-wise max-drop : Find the maximum activation across all feature maps in the same spatial position (dim_option: 2 or other integer)

2) Stochastic Dropout

StochasticDropout layer implements stochastic dropout proposed in the paper.

- type: select the type of probability distribution (default: 0)
 
  - type: 0 : uniform distribution
 
  - type: 1 : normal distribution

- prob_arg1, prob_arg2 : arguments for the probability distribution. For the case of uniform distribution, dropping probability is drawn from U(prob_arg1, prob_arg2). For the case of normal distribution, dropping probability is drawn from N(prob_arg1, prob_arg2).

3) Spatial Dropout

SpatialDropout layer is the implementation of spatial dropout proposed in ''Tompson et al., 'Efficient Object Localization Using Convolutional Network', CVPR 2015''

 - prob: Probability that drop the entire feature map. (default: 0.5)
