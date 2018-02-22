Differentiable Particle Filters
==================================================

Author and Contact
------------------

Rico Jonschkowski (rico.j@gmx.de)


Introduction
------------

This folder contains our source code for differentiable particle filters (DPFs) described in the paper "Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors" (Jonschkowski et al. 2018). This implementation complements the paper to provide sufficient detail for reproducing our results and for reusing the method in other research (extensive explanations and descriptions are omitted here and can be found in the paper [TODO add link here]).

If you are using this implementation in your research, please consider giving credit by citing our paper:

    @article{TODO
    }

Dependencies
------------

Our code builds on the following python3 libraries:

numpy

    sudo apt-get install python3-numpy

matplotlib 

    sudo apt-get install python3-matplotlib

Sonnet --> https://github.com/deepmind/sonnet 

TensorFlow --> https://www.tensorflow.org/install/


Setup
-----

The setup.sh script downloads the data for training and testing an creates additional folders. To perform these steps, simply run (in the main folder):

    chmod +x setup.sh
    ./setup.sh

Usage
-----

If all dependencies are met and, you can execute a simple experiment by running the following command in the main folder:

    python3 -m experiments.simple

which will train and test a differentiable particle filter for global localization in maze 1. 
