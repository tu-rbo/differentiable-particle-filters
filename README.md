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

Our code builds on python3 and the following libraries:

numpy

    sudo apt-get install python3-numpy

matplotlib 

    sudo apt-get install python3-matplotlib

Sonnet --> https://github.com/deepmind/sonnet 

TensorFlow --> https://www.tensorflow.org/install/


Setup
-----

The setup script **downloads the data** for training and testing and **creates additional folders** (for logging etc.). To perform these steps, simply run (in the main folder):

    chmod +x setup.sh
    ./setup.sh

Usage
-----

If all dependencies are met and setup is done, you can train and test a differentiable particle filter for global localization in maze 1 by running the following in the main folder:

    cd experiments; export PYTHONPATH=../; python3 simple.py; cd ..
    
*Alternatively, you can import the project into the PyCharm IDE and and run `experiments/simple.py` from there. (The convoluted command comes from how PyCharm handles relative imports and relative paths.)*

This command will first learn the different models (motion model, observation likelihood estimator, and particle proposer) separately and then jointly end-to-end. The command line output will show the current losses on training and validation data (mean +- standard error), where ">>" indicates a new lowest validation loss. Training will stop if the best validation loss has decreased for a while (e.g. 200 epochs). You should see something like this:

![Screenshot](https://raw.githubusercontent.com/tu-rbo/differentiable-particle-filters/master/screenshot.png)

As next steps, you can play around with the hyperparameters in `utils/exp_utils.py`, go through the differentiable particle filter code in `methods/dpf.py`, and run other experiments, e.g. applying the filter to the KITTI visual odometry task by running:

    cd experiments; export PYTHONPATH=../; python3 cross_validation_kitti.py; cd ..
