# jDAS: coherence-based Deep Learning denoising of DAS data

[![Documentation Status](https://readthedocs.org/projects/jdas/badge/?version=latest)](https://jdas.readthedocs.io/en/latest/?badge=latest)

--------------

## Overview

jDAS is a self-supervised Deep Learning model for denoising of Distributed Acoustic Sensing (DAS) data. The principle that underlies jDAS is that spatio-temporally coherent signals can be interpolated, while incoherent noise cannot. Leveraging the framework laid out by Batson & Royer ([2019; ICML](http://arxiv.org/abs/1901.11365)), jDAS predicts the recordings made at a target channel using the target's surrounding channels. As such, it is a self-supervised method that does not require "clean" (noise-free) waveforms as labels. Retraining the model on new data is quick and easy, and will produce optimal separation between coherent signals and incoherent noise.
