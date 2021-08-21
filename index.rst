.. jDAS documentation master file, created by
   sphinx-quickstart on Sat Aug 21 15:42:30 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

jDAS documentation
==================

*jDAS* is a self-supervised Deep Learning model for denoising of Distributed Acoustic Sensing (DAS) data. The principle that underlies *jDAS* is that spatio-temporally coherent signals can be interpolated, while incoherent noise cannot. Leveraging the framework laid out by Batson & Royer (`2019; ICML <http://arxiv.org/abs/1901.11365>`_), *jDAS* predicts the recordings made at a target channel using the target's surrounding channels. As such, it is a self-supervised method that does not require "clean" (noise-free) waveforms as labels. Retraining the model on new data is quick and easy, and will produce optimal denoising performance.


Resources
---------

Test

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
