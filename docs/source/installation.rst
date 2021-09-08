Installation
============

Requirements
------------

*jDAS* depends on the following Python libraries:

- `TensorFlow <https://www.tensorflow.org/>`_ (``>= 2.2.0``): while training and inference is much faster on a GPU, the CPU version of TensorFlow is sufficient in case problems arise installing the CUDA dependencies.
- `NumPy <https://numpy.org/>`_ and `SciPy <https://scipy.org/>`_ for numerical manipulations.
- (Optional) `Matplotlib <https://matplotlib.org/>`_ for visualisation.
- (Optional) `h5py <https://www.h5py.org/>`_ for IO.
- (Optional) `Jupyter <https://jupyter.org/>`_ notebook or lab to run the examples

The optional dependencies are required to run the examples. All of these can be installed with `Anaconda <https://www.anaconda.com/products/individual>`_::

    conda install -c conda-forge "tensorflow-gpu>=2.2.0" numpy scipy matplotlib h5py notebook

Or through `PyPI <https://pypi.org/>`_::

    pip install "tensorflow-gpu>=2.2.0" numpy scipy matplotlib h5py notebook


Setting-up *jDAS*
-----------------

To obtain the *jDAS* source code, you can pull it directly from the GitHub repository::

    git clone https://github.com/martijnende/jDAS.git

No additional building is required. To test the installation, try running one of the examples Jupyter notebooks in the ``examples`` directory.

Please open a ticket under the tab "Issues" on the GitHub repository if you have trouble setting-up *jDAS*.