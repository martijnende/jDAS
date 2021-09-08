import os
import sys

cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, ".."))

if pardir not in sys.path:
    sys.path.append(pardir)

from jDAS import JDAS


""" Basic denoising """

rng = np.random.default_rng(42)

data = rng.normal(size=(100, 10_000))

jdas = JDAS()
model = jdas.load_model()
clean_data = jdas.denoise(data, postfilter=True, filter_band=(1, 10, 50))
