"""moelan: distribute Mixture-of-Experts LLM inference across machines on a LAN."""

import importlib.util as _ilu
import os as _os

__version__ = "0.1.0"

# With the numba kernels doing the heavy matmuls, OpenBLAS's spinning worker
# threads only compete with numba's (prefill measured 2x slower). Must be set
# before numpy is imported; an explicit setting by the user wins.
if _os.environ.get("MOELAN_KERNELS", "1") != "0" and _ilu.find_spec("numba") is not None:
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
