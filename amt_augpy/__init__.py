"""
Compatibility shim for old package name.
This module redirects imports from amt_augpy to amt_augmentor.
"""
import warnings
warnings.warn(
    "The 'amt_augpy' import is deprecated. Use 'amt_augmentor' instead. "
    "This compatibility layer will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from the new package
from amt_augmentor import *
from amt_augmentor import __version__