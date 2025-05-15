# read version from installed package
from importlib.metadata import version
__version__ = "0.0.1" #version("dialogue2data")

# Import core functions or modules to expose them at the package level
from .read_interview import read_interview