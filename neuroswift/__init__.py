from ._version import __version__
from .model import NeuroSwiftConfig, NeuroSwiftLM
from .plasticity import HebbianUpdater
from .layers import LinearSSM, RMSNorm, SparseMoE
from .tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "HebbianUpdater",
    "LinearSSM",
    "NeuroSwiftConfig",
    "NeuroSwiftLM",
    "RMSNorm",
    "SparseMoE",
    "__version__",
]
