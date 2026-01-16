
import os
import sys

ROOT = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from DummyModels.LitePT.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)
