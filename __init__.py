"""ComfyUI-VDN: Video Delta Net (VDN-H3) hybrid attention for MiniMax-H3.

Reference implementation: github.com/OpenVDN/vdn-minimax-h3 (Apache-2.0).
This package ports the released Video Delta Attention onto ComfyUI's native
MiniMax-H3 model as model patches; no ComfyUI core files are modified.
"""

import logging
import os
import sys

_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from vdn_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from vdn_h3 import hybrid as _hybrid

logging.getLogger("comfy.vdn").info(
    "[vdn] sampling admission source policy=%s module_file=%s package_file=%s",
    getattr(_hybrid, "SAMPLING_ADMISSION_POLICY", "unversioned"),
    _hybrid.__file__,
    __file__,
)
del _hybrid

# Frontend compatibility shim for legacy ApplyVDNH3Advanced positional workflows.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
