"""ComfyUI-VDN: Video Delta Net (VDN-H3) hybrid attention for MiniMax-H3.

Reference implementation: github.com/OpenVDN/vdn-minimax-h3 (Apache-2.0).
This package ports the released Video Delta Attention onto ComfyUI's native
MiniMax-H3 model as model patches; no ComfyUI core files are modified.
"""

import os
import sys

_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

# Diagnostic-only overlays install before node construction so ApplyVDN captures
# their forwarding factories. E installs after W; M multiplexes only its own
# separately versioned request mode and replaces only E's local-window dispatcher.
from vdn_h3.first_high_operator_diagnostic import install as _install_first_high_operator_diagnostic
from vdn_h3.first_high_operator_sol_bridge import install as _install_first_high_operator_sol_bridge
from vdn_h3.first_high_sol_local_diagnostic import install as _install_first_high_sol_local_diagnostic
from vdn_h3 import first_high_sol_local_diagnostic as _first_high_sol_local_diagnostic
from vdn_h3.first_high_sol_local_bridge import parse_sol_request as _parse_sol_request
from vdn_h3 import first_high_mapped_neighbor_diagnostic as _first_high_mapped_neighbor_diagnostic

_install_first_high_operator_diagnostic()
_install_first_high_operator_sol_bridge()
_install_first_high_sol_local_diagnostic()

_ORIGINAL_E_PARSE_REQUEST = _first_high_sol_local_diagnostic.parse_request
_ORIGINAL_M_REQUEST = _first_high_mapped_neighbor_diagnostic._request


def _raw_request_mode(value):
    if not isinstance(value, tuple):
        return None
    modes = [
        item[1]
        for item in value
        if isinstance(item, tuple) and len(item) == 2 and item[0] == "mode"
    ]
    return modes[0] if len(modes) == 1 else None


def _parse_e_or_m_request(options):
    options = options or {}
    raw = options.get(_first_high_sol_local_diagnostic.REQUEST_KEY)
    if _raw_request_mode(raw) != "mapped_neighbor_m":
        return _ORIGINAL_E_PARSE_REQUEST(options)
    request = _parse_sol_request(dict(options))
    if not isinstance(request, dict) or request.get("mode") != "mapped_neighbor_m":
        raise RuntimeError("mapped-neighbor M VDN request differs from the Sol companion")
    return request


def _mapped_neighbor_request_only(options):
    options = options or {}
    raw = options.get(_first_high_sol_local_diagnostic.REQUEST_KEY)
    if _raw_request_mode(raw) != "mapped_neighbor_m":
        return None
    return _ORIGINAL_M_REQUEST(options)


_first_high_sol_local_diagnostic.parse_request = _parse_e_or_m_request
_first_high_mapped_neighbor_diagnostic._request = _mapped_neighbor_request_only
_first_high_mapped_neighbor_diagnostic.install()

del _install_first_high_operator_diagnostic
del _install_first_high_operator_sol_bridge
del _install_first_high_sol_local_diagnostic

from vdn_h3.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from vdn_h3.audio_node import ApplyVDNH3AdvancedAudioSafe

# Preserve the established node id/workflow schema while extending the Advanced node
# with generated-audio-only adapter scoping. Existing workflows default to strength 1.0
# and therefore keep the original released execution path exactly.
NODE_CLASS_MAPPINGS["ApplyVDNH3Advanced"] = ApplyVDNH3AdvancedAudioSafe

# Frontend compatibility shim for legacy ApplyVDNH3Advanced positional workflows.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
