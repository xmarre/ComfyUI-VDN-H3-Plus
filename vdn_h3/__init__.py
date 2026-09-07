"""VDN-H3 runtime package initialization.

Normal ComfyUI imports already have the Comfy checkout on ``sys.path``. Standalone
training tools can import :mod:`vdn_h3` earlier, before their own bootstrap helper has
run. When those tools provide ``--comfy-root`` (or ``COMFYUI_ROOT``), make that checkout
importable before loading the Comfy-dependent runtime-introspection bridge.
"""
from __future__ import annotations

import os
import sys


def _standalone_comfy_root() -> str | None:
    root = os.environ.get("COMFYUI_ROOT")
    argv = sys.argv[1:]
    for index, arg in enumerate(argv):
        if arg == "--comfy-root" and index + 1 < len(argv):
            root = argv[index + 1]
            break
        if arg.startswith("--comfy-root="):
            root = arg.split("=", 1)[1]
            break
    if not root:
        return None
    root = os.path.abspath(os.path.expanduser(root))
    return root if os.path.isfile(os.path.join(root, "comfy", "patcher_extension.py")) else None


def _ensure_comfy_importable() -> None:
    try:
        import comfy  # noqa: F401
        return
    except ModuleNotFoundError as exc:
        if exc.name != "comfy":
            raise

    root = _standalone_comfy_root()
    if root is not None and root not in sys.path:
        sys.path.insert(0, root)


_ensure_comfy_importable()
del _ensure_comfy_importable, _standalone_comfy_root

from .runtime_introspection import (  # noqa: E402
    install_runtime_introspection_bridge as _install_runtime_introspection_bridge,
)

_install_runtime_introspection_bridge()
del _install_runtime_introspection_bridge
