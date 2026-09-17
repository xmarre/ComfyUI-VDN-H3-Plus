"""VDN-H3 runtime package initialization."""

from .runtime_introspection import (
    install_runtime_introspection_bridge as _install_runtime_introspection_bridge,
)

_install_runtime_introspection_bridge()
del _install_runtime_introspection_bridge
