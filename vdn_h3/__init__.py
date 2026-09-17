"""VDN-H3 runtime package initialization."""

from .partitioned_runtime import (
    install_partitioned_external_sequence_bridge as _install_partitioned_external_sequence_bridge,
)
from .runtime_introspection import (
    install_runtime_introspection_bridge as _install_runtime_introspection_bridge,
)

_install_runtime_introspection_bridge()
_install_partitioned_external_sequence_bridge()
del _install_runtime_introspection_bridge
del _install_partitioned_external_sequence_bridge
