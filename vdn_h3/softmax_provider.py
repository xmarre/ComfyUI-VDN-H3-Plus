"""Opt-in softmax subcall contract. Model-level dense overrides never leak here.

provider(native, q, k, v, *, kind, scale, square_aligned) returns [rows, H, D].
The caller has already restricted KV to the trained window. Only local calls with
identically ordered query/key row domains advertise square_aligned. Providers
must call native for unrepresentable operations, preserving global/anchor/Flex
semantics. Gates, projections and the linear complement remain VDN-owned.
"""
KEY = "vdn_softmax_provider_v1"


def dispatch(options, native, q, k, v, *, kind, scale, square_aligned=False):
    provider = (options or {}).get(KEY)
    if provider is None:
        return native()
    result = provider(native, q, k, v, kind=kind, scale=scale,
                      square_aligned=square_aligned)
    if result.shape != q.shape or result.dtype != q.dtype or result.device != q.device:
        raise RuntimeError("VDN softmax provider returned incompatible shape/dtype/device")
    return result
