"""Opt-in softmax subcall contracts. Generic model-level dense overrides do not leak here.

v1 providers receive the already-requested Q rows and already-restricted KV rows.
v2 additionally receives the real square query domain aligned 1:1 with those KV
rows plus a mapping back to the original requested rows. Providers may evaluate
extra queries to satisfy square-QKV kernels, but VDN still owns the KV domain,
gates, projections and learned linear complement.
"""
KEY = "vdn_softmax_provider_v1"
KEY_V2 = "vdn_softmax_provider_v2"


def has_v2(options):
    return callable((options or {}).get(KEY_V2))


def dispatch(options, native, q, k, v, *, kind, scale, square_aligned=False,
             square_q=None, query_positions=None, sink_rows=0):
    options = options or {}
    provider = options.get(KEY_V2)
    if provider is not None:
        result = provider(
            native, q, k, v, kind=kind, scale=scale,
            square_aligned=square_aligned, square_q=square_q,
            query_positions=query_positions, sink_rows=sink_rows)
    else:
        provider = options.get(KEY)
        if provider is None:
            return native()
        result = provider(native, q, k, v, kind=kind, scale=scale,
                          square_aligned=square_aligned)
    if result.shape != q.shape or result.dtype != q.dtype or result.device != q.device:
        raise RuntimeError("VDN softmax provider returned incompatible shape/dtype/device")
    return result
