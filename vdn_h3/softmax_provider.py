"""Opt-in attention subcall contracts. Generic dense overrides do not leak here.

v1 providers receive the already-requested Q rows and already-restricted KV rows.
v2 additionally receives the real square query domain aligned 1:1 with those KV
rows plus a mapping back to the original requested rows. Providers may evaluate
extra queries to satisfy square-QKV kernels, but VDN still owns the KV domain,
gates, projections and learned linear complement.

A separate preprocessing hook is limited to shape-preserving Q/K/V transforms and
runs on the full post-RoPE VDN tensors before grouped-window gathering. This lets
transforms whose metadata uses original packed-row coordinates compose without
remapping those coordinates inside each local window.
"""
PROVIDER_API_VERSION = 2
KEY = "vdn_softmax_provider_v1"
KEY_V2 = "vdn_softmax_provider_v2"
PREPROCESS_KEY = "vdn_attention_preprocess_v1"


def has_v2(options):
    return callable((options or {}).get(KEY_V2))


def preprocess(options, q, k, v, heads):
    provider = (options or {}).get(PREPROCESS_KEY)
    if provider is None:
        return q, k, v
    shapes = (q.shape, k.shape, v.shape)
    dtypes = (q.dtype, k.dtype, v.dtype)
    devices = (q.device, k.device, v.device)
    q, k, v = provider(q, k, v, heads=heads, transformer_options=options)
    if (q.shape, k.shape, v.shape) != shapes:
        raise RuntimeError("VDN attention preprocessing changed QKV topology")
    if (q.dtype, k.dtype, v.dtype) != dtypes or (q.device, k.device, v.device) != devices:
        raise RuntimeError("VDN attention preprocessing changed QKV dtype/device")
    return q, k, v


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
