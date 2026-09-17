"""Opt-in attention subcall contracts. Generic dense overrides do not leak here.

v1 providers receive the already-requested Q rows and already-restricted KV rows.
v2 additionally receives the real square query domain aligned 1:1 with those KV
rows plus a mapping back to the original requested rows. Providers may evaluate
extra queries to satisfy square-QKV kernels, but VDN still owns the KV domain,
gates, projections and learned linear complement.

v3 is the direct rectangular contract: providers receive only the requested Q rows,
the already-restricted K/V rows, and the leading global/prefix K/V row count. This
lets rectangular kernels avoid constructing the v2 square-Q compatibility payload.

v4 keeps v3's direct rectangular tensors and additionally transports VDN's exact
query-row positions inside the already-gathered restricted K/V domain.  The map is
immutable CPU metadata owned by VDN; providers must fail a malformed/unsupported
v4 call back to the supplied native restricted-domain callback rather than silently
re-entering an older sparse provider.

A separate preprocessing hook is limited to shape-preserving Q/K/V transforms and
runs on the full post-RoPE VDN tensors before grouped-window gathering. This lets
transforms whose metadata uses original packed-row coordinates compose without
remapping those coordinates inside each local window.
"""
PROVIDER_API_VERSION = 4
KEY = "vdn_softmax_provider_v1"
KEY_V2 = "vdn_softmax_provider_v2"
KEY_V3 = "vdn_softmax_provider_v3"
KEY_V4 = "vdn_softmax_provider_v4"
PREPROCESS_KEY = "vdn_attention_preprocess_v1"


def has_v2(options):
    return callable((options or {}).get(KEY_V2))


def has_v3(options):
    return callable((options or {}).get(KEY_V3))


def has_v4(options):
    """Return whether v4 is present, including malformed presence.

    Presence suppresses the v2 square-Q compatibility allocation.  dispatch()
    handles a non-callable v4 by running native for that call, as required by the
    fail-closed v4 compatibility contract.
    """
    return KEY_V4 in (options or {})


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
             square_q=None, query_positions=None, sink_rows=0,
             query_position_map=None):
    options = options or {}
    if KEY_V4 in options:
        provider = options.get(KEY_V4)
        if not callable(provider):
            result = native()
        else:
            result = provider(
                native, q, k, v, kind=kind, scale=scale,
                square_aligned=square_aligned, sink_rows=sink_rows,
                query_position_map=query_position_map)
    else:
        provider = options.get(KEY_V3)
        if provider is not None:
            result = provider(
                native, q, k, v, kind=kind, scale=scale,
                square_aligned=square_aligned, sink_rows=sink_rows)
        else:
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
