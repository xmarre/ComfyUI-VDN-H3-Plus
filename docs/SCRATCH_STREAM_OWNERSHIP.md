# Retained scratch stream ownership

VDN's retained pool is leased with a Python lock. CUDA launches are asynchronous,
so releasing that lock does not complete the previous execution's scan, gather or
linear work. Reusing the same mutable storage on another CUDA stream without a
dependency can race earlier reads/writes, even when Python calls are sequential.

Mutable scan banks, delta scratch, raw activation scratch and K/V gather scratch
are now keyed by the resolved CUDA device and allocation stream as well as their
existing geometry/dtype keys. Cached window index plans use the same stream
identity. Repeated work on the same stream reuses storage. A different stream
receives a separate allocation; existing single-entry eviction and bounded plan
cache limits still apply. No device synchronization, per-block events, profiler
or attention-arithmetic change is introduced. Stream changes can require fresh
allocation and plan construction. CPU cache behavior remains unchanged.

Partitioned gather indices already have their own CUDA event and record-stream
handoff and retain that behavior. Prefetched weights also retain their existing
producer/consumer dependency. This patch does not change model weight residency.

CPU tests simulate stream identities to verify cache separation and unchanged
same-stream reuse for all five resources. A CUDA test leaves a scan write pending
on one stream while another execution borrows scratch on a second stream. The
CUDA test requires a GPU; a skipped result is not hardware qualification.

This closes a state-ownership hole. It does not establish the cause of a specific
same-seed generation mismatch. In particular, an execution using Core BSA without
VDN cannot be explained by VDN retained scratch. Trained-model repeatability and
performance still require separate evidence.
