# ComfyUI-VDN-H3-Plus — MiniMax-H3 的 VDN-H3

> **Plus 分支：** 这是 `xmarre` 维护的 [Saganaki22/ComfyUI-VDN-H3](https://github.com/Saganaki22/ComfyUI-VDN-H3) Plus 分支。它保留上游项目基础，同时维护可能有意与上游不同的功能、集成和修复。

<img width="1039" height="505" alt="VDN-H3" src="https://github.com/user-attachments/assets/ab4c1691-bff5-46fe-8b3e-635429b0700f" />

**[English](README.md)**

这是 [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) 发布版混合注意力架构在 ComfyUI 原生 MiniMax-H3 模型上的移植。

VDN-H3 在局部帧窗口内保留精确 softmax 注意力，并用双向 Video Delta Attention 线性分支覆盖窗口外的长距离时序上下文。本仓库直接读取官方 VDN stage 目录，不修改 ComfyUI 核心文件。

## 保留的发布架构

默认按照检查点 `model_spec.json` 执行，包括：

- MiniMax-H3 text/video/audio 打包布局；
- frame/chunk 对齐 softmax window；
- `none` / `rows` / `columns` / `both` anchor；
- 线性分支共享 QKNorm/RoPE 前的原始 Q/K/V；
- 检查点指定的 separable short-conv；
- beta、逐帧 KDA alpha 和 delta rule；
- 正向/反向 recurrent scan；
- 可选 text-state 和 alpha bridge；
- branch RMSNorm/output gate/`to_out_linear`；
- 可选 softmax gate；
- window 覆盖整个 clip 时走精确 dense-attention fallback。

Advanced 节点默认 `architecture_mode=checkpoint`。只有显式选择 `override` 后才会应用架构消融字段，这些设置不声称与训练时检查点一致。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xmarre/ComfyUI-VDN-H3-Plus.git ComfyUI-VDN-H3
```

保持官方目录结构下载到 `ComfyUI/models/vdn/`：

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

官方当前 stage：

- `stage-dmd-step-250`：8-step VDN-H3，包含 Turbo/DMD adapter；
- `stage-b-step-2000`：50-step VDN-H3，包含 Stage-B/default adapter。

**模型/检查点权重不是 Apache-2.0。** 下载或使用前请阅读“许可证与来源”。

## 节点

### Apply VDN-H3

`MODEL -> MODEL`

| 输入 | 含义 |
|---|---|
| `vdn_checkpoint` | `models/vdn/` 下的官方 stage |
| `apply_turbo_adapter` | stage 存在 Turbo/DMD adapter 时应用；发布的 8-step DMD stage 需要开启 |
| `strength` | adapter 强度；`1.0` 为发布设置 |
| `lora_mode` | `merge` 或安全 runtime `bypass` |
| `branch_weights` | `auto` / `stream` / `resident`；控制 **VDN linear branch**，不是 LoRA 模式 |
| `retain_buffers` | `auto` / `on` / `off`；控制可复用 scratch，不是模型权重 |
| `attention_backend` | 默认 `grouped`，或可选 `flex` |
| `verbose` | 额外布局/adapter 日志 |

### Apply VDN-H3 Advanced

增加独立 Stage-B/Turbo 强度、可选 fast kernels 和显式架构消融。`architecture_mode=checkpoint` 为默认；只有选择 `override` 后 `window_radius`、`window_chunk`、`anchor_frames`、`text_state`、`linear_branch` 才生效。

## LoRA adapter 模式

`lora_mode`、`branch_weights`、`retain_buffers` 分别解决不同显存问题，可以独立组合。

### `lora_mode=merge`

- 使用正常 `ModelPatcher.add_patches()`；
- backup/restore、load/offload、自定义权重转换和重新量化由 Comfy 管理；
- 仍是质量/数值验证的 reference/eager 路径；
- 量化基座可能产生较大的 eager dequantize -> patch -> requantize 临时显存峰值。

### `lora_mode=bypass` — 安全 runtime 低显存路径

保留 `bypass` 名字用于 workflow 兼容，但**不再使用旧 `BypassForwardHook`**。

当前实现：

- 使用 Comfy 公共 `ModelPatcher.add_weight_wrapper()` / `weight_function` 生命周期；
- 不替换、遍历、拼接或恢复 LoRA 目标的 `module.forward`；
- 不安装 VDN LoRA `PatcherInjection`、`_vdn_live_hooks` 或私有 forward owner；
- base parameter 保持未合并；
- LoRA factor 放在独立的 Comfy-managed additional `ModelPatcher` 中，而不是私有 GPU cache；
- 同一目标的 Stage-B/Turbo 项合并为一个 runtime wrapper；
- 每次只创建当前 layer 的临时 compute weight；
- `B @ A` 按输出行分块计算，再按 merge 风格执行 scale/add；额外 delta 临时 buffer 上限为 8 MiB，而不是另一份完整 weight；
- 保留检查点 factor 存储 dtype，并使用 Comfy 选择的 LoRA compute dtype；
- 每次不同 Apply 配置使用不同 runtime ownership key，避免不同 strength/config 被错认成相同已加载状态；同一次 Apply 的 clone 仍保持等价。

这样恢复低显存 adapter 选项，同时不再引入曾导致 Continuum 在 chunk 2 第一次 transformer evaluation 之前递归崩溃的跨 provider `module.forward` 链。

对 fused/quantized MiniMax-H3，Comfy cast 路径仍是唯一权威。runtime wrapper 可能使对应 INT8 layer 本次调用走反量化 compute fallback，因此必须在真实 workflow 上测量速度/显存。

历史 bypass 测试使用的是旧 activation-level forward-hook 实现，不能代表新的 weight-level runtime 路径。完成匹配 GPU render 前，Stage-DMD 质量对照仍以 `merge` 为 reference。

## Curve / pruned MiniMax-H3

部分 MiniMax-H3 检查点把 dense AdaLN timestep 表示压缩为 `adaln_t_table` 等小型坐标表。支持的 pruned 模型谱系使用如下 affine 近似：

```text
dense(t) ≈ mean + curve(t) @ basis
```

发布版 VDN Turbo adapter 中包含完整宽度 AdaLN LoRA。对于一个 `B @ A` 更新，VDN 现在只做一次 pruning-native 投影：

```text
A_pruned   = A @ basis.T
bias_delta = B @ (A @ mean)
```

两个部分都必须保留；常数 `bias_delta` 不能省略，也不会静默丢弃。

投影使用存储的 adapter 和 pruning-affine tensor 在 float64 中一次性计算。投影后的 A 和常数 bias offset 保存为 float32；B 保留检查点存储 dtype，直到 Comfy 选择本次调用的 compute dtype。这不会在 pruned base 已有的 affine 近似之外再引入新的模型近似，但也不声称与原始未剪枝 dense timestep MLP bitwise 相同。

运行时仍直接使用 pruned 模型的原生小宽度 AdaLN：

- sampling loop 中不重建或执行 dense timestep MLP；
- 不替换 AdaLN `forward`；
- `merge` 把投影后的 low-rank weight 更新和常数 bias 作为普通 Comfy patch；
- `bypass` 把投影后的 A/B 和 float32 bias offset 放进同一个 Comfy-managed additional model，并通过 Comfy weight/bias wrapper 应用。

### pruning affine 的解析

必须使用与当前 `adaln_t_table` 对应的**准确** `adaln_basis` + `adaln_mean`；不同模型的 basis 不能互换。解析顺序：

1. 当前 VDN stage 下的 `adaln_affine.safetensors`；
2. 当前选择的 diffusion checkpoint 及其同目录 sibling `.safetensors`；
3. `models/diffusion_models` 中其它已安装候选。

已安装 checkpoint 候选必须能证明其 curve table 与当前 base 相同；table 不匹配或无法验证的 affine 会 fail closed。

标准 Comfy-Org MiniMax-H3 `*_pruned_*` 单文件检查点包含折叠后的 curve table，但**不包含** `adaln_basis` 或 `adaln_mean`。因此不要把 `minimax_h3_fl2va_pruned_bf16.safetensors` 或 `minimax_h3_ref2va_pruned_bf16.safetensors` 传给 `extract_h3_adaln_affine.py`；该工具无法从这些文件中提取不存在的 tensor。

与这些 Comfy-Org curve table 对应的约 97 KB sidecar 已由 [multimodalart/MiniMax-H3-Pruned](https://huggingface.co/multimodalart/MiniMax-H3-Pruned) 发布：

- T2VA / FL2VA：`transformer/adaln_affine.safetensors`
- Ref2VA：`transformer_ref/adaln_affine.safetensors`

下载与当前 base 匹配的文件，并放到当前 VDN stage 中，文件名必须为 `adaln_affine.safetensors`。例如 T2VA/FL2VA：

```bash
TMP="$(mktemp -d)"
hf download multimodalart/MiniMax-H3-Pruned \
  transformer/adaln_affine.safetensors \
  --local-dir "$TMP"
cp "$TMP/transformer/adaln_affine.safetensors" \
  <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors
rm -rf "$TMP"
```

Ref2VA 时将命令中的 `transformer/adaln_affine.safetensors` 两处都替换为 `transformer_ref/adaln_affine.safetensors`。

`tools/extract_h3_adaln_affine.py` 只保留给真正含有 `adaln_basis` 和 `adaln_mean` 的匹配源检查点使用，例如有意保留 pruning auxiliary tensor 的 repaired/private artifact：

```bash
python tools/extract_h3_adaln_affine.py \
  <source-containing-adaln_basis-and-adaln_mean.safetensors> \
  <ComfyUI>/models/vdn/<stage>/adaln_affine.safetensors
```

源文件还含 `adaln_t_table` / `time_embedder.table` 时，工具会写入 table identity。若存在已验证的 table mismatch，VDN 仍会 fail closed；不会丢掉 Turbo 的 51 个 AdaLN 更新或猜测 basis。

## VDN branch 权重驻留

### `branch_weights=auto`

吸收 upstream v1.4 的 VRAM-aware 意图，但保持更严格的 ownership。这里使用的是**有效空闲显存**：Comfy 当前报告的 free VRAM 减去传入 base `MODEL` 尚未 resident 的字节数，避免尚未加载的 H3 base 被误当成 VDN 可用空间。

- 普通 BF16 branch 在该有效预算中满足 `1.5 x branch size + 4 GiB` headroom 时使用 `resident`；
- 否则若存在 `model_int8_convrot_comfyui.safetensors`，选择该文件并使用 `stream`；
- 否则 stream 普通 branch。

INT8 branch 在 auto 模式下仍然 **stream**。这里的 `resident` 必须是真正由 Comfy additional `ModelPatcher` 管理的 parameter tree，不会把量化 branch 偷偷放进未追踪的 VDN GPU cache。

### `branch_weights=stream`

- 不常驻完整 branch；
- 每个 block 需要时才从 stage 解析；
- `safe_open` 生命周期限制在单次读取内，不保留 process-global mmap；
- CUDA + retained buffers 时使用 one-block lookahead；全局只有一个有界 worker executor，它不保存模型 tensor cache，每个 VDN state 最多只有一个可取消的 in-flight result；
- prefetch identity 包含 block index、完整 device 和 compute dtype，因此 placement/dtype 改变后不会复用旧 lookahead 结果。

### `branch_weights=resident`

- 将普通 branch tensor 包装进独立 Comfy `ModelPatcher`；
- 作为 additional model 由 Comfy 管理 device/load/offload；
- 不使用私有全局 GPU branch cache。

量化 branch 当前必须使用 `stream`；显式 `resident` 会 fail closed。

最低 residency 组合可使用 `lora_mode=bypass` + `branch_weights=stream`。

## Retained runtime buffers

Upstream v1.4 证明重复 scratch allocation 有明显成本。本分支保留优化目标，但不采用 process-global CUDA scratch bank。

`retain_buffers=on` 可复用：

- linear complement 的 raw video/text Q/K/V copy；
- forward/reverse recurrence bank；
- grouped-window row-index plan；
- grouped-window K/V gather storage；
- stream one-block prefetch 状态。

Ownership：

- retained pool 属于单个 `VDNState` / Apply 结果；
- 只在一次 diffusion-model execution 期间租用；
- nested/concurrent execution 无法取得同一 lease 时自动使用隔离 transient scratch；
- 大型 scratch 类别只保留最近 geometry，小型 index plan 另有上限；
- cancel 会清理该 state 的 retained scratch/prefetch；
- branch weight 和 LoRA factor 不存放在 scratch pool。

`off` 使用 transient allocation。`auto` 把 `selected branch size + 10 GiB` headroom 规则应用到同一个有效空闲显存预算，也就是先为尚未 resident 的 base `MODEL` 保留空间。

CPU parity test 要求 retained/transient scan、window 和完整 linear-branch 路径与 reference 完全一致；真实 CUDA allocator/速度仍需 GPU 验证。

## 组合与生命周期

VDN 混合注意力本身通过 Comfy object patch 替换 `diffusion_model.blocks.*.attn.forward`。如果其它扩展已经占用同一 attention object-patch 目标，VDN 会明确拒绝冲突。

这与 LoRA runtime 模式不同：`lora_mode=bypass` 不改写 LoRA 目标 `module.forward`，而是使用 weight/bias wrapper。Curve AdaLN 的常数项也通过 bias wrapper，而不是 forward patch。

回归测试覆盖重复 `ModelPatcher` clone/load/unload、pseudo-Continuum `preprocess_text_embeds -> token_refiner.fc1 -> transformer` 顺序、外部 forward owner、不同 strength/config reload、常驻 base weight 不变、无 2x/3x 累积、curve affine constant bias、错误 table 拒绝和文件替换 invalidation。

## Attention backend

- `grouped`：便携默认 exact-window 路径；会把 transformer options 继续传给 Comfy optimized-attention API；
- `flex`：支持时使用 PyTorch FlexAttention；失败仅对本次调用回退 grouped；process-level BlockMask cache 为最多 8 项的 LRU，并按完整 device/layout identity 区分；
- full coverage：走 Comfy 普通完整 attention，并关闭不存在的 linear complement。

## Upstream v1.4 对齐

本 PR 开发期间原始 Comfy port 升级到 v1.4.0，加入更快 streaming 和 VRAM-aware retention。本分支吸收其目标，但没有原样复制资源实现。

保留的目标：

- auto branch placement；
- 内存压力下选择 INT8 ConvRot；
- one-block stream lookahead；
- 可复用 scan/window/activation scratch；
- auto scratch-retention policy。

替换的实现：

- process-global safetensors handle -> 有界 open + file-identity invalidation；
- private global GPU branch cache -> Comfy-managed resident model 或 stream；
- global CUDA scan/KV scratch -> 每个 VDN execution 租用的 state-owned scratch；
- 每 state 永久 worker -> 单一有界、无 tensor cache 的 executor + 每 state 最多一个 in-flight future；
- 无界且只按 device type 区分的 Flex BlockMask cache -> 最多 8 项、按完整 device/layout identity 区分的 LRU。

因此 upstream v1.4 benchmark 数字不能直接当成本分支性能数据，必须重新做真实 GPU 测试。

## 验证

CI 两个 lane：

1. **Pinned Comfy + official oracle**
   - ComfyUI `6c53f8c9a06d95f3d847009ceaae55c624169247`
   - OpenVDN `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`
   - 直接导入并比较 OpenVDN 数学路径；
   - 直接实例化并比较发布版 `HybridAttention` orchestration；
   - adapter、ModelSpec/checkpoint、curve-affine、custom/quantized weight、runtime-buffer、placement policy 和 lifecycle 回归。
2. **Current Comfy main smoke**
   - 当前 Comfy `master` 的包导入和节点注册。

当前测试数量以 pinned CI 结果为准；测试包括 production-shaped KJ selected-INT8 + matching-BF16-sibling affine 解析回归。官方 oracle 覆盖 window/anchor、frame statistics、全部支持的 delta rule、双向 scan、alpha bridge、feature/short-conv、完整 `BidirectionalLinearBranch` 和 reduced `HybridAttention`；其它测试覆盖 fused adapter naming、curve affine 投影及 constant bias、wrong-table rejection、file replacement invalidation、retained/transient parity 和 runtime lifecycle。

绿色 CI 不等于真实渲染质量、峰值 VRAM 或 wall-clock 性能验证。

## 兼容要求

- 当前 ComfyUI MiniMax-H3 fused `qkv_proj` 实现；
- runtime `bypass` 的 weight 目标需要 Comfy `weight_function`；projected curve AdaLN bias 还需要 `bias_function`；
- curve/pruned base + full-width released AdaLN adapter 需要与当前 `adaln_t_table` 匹配的 verified `adaln_basis` + `adaln_mean`；
- 官方 VDN v2 ModelSpec/hybrid-transform contract；
- stage/base block 数和所有启用 branch tensor shape 必须匹配；
- malformed/incomplete/unsupported/stale-replaced 资源会提前 fail closed。

## 许可证与来源

**源码：** Apache License 2.0。起源于 Saganaki22 的 ComfyUI-VDN-H3，并移植/改编 OpenVDN 发布的 VDN-H3 架构与算法。见 `LICENSE`、`NOTICE`。

**OpenVDN：** OpenVDN 源码为 Apache-2.0；其 NOTICE 单独说明 VDN-H3 权重是 MiniMax-H3 衍生权重并按 MiniMax-H3 Community License Agreement 分发。

**模型/检查点权重：** 本仓库不会重新授权 MiniMax-H3 或 VDN-H3 权重。下载和使用仍受对应许可证和资格/地域限制。

来源：

- OpenVDN VDN-H3: https://github.com/OpenVDN/vdn-minimax-h3
- 原始 ComfyUI 移植: https://github.com/Saganaki22/ComfyUI-VDN-H3
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- MiniMax-H3: https://huggingface.co/Comfy-Org/MiniMax-H3
- VDN-H3 weights: https://huggingface.co/OpenVDN/vdn-minimax-h3