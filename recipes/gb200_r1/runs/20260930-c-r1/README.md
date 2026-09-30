# 20260930-c-r1 实验记录

本目录逐项保存完成的独立最终评测。运行源码固定为 `67a9cf3`，场景 C、请求哈希和优化预算见 `run_manifest.json`。每种方法使用单节点四张 GB200、完整 61 层 / 256 routed experts 的 DeepSeek-R1、FP8 dummy 权重和 BF16 激活 / KV。

## 已完成的最终评测

| 方法 | 节点 | burst req/s | Poisson req/s | constant req/s | 几何平均 req/s | 完整请求 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gpt-6.1-sol / max | gb200-1 | 6.546796 | 7.127967 | 5.761992 | **6.454397** | 768/768 |
| gpt-6-astra / max | gb200-2 | 5.208178 | 4.627873 | 3.544380 | **4.404222** | 768/768 |

Random Search 和 SMAC 的最终结果仍待完成。上表只采用 seed=1337 的 held-out 评测；Agent 自行保存的开发集“final”文件不作为正式成绩。

Sol 的 Codex exec 用满 7200 秒并以预算超时码 124 结束，随后完成上游预览评测。独立最终容器以 0 退出，`performance_passed=true`。时间戳保留节点原值；完成时间为节点 UTC 11:18:15。准备、预览、最终评测和服务重启的额外时间不计入 7200 秒搜索预算。

Astra 同样用满 7200 秒：首轮 exec 提前结束后，上游 resume 使用剩余 34 秒，最终返回 124。独立最终容器以 0 退出，`performance_passed=true`，节点 UTC 11:31:31 完成。两种 Agent 均沿用原 `agents/codex/solve.sh`，由真实 Codex CLI 调用 Responses API 和 shell 工具。

## Sol 选定的配置

- TP4 / EP1，TRT-LLM MLA、FlashInfer TRT-LLM FP8 GEMM / MoE；普通 NCCL 通信约束保持生效。
- NEXTN：31 draft steps / 32 verification tokens，使用真实 target verification 和 rejection sampling；不模拟接受率。
- KV 上限 92160 tokens，请求上限 40，prefill chunk 8192，调度接收间隔 8。
- decode CUDA Graph batch sizes 为 1/2/4/8/16/32/40；prefill graph 和 radix cache 关闭。
- `sampling-defaults=openai`，服务 seed=42，stream interval=32。请求中的 temperature=0.3 和输出长度保持不变。
- `diagnostics_plugin/r1_runtime.py` 在未启用 top-k/top-p 时跳过对应的无效过滤；启用时仍调用原过滤函数。完整 target 模型前向和拒绝采样保留。附带的 GPU 验证覆盖接受、拒绝、top-k 和 top-p 分支；诊断记录插件未在最终服务中启用。

准确启动命令、插件源码、验证输出及原始评测 JSON 均保存在方法子目录。指标中的 `generation_throughput_tokens_per_s` 是上游评测器定义的按请求 decode 时间聚合值，不能作为整节点总输出 tokens/s 使用；本实验的目标始终是三个 profile 的请求吞吐几何平均。

## Astra 选定的配置

- TP4，TRT-LLM MLA；prefill 使用 FA4，FP8 GEMM / MoE 使用 FlashInfer TRT-LLM。
- EAGLE：11 draft steps / 12 verification tokens，使用真实 target verification 和 rejection sampling；服务 seed=1，保留模型默认采样参数。
- KV 上限 131072 tokens，请求上限 128，prefill chunk 2048，单次 prefill 请求上限 2，调度接收间隔 4，stream interval=8。
- decode CUDA Graph batch sizes 为 1/2/4/8/12/16/24/32/40/48/56/64；breakable prefill graph buckets 为 64/1024/2048。
- `prepare_graph_overlay.py` 在任务目录生成三个 SGLang 模块的覆盖文件，由 `sitecustomize.py` 定向加载。修改允许 CUDA 选择既有 MHA companion metadata；FA4 遇到 prefix cache 命中时退回普通 prefill；无前缀图重放前恢复三个 attention metadata 标记。保留完整模型计算。
- Agent 的图路径检查在两个独立输入上各生成 16 个 greedy tokens，与 eager 路径结果一致，参考 logprobs 有限。验证脚本与输出已归档；这不构成对所有采样、前缀和长度组合的数值正确性证明。

Sol 设置 `sampling-defaults=openai`（默认 top_p=1），Astra 保留模型默认值（top_p=0.95）；两者服务随机种子也不同。请求中的 temperature=0.3 与输出长度固定，以上服务选择作为各 Agent 的优化结果记录。

## 机械搜索记录

Random Search 已完成两小时搜索：9 次尝试中，7 次通过快速评测，覆盖 7 个不同的有效配置；第 2 轮因端口范围错误启动失败，第 8 轮因剩余预算耗尽而未完成快速评测。最佳为第 6 轮：请求上限 256、prefill chunk 4096、静态显存比例 0.9、调度策略 lpm。开发集快速分数为 0.018643 req/s，不能与上表完整最终评测分数直接比较。原始搜索记录、选定配置和快速评测 JSON 位于 `random/`；完整最终评测仍在运行，SMAC 搜索仍待完成。

上游机械工作流每个 profile 只用 4 个请求筛选候选，最后用 256 个 held-out 请求评测选中的配置。快速评测中的请求数低于全部并发上限候选值（16–256），因此不能直接观察这些并发上限在高负载下的差异。本次保留这一原版工作流。

当前 SGLang 默认派生 gRPC 端口为 HTTP 端口加 10000，而原版工作流从操作系统分配随机空闲 HTTP 端口。HTTP 端口超过 55535 时，派生端口校验失败；这些基础设施失败按原流程消耗搜索预算，未补时或人工重选候选。具体失败保留在方法记录中。

## 复现与解释范围

将方法目录中的启动文件和插件放回容器的 `/home/agent/task`，使用本配方固定的模型元数据及镜像环境。完整任务目录和运行缓存仍保留在对应节点的 `results/gb200_r1/20260930-c-r1/` 下；本目录不纳入大型编译缓存、镜像或原始 Codex 会话。最终镜像标识和退出状态见 `run_record.json`。

这是 dummy 权重和合成输入下的性能测量，`quality_evaluated=false`。随机权重还会影响路由分布和 MTP 接受率，不能把这里的推测解码收益直接外推到真实 R1 权重。输入、输出 token 数校验通过不等于生成质量或所有数值实现均已验证。

Agent 可以修改服务参数和实现；上游机械搜索只搜索固定的四个可变维度，固定关闭 CUDA Graph 和推测解码。方法间分差因此包含搜索范围的差异。每种方法只有一次搜索、一次独立最终评测，不据此作统计显著性结论。两节点保守准备配置的参考吞吐分别为 0.120227 和 0.111894 req/s，也表明存在节点或运行波动。
