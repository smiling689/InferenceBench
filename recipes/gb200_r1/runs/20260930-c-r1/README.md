# 20260930-c-r1 实验记录

本目录逐项保存完成的独立最终评测。运行源码固定为 `67a9cf3`，场景 C、请求哈希和优化预算见 `run_manifest.json`。每种方法使用单节点四张 GB200、完整 61 层 / 256 routed experts 的 DeepSeek-R1、FP8 dummy 权重和 BF16 激活 / KV。

## 已完成的最终评测

| 方法 | 节点 | burst req/s | Poisson req/s | constant req/s | 几何平均 req/s | 完整请求 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| gpt-6.1-sol / max | gb200-1 | 6.546796 | 7.127967 | 5.761992 | **6.454397** | 768/768 |

Astra、Random Search 和 SMAC 的最终结果仍待完成。上表只采用 seed=1337 的 held-out 评测；Agent 自行保存的开发集“final”文件不作为正式成绩。

Sol 的 Codex exec 用满 7200 秒并以预算超时码 124 结束，随后完成上游预览评测。独立最终容器以 0 退出，`performance_passed=true`。时间戳保留节点原值；完成时间为节点 UTC 11:18:15。准备、预览、最终评测和服务重启的额外时间不计入 7200 秒搜索预算。

## Sol 选定的配置

- TP4 / EP1，TRT-LLM MLA、FlashInfer TRT-LLM FP8 GEMM / MoE；普通 NCCL 通信约束保持生效。
- NEXTN：31 draft steps / 32 verification tokens，使用真实 target verification 和 rejection sampling；不模拟接受率。
- KV 上限 92160 tokens，请求上限 40，prefill chunk 8192，调度接收间隔 8。
- decode CUDA Graph batch sizes 为 1/2/4/8/16/32/40；prefill graph 和 radix cache 关闭。
- `sampling-defaults=openai`，服务 seed=42，stream interval=32。请求中的 temperature=0.3 和输出长度保持不变。
- `diagnostics_plugin/r1_runtime.py` 在未启用 top-k/top-p 时跳过对应的无效过滤；启用时仍调用原过滤函数。完整 target 模型前向和拒绝采样保留。附带的 GPU 验证覆盖接受、拒绝、top-k 和 top-p 分支；诊断记录插件未在最终服务中启用。

准确启动命令、插件源码、验证输出及原始评测 JSON 均保存在方法子目录。指标中的 `generation_throughput_tokens_per_s` 是上游评测器定义的按请求 decode 时间聚合值，不能作为整节点总输出 tokens/s 使用；本实验的目标始终是三个 profile 的请求吞吐几何平均。

## 复现与解释范围

将方法目录中的启动文件和插件放回容器的 `/home/agent/task`，使用本配方固定的模型元数据及镜像环境。完整任务目录和运行缓存仍保留在对应节点的 `results/gb200_r1/20260930-c-r1/` 下；本目录不纳入大型编译缓存、镜像或原始 Codex 会话。最终镜像标识和退出状态见 `run_record.json`。

这是 dummy 权重和合成输入下的性能测量，`quality_evaluated=false`。随机权重还会影响路由分布和 MTP 接受率，不能把这里的推测解码收益直接外推到真实 R1 权重。输入、输出 token 数校验通过不等于生成质量或所有数值实现均已验证。

Agent 可以修改服务参数和实现；上游机械搜索只搜索固定的四个可变维度，固定关闭 CUDA Graph 和推测解码。方法间分差因此包含搜索范围的差异。每种方法只有一次搜索、一次独立最终评测，不据此作统计显著性结论。两节点保守准备配置的参考吞吐分别为 0.120227 和 0.111894 req/s，也表明存在节点或运行波动。
