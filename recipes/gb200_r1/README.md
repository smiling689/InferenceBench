# 单节点 GB200 / DeepSeek R1 原始工作流实验

本配方基于上游 `24cdf88f6a4e14ed85d665aa132cecccb3ee95ef`。每个实验独占一个节点的四张 GPU，两个节点并行；正式优化必须在完整场景 C 基线验证通过后启动。

## 固定环境

| 项目 | 配置 |
| --- | --- |
| 节点 | `gb200-1` 和 `gb200-2`；每次实验单节点，不跨节点 |
| GPU | GB200 `0,1,2,3`，每张 189471 MiB，TP=4 / EP=1 |
| 容器 | `lmsysorg/sglang:v0.5.15.post1-cu130`，按 `settings.sh` 中的 digest 固定 |
| 已检查版本 | ARM64；SGLang 0.5.15.post1；PyTorch 2.11.0+cu130；Transformers 5.12.1 |
| 被优化模型 | `deepseek-ai/DeepSeek-R1` 原始完整 61 层 / 256 experts 配置 |
| 权重 | `--load-format dummy --quantization fp8`；BF16 激活和 KV |
| 服务 | `http://127.0.0.1:30080/v1`，标准模型查询和流式 chat completions |
| GPU 容器设备 | 显式暴露 `/dev/nvidia-caps-imex-channels/channel0`，供 NVLink Fabric 内存分配使用 |
| 通信约束 | `NCCL_NVLS_ENABLE=0`、`TORCH_SYMM_MEM_DISABLE_MULTICAST=1`，关闭 FlashInfer allreduce fusion 和 custom allreduce，统一使用普通 NCCL |
| 初始运行参数 | context=16384；chunked prefill=4096；KV pool 上限 65536 tokens；max running=64 |
| 准备阶段开关 | CUDA Graph、radix cache、overlap schedule 关闭；这是启动检查配置，不是最佳配置结论 |

配置与 tokenizer 文件位于 `data/model_metadata/deepseek-ai_DeepSeek-R1/`，不纳入 Git。`model_metadata.json` 固定 Hugging Face revision 和四个文件的 SHA-256。没有下载权重或数据集。`check` 会校验文件哈希、完整模型结构和四张 GPU。

## 执行位置与同步

在 devbox 编辑；远端项目目录使用 `~/ShrinkWrap-v2/InferenceBench`。实际同步必须先检查远端、运行 `rsync --dry-run` 并获得用户明确批准。不使用 `--delete`；保留远端结果。

下面的命令只在已批准同步后的 GB200 节点执行。宿主机脚本只管理 Docker；检查、测试和推理在容器内执行。仓库和模型元数据均以只读方式挂载。

```bash
cd ~/ShrinkWrap-v2/InferenceBench
bash recipes/gb200_r1/docker.sh check
bash recipes/gb200_r1/docker.sh tests
bash recipes/gb200_r1/docker.sh start
bash recipes/gb200_r1/docker.sh smoke
bash recipes/gb200_r1/docker.sh logs
bash recipes/gb200_r1/docker.sh stop
```

`smoke` 最多等待 20 分钟启动，发送一个要求 16 个输出 tokens 的真实 GPU 推理请求，验证 SSE 结束标记和服务端 usage。随机输出不按语义判断正误；这次请求不作为性能成绩。脚本只停止和删除带有本配方标签的指定容器。

## 优化 Agent 的外部 API

外部 API 与本地 DeepSeek R1 服务是两个独立角色。前者用于优化 Agent；模型 ID 必须从提供方 `/models` 列表确认后显式选择。API 探针只验证连接，`pipeline` 才启动正式优化会话。

API 配置位于 devbox 仓库外的 `~/.config/inferencebench/smiling-api.json`，权限为 `0600`，结构为 `base_url` 和 `api_key` 两个字段。不要将该文件加入仓库、镜像、同步目录或日志。

获得同步批准后，可以从 devbox 经 SSH 标准输入向一次性节点容器提供认证，不在节点落盘：

```bash
ssh gb200-1 'cd ~/ShrinkWrap-v2/InferenceBench && bash recipes/gb200_r1/docker.sh api-models' < ~/.config/inferencebench/smiling-api.json
ssh gb200-1 'cd ~/ShrinkWrap-v2/InferenceBench && bash recipes/gb200_r1/docker.sh api-chat --model MODEL_ID' < ~/.config/inferencebench/smiling-api.json
```

探针接受用户配置的 HTTP(S) 地址；HTTPS 保持证书校验。不跟随重定向，也不打印认证头或服务端错误正文。`api-chat` 只检查普通 Chat Completions；它不会声称已验证工具调用或 Codex Responses 兼容性。

## 四组实验

`experiment.sh` 只在宿主机管理 Docker；`experiment.py` 和所有工作负载在容器内执行。运行环境额外安装 Codex CLI 0.156.1、SMAC 2.3.1、ConfigSpace 1.2.1。

| 节点 | 第一轮（7200 秒） | 第二轮（7200 秒） |
| --- | --- | --- |
| gb200-1 | gpt-6.1-sol，max | Random Search / SGLang |
| gb200-2 | gpt-6-astra，max | SMAC / SGLang |

共 8 节点小时 / 32 GPU 小时的优化预算。每轮结束后的服务器重启和最终评测另外计时；准备和故障恢复不计入优化预算。

所有入口统一使用 600 秒的单请求超时。完整 R1 在普通 NCCL、关闭 CUDA Graph 的参考配置下生成较慢，因此机械搜索的 quick 评测进程最多等待 1800 秒，最终完整评测最多等待 10800 秒；quick 的实际可用时间仍受剩余 7200 秒优化预算限制。这些是失败保护上限，不改变请求数、到达模式或吞吐计算，也不表示评测一定会用满该时间。

```bash
bash recipes/gb200_r1/experiment.sh build
bash recipes/gb200_r1/experiment.sh tests
bash recipes/gb200_r1/experiment.sh prepare
# 使用 docker.sh start / smoke 启动并检查初始完整 R1 服务后：
bash recipes/gb200_r1/experiment.sh baseline
bash recipes/gb200_r1/docker.sh stop
# pipeline 的标准输入必须为一行 base_url / api_key JSON，由 devbox 经 SSH 提供。
bash recipes/gb200_r1/experiment.sh pipeline gpt-6.1-sol random
```

第二台的 pipeline 参数为 `gpt-6-astra smac`。结果位于 `results/gb200_r1/20260930-c-r1/`。启动脚本会检查每种负载均已成功完成 256 个请求，否则拒绝启动优化计时。

两个 AI 使用原始 `agents/codex/solve.sh` 的 exec/resume 工作流，只增加模型后缀和 provider 配置支持；起始 launcher 仍为空模板。最终评测在新容器中重新运行其 `start_server.sh`，并保留其安装依赖。源码、评测器和模型配置通过只读挂载提供；API key 仅通过标准输入进入进程环境，不写入镜像配置。

机械搜索沿用上游 Random / SMAC 和 quick/full 协议。`sglang.json` 保留四个可变维度：请求上限、prefill chunk、显存比例、调度策略（90 种组合）；固定 TP4 / EP1、FP8 dummy、GB200 attention/MoE 后端等运行条件。上游 HPO 的 quick 阶段仍为每种负载 4 个请求，因此其搜索反馈不能完整代表高负载表现；最终比较统一采用全部 256 个请求和三种负载。

## 复现边界

- 这是 GB200 + R1 + dummy weights 的性能实验环境，不能直接视作论文 H100 + Mistral + 真实权重的榜单复现。
- 不运行 MMLU-Pro，不把跳过准确率评测写成通过。未来实验结果应明确记录 `quality_evaluated=false`。
- 场景 C 的配置文件保持不变。合成文本替代 LongBench 内容，开发 seed=21，最终 seed=1337，每组 256 条；模型 tokenizer 验证输入长度为 820–1024，输出目标也为 820–1024，temperature=0.3、ignore_eos=true。
- `INFERENCE_BENCH_PERFORMANCE_ONLY=1` 阻止数据集下载和质量评测，显式记录 `quality_evaluated=false`；失败或输出 token 数不足的测量不能作为有效分数。健康请求的吞吐计算和最终几何平均目标沿用上游。
- Poisson 到达分布不变，在性能模式中固定随机种子以便不同方法重放。增加 Transformers 5 BatchEncoding 的 token 计数兼容性；本配方设置 `INFERENCE_BENCH_TOKENIZER_BACKEND=sglang`，与服务端使用相同的 tokenizer 兼容修复，并核对服务端实际 prompt token 数。
- 两个模型已通过容器内真实 Codex CLI shell-tool 调用测试。`gpt-6.1-sol` 在 CLI 0.156.1 中使用 fallback model metadata；接口实际接受 max。
- 首次准备时，两台节点曾遭遇 NVLink Xid 149 / cudaErrorNvlinkUncorrectable；完整权重和 KV 分配已成功。GPU reset、恢复监控服务和 IMEX 重启后，Fabric 经过重新注册恢复为 Completed / Success。应保留诊断记录，并在任何类似故障后重新通过完整基线验证。
- 容器缺少 IMEX channel 时，Fabric `cuMemCreate` 返回 `CUDA_ERROR_NOT_PERMITTED`，SGLang 会禁用 FlashInfer 通信融合。设备映射已通过实际 CUDA 分配验证，所有正式 GPU 容器使用相同映射。
- 映射 IMEX 后，MNNVL fusion 初始化成功，但持续 R1 负载触发 Xid 145 / NVLINK_UNCORRECTABLE；独立 NCCL 测试还发现 NVLS 初始化卡在 `cuMulticastBindMem`。因此本次四组实验统一禁用 NVLS、FlashInfer allreduce fusion 和 custom allreduce；这属于当前机架的运行限制，不改变 exec/resume、Random 或 SMAC 算法。
- logits all-gather 另有默认开启的 PyTorch 对称内存多播路径，独立于上述开关；使用 `TORCH_SYMM_MEM_DISABLE_MULTICAST=1` 使其回退到 NCCL，避免相同的 `cuMulticastBindMem` 卡死。普通 NCCL 已在两台节点分别通过 300 秒同步退出的压力测试（2,645,700 / 2,096,000 次 all-reduce）。
