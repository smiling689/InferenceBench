# 2026-09-30 准备阶段记录

**恢复完成，两台节点均已通过完整场景 C 基线，两个 Agent 已开始正式优化。** gb200-1 的 Sol 于节点时间 09:10:52 UTC 启动；gb200-2 的 Astra 于节点时间 09:19:32 UTC 启动。各自独立计时 7200 秒，随后运行预览、最终评测和机械搜索。这里记录的是准备阶段证据，尚无四种方法的最终比较成绩。

devbox 约 03:18 UTC，最后一次完整 R1 重启期间，两台节点的 SSH 会话同时中断；后续检查确认节点发生过重启。09:21 UTC 的并行时钟检查发现 gb200-1、gb200-2 的系统时钟分别比 devbox 慢约 113 秒、77 秒；本文件中的启动和完成时间保留节点原始记录，不据此比较跨节点耗时，也没有在实验期间修改系统时钟。

## 重启后的恢复检查

- 两台节点 SSH 均已恢复；恢复检查时四张 GPU 均空闲，Fabric 为 Completed / Success / Healthy。完整基线运行期间未发现新的 Xid，之后 GPU 由实验队列独占使用。
- Docker 的实际 `data-root` 为 `/dev/shm/docker`。重启后镜像和容器均已丢失；系统盘上的源码、请求集和已保存的诊断日志仍在。不能恢复此前仅存于容器内部的编译缓存或未导出的日志。
- 两台节点的 Dockerfile、实验入口、宿主机 wrapper 和服务启动脚本均与 devbox 的 SHA-256 一致；两个请求文件的哈希也与下表一致。
- 已用同一基础镜像 digest 完成重建。两台节点重新通过环境、API、18 项测试、完整 R1 smoke 和完整基线后，各自自动启动原定优化队列。
- 两台节点均已将约 25 GB 实验镜像导出到持久磁盘 `/srv/shrinkwrap-data/inferencebench/images/inferencebench-gb200-20260930.tar`，便于未来通过 `docker load` 恢复。未修改系统 Docker 存储设置。
- 运行源码版本为 `67a9cf389bd7568afac740e0e7cb8b6edcddc59b`；123 个同步文件与 devbox 的 SHA-256 全部一致。两个容器环境的 pip freeze 哈希均为 `029315f8f9b1150173ad079be3161dd946a28757acbe367f9eace97e90138b2c`。
- 节点直连外部 API 超时，已通过 devbox 的 SSH 反向转发恢复。两个专用用户 systemd 服务维持隧道并自动重连，节点入口仅绑定 `127.0.0.1:38081`；两个模型均经此路径通过真实 Codex Responses / shell-tool 测试。密钥不写入源码、镜像配置或命令参数。
- 恢复状态和日志位于原结果根目录的 `pipeline_stage.txt`、`recovery-queue.log` 和 `recovery-image-build.log`；先前失败基线目录会保留为 `preflight-before-recovery-*`。

## 已固定的实验

- 单场景 C，完整 DeepSeek-R1，单节点四张 GB200，FP8 dummy 权重。
- 输入、输出目标均为 820–1024 tokens；burst、Poisson、constant 三种流量保持原配置。
- 固定合成文本替代数据集内容，开发 seed=21、最终 seed=1337，各 256 条请求。不下载权重或数据集，不评估生成质量。
- gb200-1：gpt-6.1-sol / max，然后 Random Search；gb200-2：gpt-6-astra / max，然后 SMAC。
- 每种方法 7200 秒优化预算，共 8 节点小时。准备、服务器重启、原始工作流的预览评测和最终评测另计，不能保证总用时为四小时。
- 两个 Agent 使用原始 Codex exec/resume 工作流；机械搜索沿用上游 Random/SMAC 与 quick/full 协议。

## 已验证

| 检查 | 结果 |
| --- | --- |
| 外部模型 API / Codex CLI | 两个指定模型均完成真实 shell tool 调用，reasoning effort=max |
| 评测器测试 | 两台节点容器内均为 18 passed |
| 搜索参数兼容性 | 两台节点已安装的 SGLang CLI 均成功解析全部 90 种配置 |
| 完整 R1 短请求 | 此前两台节点均返回完整 16-token 流式响应；不是吞吐成绩 |
| 普通 NCCL 压力测试 | 关闭 NVLS 后，两台节点分别运行 300.006 / 300.015 秒，完成 2,645,700 / 2,096,000 次 all-reduce，数据校验通过、进程正常退出 |
| logits 通信回退 | 直接调用 SGLang 的 create_state，八个 GPU 进程均确认 multicast_ptr=0，并通过 NCCL all-gather 数据校验 |
| 完整场景 C 基线 | 两台节点的三个 profile 均为 256/256 成功，performance_passed=true |

完整基线使用保守准备配置，未纳入四种优化方法的成绩：

| 节点 | burst req/s | Poisson req/s | constant req/s | 几何平均 req/s | 完成时间（节点 UTC） |
| --- | ---: | ---: | ---: | ---: | --- |
| gb200-1 | 0.168925 | 0.142876 | 0.072003 | 0.120227 | 09:10:26 |
| gb200-2 | 0.158419 | 0.130386 | 0.067823 | 0.111894 | 09:19:07 |

这两次单次参考测量相差约 7.4%，后续跨节点结果应报告这一背景，不将小幅分差直接解释为方法优劣。没有评估 dummy 输出的质量。

两台节点生成的请求哈希相同：

```text
requests_21.jsonl    e3329da09722133ae0d3fd368b3438b2c8f230815ab32ee32cb77773e71f3ba5
requests_1337.jsonl  08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae
```

## 通信问题与配方修复

1. 最初出现 Xid 149 / NETIR_LINK_DOWN，GPU reset 和 IMEX 服务恢复后，Fabric 回到 Completed / Success。
2. 缺少容器 IMEX 设备映射时，Fabric cuMemCreate 返回 CUDA_ERROR_NOT_PERMITTED；显式映射 channel0 后分配成功。
3. 开启 MNNVL fusion 的完整 R1 持续负载随后出现 Xid 145 / NVLINK_UNCORRECTABLE。独立 NCCL 测试定位到 NVLS 初始化卡在 cuMulticastBindMem。通过 NCCL_NVLS_ENABLE=0 验证普通 NCCL 路径。
4. SGLang 的 logits all-gather 另有独立的 PyTorch 对称内存多播路径，也卡在 cuMulticastBindMem；TORCH_SYMM_MEM_DISABLE_MULTICAST=1 已通过直接调用该代码路径的验证。

当前运行配方在 `040df93` 中统一设置以上两个环境变量，并固定 `--enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce`。没有修改推理内核源码或跳过模型计算。重建后的完整 R1 基线已在两台节点持续运行并全部通过；这验证了本次受限通信配置，未验证重新开启多播的可靠性。

`f0205e8` 统一了 600 秒单请求超时，并为机械搜索设置 1800 秒 quick、10800 秒 final 的进程超时。`67a9cf3` 将准备阶段的 KV 上限、关闭 radix 和关闭 overlap 限制从机械搜索中移除，保留原有四个可变维度和 90 种组合。修复均发生于正式优化之前；完整请求数、流量模式和吞吐目标不变。

早期诊断循环曾因各进程按各自时钟退出而在收尾处产生 collective mismatch；上表只采用改为 rank 0 广播停止信号后、正常退出的测试结果。

## 证据与恢复

节点上的输出根目录为 `~/ShrinkWrap-v2/InferenceBench/results/gb200_r1/20260930-c-r1/`。恢复证据包括 `recovery-tests.log`、`recovery-environment.json`、`recovery-codex-tool-probe.json`、`recovery-hpo-defaults-check.log`、`recovery-server.log` 和 `preflight/baseline_summary.json`。早期诊断包括 `nccl-synchronized.log`、`logits-no-multicast.log`、`nccl-preflight-stack.txt`、`logits-multicast-stack.txt` 和各次 `preflight-*` 记录。

devbox 的同名结果根目录下，`source-audit/` 保存源码哈希，`gb200-1/preflight/baseline_summary.json` 和 `gb200-2/preflight/baseline_summary.json` 保存已回收的基线摘要。远端 `pipeline_stage.txt` 和方法目录中的 `task/run_status.json` 是运行状态来源。既有失败记录保留，不重复启动已运行的队列。
