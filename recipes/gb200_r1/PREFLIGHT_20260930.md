# 2026-09-30 准备阶段记录

截至本次恢复检查，**四组正式优化均未启动，尚无可比较的实验成绩**。devbox 约 03:18 UTC，最后一次完整 R1 重启期间，两台节点的 SSH 会话同时中断；跳板机可登录，但从跳板机连接两台节点的 SSH 端口均超时。后续连接重试已成功，正在恢复运行环境，仍须完成完整场景验证。

## 重启后的恢复检查

- 两台节点 SSH 均已恢复，uptime 确认期间发生过重启；当前四张 GPU 均空闲，Fabric 为 Completed / Success / Healthy，本次启动的内核日志未发现 Xid。
- Docker 的实际 `data-root` 为 `/dev/shm/docker`。重启后镜像和容器均已丢失；系统盘上的源码、请求集和已保存的诊断日志仍在。不能恢复此前仅存于容器内部的编译缓存或未导出的日志。
- 两台节点的 Dockerfile、实验入口、宿主机 wrapper 和服务启动脚本均与 devbox 的 SHA-256 一致；两个请求文件的哈希也与下表一致。
- 已用同一基础镜像 digest 启动重建，并挂起恢复队列：检查环境和 API、运行测试、启动完整 R1、完成 smoke 和完整基线，通过后才启动原定四组实验。恢复队列启动不代表正式优化启动。
- 恢复队列同时安排将实验镜像导出到持久磁盘 `/srv/shrinkwrap-data/inferencebench/images/`；仅成功导出后才生成最终 `.tar` 文件，便于未来通过 `docker load` 恢复。未修改系统 Docker 存储设置。
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
| 搜索参数兼容性 | 两台节点已安装的 SGLang CLI 校验通过 |
| 完整 R1 短请求 | 此前两台节点均返回完整 16-token 流式响应；不是吞吐成绩 |
| 普通 NCCL 压力测试 | 关闭 NVLS 后，两台节点分别运行 300.006 / 300.015 秒，完成 2,645,700 / 2,096,000 次 all-reduce，数据校验通过、进程正常退出 |
| logits 通信回退 | 直接调用 SGLang 的 create_state，八个 GPU 进程均确认 multicast_ptr=0，并通过 NCCL all-gather 数据校验 |
| 完整场景 C 基线 | 尚未通过；正式优化的启动条件仍未满足 |

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

当前运行配方在 `040df93` 中统一设置以上两个环境变量，并固定 `--enforce-disable-flashinfer-allreduce-fusion --disable-custom-all-reduce`。没有修改推理内核源码或跳过模型计算。最后一次使用全部修复的完整 R1 验证因 SSH 中断而状态未明，不能据此宣称问题已全部解决。

早期诊断循环曾因各进程按各自时钟退出而在收尾处产生 collective mismatch；上表只采用改为 rank 0 广播停止信号后、正常退出的测试结果。

## 证据与恢复

节点上的输出根目录为 `~/ShrinkWrap-v2/InferenceBench/results/gb200_r1/20260930-c-r1/`。关键文件包括 `nccl-synchronized.log`、`logits-no-multicast.log`、`nccl-preflight-stack.txt`、`logits-multicast-stack.txt` 和各次 `preflight-*` 记录。连接中断前，新增远端记录尚未全部回收到 devbox；devbox 已保留最初故障诊断和源码哈希记录。

恢复连接后，先核对 GPU、现有容器、队列状态和上述证据；避免重复启动评测客户端。随后重新完成完整场景 C 基线，并以通过后的时间作为正式优化起点。既有失败记录不删除，四组方法的完整优化预算尚未使用。
