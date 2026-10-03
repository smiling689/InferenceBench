# 20261003-c-stability 重复实验

状态：两次独立的 gpt-6.1-sol / max 正在优化，尚无新增正式成绩。机械搜索按原代码和原搜索种子，在对应节点的 Sol 正式评测结束后运行。本轮不以代码同步为前提；分离搜索种子的三个文件仍是 devbox 上未同步的可选草稿。

## 范围和固定条件

用户要求新增两次 Sol，以及 Random Search 和 SMAC 各一次；不重测 Astra。每种方法保持 7200 秒优化预算、单节点四张 GB200、完整 DeepSeek-R1 的 FP8 dummy weights、BF16 激活/KV，以及原始 exec/resume 或 quick/full 工作流。

场景 C 的三种流量、请求数和温度保持首轮配置。开发集 seed=21，正式 held-out 评测 seed=1337，各 profile 256 请求。每次 Sol 使用空白 task 目录、独立的 Codex home 和原始空白 launcher；不提供首轮解法、插件或最终镜像。

| 节点 | Sol 优化 | 之后的机械搜索 | 远端 run ID |
| --- | --- | --- | --- |
| gb200-1 | Sol 第 2 次，已启动 | Random Search 第 2 次，原搜索 seed=21 | 20261003-c-r2 |
| gb200-2 | Sol 第 3 次，已启动 | SMAC 第 2 次，原 ConfigSpace seed=21 / Scenario seed=0 | 20261003-c-r3 |

机械重复继承首轮的随机序列和默认行为，不调整 90 点搜索空间、候选反馈请求数或失败计时规则。SMAC 2.3.1 的 Scenario 默认 seed=0 已在节点容器中通过实际构造函数签名确认。机械重复用于观察相同搜索种子下的运行、反馈和预算截断波动，不代表跨搜索种子的稳定性估计。Sol 的三次会话彼此独立。

## 启动证据

截至 devbox 2026-10-03 16:16:37 UTC：

- 两节点 GPU 原先空闲，Fabric Completed / Success；当日检查未发现新的 Xid。
- 两节点现有测试均为 18 passed；该结果验证的是未同步新参数前的原代码。
- 两节点经 SSH 反向转发完成真实 Codex CLI Responses / shell-tool 检查：模型 gpt-6.1-sol，effort=max，退出码 0。
- Sol 第 2 次容器 `ib-20261003-c-r2-gpt-6.1-sol` 正在运行，节点记录开始时间为 `2026-10-03T16:11:44.235688+00:00`。
- Sol 第 3 次容器 `ib-20261003-c-r3-gpt-6.1-sol` 正在运行，节点记录开始时间为 `2026-10-03T16:14:07.550398+00:00`。
- 两次 `run_status.json` 均记录 `budget_seconds=7200`、`status=optimizing`；solve.sh 记录 `RUN_INDEX=1`，不是恢复首轮会话。

16:26 UTC 的只读容器检查逐文件对照 `67a9cf3` 的 Git blob：两节点的 96 个已部署源码、配置和测试文件全部一致，没有内容差异。该版本中的 `recipes/gb200_r1/PREFLIGHT_20260930.md` 未部署；它是准备阶段记录，不参与运行。后续批准的种子修改应另记源码版本和哈希，不能继续把机械搜索标为未改动的首轮版本。

节点时间原样保留，不能直接比较跨节点的时间戳。复用首轮通过的完整基线记录，标为历史 preflight，不作为本次的新测量。

首次启动队列发生 systemd 对 `${gpu_args[@]}` / `${common[@]}` 的参数展开，Docker 拒绝启动；此时任务和 Codex 目录仍为空，尚未开始优化计时。修复为 `systemd-run --expand-environment=no` 后启动。节点 2 在启动前又发生一次 SSH 中断，经确认没有实验容器或 `run_status.json` 后才重试。失败日志保留在远端方法目录的 `bootstrap_failure.log`。

devbox 的活动 Sol 队列服务为 `ib-repeat-20261003-c-r2-queue-v2.service` 和 `ib-repeat-20261003-c-r3-queue-v3.service`。API 转发服务为 `ib-repeat-20261003-gb200-1-api.service` 和 `ib-repeat-20261003-gb200-2-api.service`。只绑定节点 localhost；认证未写入源码、镜像配置或实验清单。

最初的机械交接监听在节点 2 启动时遇到 SSH 中断；它未启动新的实验。交接控制改为节点上的独立 `nohup` Docker 管理进程，避免依赖长 SSH 连接。16:46:42 UTC 已确认 gb200-1 PID `1157699`、gb200-2 PID `874247` 存活，两次 Sol 容器也仍在运行。控制进程等待原 Sol 容器及正式评测容器实际结束，然后仅退出旧队列的同步等待段，使用原配方启动一个机械容器；优化会话和计时保持原样。控制 PID 和输出保存为远端 `baseline_controller.pid`、`baseline_handoff.log`。

## 待完成

1. 完成两次 Sol 优化、原工作流预览和全新容器的 seed=1337 正式评测。
2. 完成 Random Search、SMAC 各一次重新搜索及正式评测。
3. 归档结果、配置、失败尝试、源码哈希、实际耗时和可恢复的镜像/任务包。
4. 与[首轮正式成绩](../20260930-c-r1/README.md)合并，报告所有个体分数、均值、样本标准差、变异系数和范围；Sol 共 3 次，机械方法各 2 次。样本较少，机械搜索种子固定，不据此断言统计显著性，也不外推 dummy weights 的推测解码收益到真实权重。

8 节点小时是新增优化预算。服务启动、预览和最终评测额外计时。
