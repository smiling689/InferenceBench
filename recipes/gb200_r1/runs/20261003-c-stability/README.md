# 20261003-c-stability 重复实验

状态：两次独立的 gpt-6.1-sol / max 已启动，尚无新增正式成绩。机械搜索待对应节点的 Sol 正式评测结束后运行；分离搜索种子的三个文件已在 devbox 准备，dry-run 完成，等待用户批准实际同步。

## 范围和固定条件

用户要求新增两次 Sol，以及 Random Search 和 SMAC 各一次；不重测 Astra。每种方法保持 7200 秒优化预算、单节点四张 GB200、完整 DeepSeek-R1 的 FP8 dummy weights、BF16 激活/KV，以及原始 exec/resume 或 quick/full 工作流。

场景 C 的三种流量、请求数和温度保持首轮配置。开发集 seed=21，正式 held-out 评测 seed=1337，各 profile 256 请求。每次 Sol 使用空白 task 目录、独立的 Codex home 和原始空白 launcher；不提供首轮解法、插件或最终镜像。

| 节点 | Sol 优化 | 之后的机械搜索 | 远端 run ID |
| --- | --- | --- | --- |
| gb200-1 | Sol 第 2 次，已启动 | Random Search 第 2 次，计划搜索 seed=22 | 20261003-c-r2 |
| gb200-2 | Sol 第 3 次，已启动 | SMAC 第 2 次，计划搜索 seed=22 | 20261003-c-r3 |

机械搜索的随机种子与请求随机种子分离；不调整 90 点搜索空间、候选反馈请求数或失败计时规则。未指定新参数时保持首轮的默认随机性，包括 SMAC 原来的 Scenario 默认种子。正式运行前必须通过新代码的容器内测试。

## 启动证据

截至 devbox 2026-10-03 16:16:37 UTC：

- 两节点 GPU 原先空闲，Fabric Completed / Success；当日检查未发现新的 Xid。
- 两节点现有测试均为 18 passed；该结果验证的是未同步新参数前的原代码。
- 两节点经 SSH 反向转发完成真实 Codex CLI Responses / shell-tool 检查：模型 gpt-6.1-sol，effort=max，退出码 0。
- Sol 第 2 次容器 `ib-20261003-c-r2-gpt-6.1-sol` 正在运行，节点记录开始时间为 `2026-10-03T16:11:44.235688+00:00`。
- Sol 第 3 次容器 `ib-20261003-c-r3-gpt-6.1-sol` 正在运行，节点记录开始时间为 `2026-10-03T16:14:07.550398+00:00`。
- 两次 `run_status.json` 均记录 `budget_seconds=7200`、`status=optimizing`；solve.sh 记录 `RUN_INDEX=1`，不是恢复首轮会话。

节点时间原样保留，不能直接比较跨节点的时间戳。复用首轮通过的完整基线记录，标为历史 preflight，不作为本次的新测量。

首次启动队列发生 systemd 对 `${gpu_args[@]}` / `${common[@]}` 的参数展开，Docker 拒绝启动；此时任务和 Codex 目录仍为空，尚未开始优化计时。修复为 `systemd-run --expand-environment=no` 后启动。节点 2 在启动前又发生一次 SSH 中断，经确认没有实验容器或 `run_status.json` 后才重试。失败日志保留在远端方法目录的 `bootstrap_failure.log`。

devbox 的活动队列服务为 `ib-repeat-20261003-c-r2-queue-v2.service` 和 `ib-repeat-20261003-c-r3-queue-v3.service`。API 转发服务为 `ib-repeat-20261003-gb200-1-api.service` 和 `ib-repeat-20261003-gb200-2-api.service`。只绑定节点 localhost；认证未写入源码、镜像配置或实验清单。

## 待完成

1. 获得上述三个文件的同步批准，并在两节点容器内验证新搜索种子与固定请求种子分离。
2. 完成两次 Sol 优化、原工作流预览和全新容器的 seed=1337 正式评测。
3. 完成 Random Search、SMAC 各一次重新搜索及正式评测。
4. 归档结果、配置、失败尝试、源码哈希、实际耗时和可恢复的镜像/任务包。
5. 与[首轮正式成绩](../20260930-c-r1/README.md)合并，报告所有个体分数、均值、样本标准差、变异系数和范围；Sol 共 3 次，机械方法各 2 次。样本较少，不据此断言统计显著性，也不外推 dummy weights 的推测解码收益到真实权重。

8 节点小时是新增优化预算。服务启动、预览和最终评测额外计时。
