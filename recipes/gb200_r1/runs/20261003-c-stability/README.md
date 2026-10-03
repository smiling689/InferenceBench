# 20261003-c-stability 重复实验

状态（devbox 2026-10-03 18:36:41 UTC）：两次新增 Sol 优化和独立正式评测均完成，全部 1536 个新增正式请求成功。Random Search、SMAC 各一次重跑已在对应节点自动开始；其新增正式成绩仍待完成。此报告尚不是整轮最终结果。

## Sol 正式结果与稳定性

只采用 seed=1337、全新容器中的 held-out 正式评测。三次独立会话的正式结果均保留，不用开发集成绩替代。

| 独立运行 | 节点 | burst req/s | Poisson req/s | constant req/s | 几何平均 req/s | 成功请求 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Sol 第 1 次，[首轮记录](../20260930-c-r1/gpt-6.1-sol-max/) | gb200-1 | 6.546796 | 7.127967 | 5.761992 | **6.454397** | 768/768 |
| Sol 第 2 次，[本轮记录](20261003-c-r2/gpt-6.1-sol-max/) | gb200-1 | 3.325642 | 2.208057 | 1.316826 | **2.130450** | 768/768 |
| Sol 第 3 次，[本轮记录](20261003-c-r3/gpt-6.1-sol-max/) | gb200-2 | 3.367160 | 2.233022 | 1.363715 | **2.172502** | 768/768 |

| 统计量 | Sol，n=3 |
| --- | ---: |
| 算术均值，req/s | 3.585783 |
| 样本标准差，req/s，分母 n−1 | 2.484382 |
| 变异系数，标准差 / 均值 | 69.2842% |
| 中位数，req/s | 2.172502 |
| 最小值–最大值，req/s | 2.130450–6.454397 |
| 最大值 / 最小值 | 3.0296 |
| 成功 / 失败的正式运行 | 3 / 0 |

一次高分和两次约 2.15 req/s 的成绩显示，原 Agent 工作流在这组条件下的优化结果有明显波动。不能把首轮 6.45 req/s 当成已证实的稳定表现。三次都完成全部请求，因此这是优化结果波动，未观察到正式评测失败。

首轮 Sol 使用 NEXTN 推测解码以及采样路径优化；新增两次均未启用推测解码。并行度、调度、图设置和服务采样默认值也有差异，不能把全部分差归因于某一个选项。样本较少，不据此作统计显著性结论；Astra 仍只有首轮一次 4.404222 req/s，不追加测试，也不能据此判断两个模型的稳定排序。

## 新增 Sol 的配置与核验

两个新增运行均为完整 DeepSeek-R1：61 层、hidden size 7168、256 routed experts、top-k 8、128 attention heads；模型配置 SHA-256 为 `79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65`。正式服务的实际参数保存于各目录的 `formal_server_info.json`，已核对 TP4/EP1/DP1、FP8 dummy 权重、BF16 激活/KV、KV 上限 147456、请求上限 64，以及普通 NCCL 通信约束。

- 第 2 次：prefill chunk 8192；decode CUDA Graph buckets 为 1/2/4/8/16/24/32/48/64，prefill 图关闭。任务目录的 `inference_tuning/sitecustomize.py` 对 TP 本地 prefill 入队做批处理等待，最多 80 次调度、divisor=4；实际模型执行继续调用原 scheduler 方法。
- 第 3 次：prefill chunk 4096；decode buckets 为 1–64，prefill 使用 `tc_piecewise`、buckets 1024/2048/4096，compiler 为 eager，scheduler receive interval=8。
- 两次均使用 TRT-LLM MLA、FlashInfer TRT-LLM MoE，radix cache 关闭，stream interval=16，服务随机种子=42，`sampling-defaults=model`。首轮 Sol 使用 `sampling-defaults=openai`，这是 Agent 选择的服务参数差异。
- 两次冻结后的容器差异检查未发现安装的 SGLang、Torch 或 Transformers 源码被修改；最终镜像配置中没有 API key。

各运行的 `result_audit.json` 记录独立核验结果：原 held-out 请求文件哈希一致，messages、temperature=0.3、ignore_eos=true、输入/输出长度等字段未变；三种流量各 256 个不同请求；全部 768 条生成记录成功、非空，server-reported 输入/输出 token 数逐条与请求相符。每个 profile 要求生成 235820 个输出 tokens。审计脚本保存为 `verify_sol_results.py`，在两个节点的 CPU Docker 容器中各执行一次，均通过。它核验日志完整性和实际服务配置，不构成生成质量或所有数值路径的正确性证明。

准确 launcher、局部插件、原始正式 JSON、审计证据、镜像 ID 和容器时间保存在上述运行目录。原始大规模生成日志仍保留在节点任务目录；不纳入 Git。

## 机械重复正在运行

| 方法 | 节点 | 本次开始时间，节点 UTC | 优化预算 | 当前状态 |
| --- | --- | --- | --- | --- |
| Random Search 第 2 次 | gb200-1 | 2026-10-03 18:31:31.416200 | 7200 秒 | 优化中 |
| SMAC 第 2 次 | gb200-2 | 2026-10-03 18:32:52.830818 | 7200 秒 | 优化中 |

两个节点使用原版 HPO 源码，SHA-256 均为 `be9b857c8b1ef0b036e12af406439d250ca8b6d3da01dc329ae62eb1875c75e8`。Random optimizer seed=21；SMAC ConfigSpace seed=21，SMAC 2.3.1 Scenario 默认 seed=0。开发请求 seed=21，正式请求 seed=1337；90 点搜索空间、4 个可变参数、每个 profile 4 请求的快速反馈和失败计时行为保持原版。

机械重复继承首轮搜索随机序列，观察相同优化器种子下的运行、反馈和预算截断波动，不代表跨搜索种子的稳定性。首轮机械正式分数为 Random 0.107483、SMAC 0.107246 req/s；新增结果尚未生成，不在这里提前计算它们的重复统计。

分离搜索种子的三个可选草稿仅在 devbox，未同步、未用于本轮。原始节点队列在 Sol 正式评测后，由节点独立控制进程退出旧同步等待段，再启动一个机械容器。对应旧队列服务退出是正常交接，不能作为实验失败。两个节点的 API 转发均已在 Agent 优化结束后停止。

## 固定条件与启动证据

每种方法为单节点四张 GB200、7200 秒优化预算、FP8 dummy 权重与 BF16 激活/KV。不用下载权重或数据集，不评估生成质量。原始 exec/resume 或 quick/full 工作流保持不变；新增两次 Sol 均以空白任务目录、独立 Codex home 和空白 launcher 启动，`RUN_INDEX=1`，未提供首轮解法、插件或最终镜像。

场景 C：输入和输出均为 820–1024 tokens，temperature=0.3，ignore_eos=true；burst 并发上限 64，Poisson 32 req/s / 并发 32，constant 16 req/s / 并发 16；各 profile 256 请求。目标为三个 profile 请求吞吐的几何平均。

运行源码固定为 `67a9cf389bd7568afac740e0e7cb8b6edcddc59b`。启动前两节点 GPU 空闲，Fabric Completed / Success，当日未发现新 Xid；现有原代码测试均为 18 passed，真实 Codex Responses / shell-tool 探测均成功。16:26 UTC 对照 Git blob，两个节点的 96 个已部署源码、配置和测试文件完全一致；一个未部署的准备 Markdown 不参与运行。这里没有声称未同步的新参数草稿通过测试。

首次队列启动因 systemd 展开 Bash 数组而失败，发生在优化开始前，任务及 Codex 目录仍为空。修复管理命令后启动；失败日志保留为 `bootstrap_failure.log`。此后未重启、补时或替换优化会话。中途 SSH 观察连接超时不改变实验，节点独立控制进程管理交接。

Sol 第 2 次首次 exec 以 0 结束，再用剩余 48 秒 resume，最终返回 124；第 3 次 exec 用满预算，返回 124。两个预览容器和两个独立正式容器均以 0 结束。节点时间原样保留；跨节点时钟存在偏差，耗时由同一节点的时间相减，详见各 `run_record.json`。两次 Agent 容器总耗时包含预算外预览，不能当成额外优化时间。

## 待完成

1. 完成两次机械搜索及其全量正式评测，核验全部请求和所选配置。
2. 归档最终镜像、任务包和输入元数据到节点持久磁盘，记录完整性哈希。
3. 更新全部个体成绩、机械方法 n=2 的均值/样本标准差/CV/范围、实际节点耗时，释放本轮资源。

8 节点小时仅是新增四份优化预算。服务启动、预览和正式评测额外计时。随机权重会影响路由和 MTP 接受率，不能外推到真实 R1 权重或生成质量。Agent 允许修改实现和更多服务参数，机械搜索只有四个可变维度并固定关闭图和推测解码，方法间差距包含搜索范围差异。
