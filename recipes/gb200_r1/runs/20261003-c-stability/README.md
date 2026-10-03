# 20261003-c-stability 重复实验最终结果

四次补测完成：GPT-6.1-Sol / max 两次，Random Search 与 SMAC 各一次。四次均通过正式评测及逐请求审计，新增请求 **3072/3072 成功**；未追加 Astra。最终统计包含首轮结果并保留所有个体成绩。

## 成绩与稳定性

主指标为 burst / Poisson / constant 三种流量的请求吞吐几何平均，单位 req/s，越高越好。每次正式评测使用 seed=1337，每种流量 256 请求，共 768 请求；开发集反馈不计入正式成绩。

| 方法 | 第 1 次 | 第 2 次 | 第 3 次 | 算术均值 | 样本标准差 | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-6.1-Sol / max | 6.454397 | 2.130450 | 2.172502 | **3.585783** | 2.484382 | **69.2842%** |
| Random Search | 0.107483 | 0.154048 | — | **0.130766** | 0.032927 | **25.1799%** |
| SMAC | 0.107246 | 0.116132 | — | **0.111689** | 0.006284 | **5.6263%** |
| Astra / max，仅首轮参考 | 4.404222 | 未追加 | — | 4.404222 | — | — |

样本标准差分母为 n−1，CV 为样本标准差 / 均值。Sol n=3 的中位数为 2.172502，范围为 2.130450–6.454397，最大 / 最小为 3.0296。所有八个历史及新增正式运行均完成 768/768 请求，无失败的正式运行。

**首轮 Sol 高分没有稳定复现。** 两次新增成绩接近，但不能剔除首轮后宣称稳定；完整样本显示原工作流的优化结果波动较大。首轮使用 NEXTN 推测解码和采样路径优化，新增两次均未启用推测解码，并且图设置、调度和采样默认值等也不同，不能把全部分差归因于单一选项。

**执行或测量波动也存在。** Random 两次均在 gb200-1，选出的完整配置（21 个字段）完全相同，正式分数仍相差 43.3234%。不能把 Sol 的全部分差解释成 Agent 判断波动。SMAC 两次成绩相差 8.2864%，但选定配置不同，不能单独分离搜索结果与执行环境的贡献。

在本场景和原工作流下，Sol 各次成绩均高于两项机械方法。但 Agent 允许修改实现和更多服务参数，机械方法只有四个可变参数，并固定关闭图和推测解码；差距包含搜索范围差异，不证明纯搜索策略优势。样本量较小，不作统计显著性结论，也不能据 Astra 单次结果判断两个 Agent 模型的稳定排序。

机械重复保持原优化器种子：Random=21；SMAC ConfigSpace=21、Scenario 默认=0。因此 n=2 反映相同种子下的运行、反馈和预算截断波动，**不是跨搜索种子的稳定性**。使用 dummy 权重，路由与推测接受率可能不同于真实权重；不评估生成质量。

完整数值、每个原始正式 JSON 的 SHA-256、全部个体分数和统计位于 [stability_summary.json](stability_summary.json)。统计脚本 `summarize_stability.py` 在节点 CPU Docker 内执行，并从各 profile 吞吐重新核对几何平均。

## 正式 profile 结果

| 运行 | 节点 | burst req/s | Poisson req/s | constant req/s | 几何平均 req/s | 成功请求 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Sol 第 1 次 | gb200-1 | 6.546796 | 7.127967 | 5.761992 | 6.454397 | 768/768 |
| Sol 第 2 次 | gb200-1 | 3.325642 | 2.208057 | 1.316826 | 2.130450 | 768/768 |
| Sol 第 3 次 | gb200-2 | 3.367160 | 2.233022 | 1.363715 | 2.172502 | 768/768 |
| Random 第 1 次 | gb200-1 | 0.149097 | 0.126879 | 0.065639 | 0.107483 | 768/768 |
| Random 第 2 次 | gb200-1 | 0.210756 | 0.182262 | 0.095169 | 0.154048 | 768/768 |
| SMAC 第 1 次 | gb200-2 | 0.149722 | 0.126213 | 0.065275 | 0.107246 | 768/768 |
| SMAC 第 2 次 | gb200-2 | 0.132827 | 0.140084 | 0.084175 | 0.116132 | 768/768 |

首轮原始数据见 [20260930-c-r1](../20260930-c-r1/)。新增运行分别保存于 [20261003-c-r2](20261003-c-r2/) 与 [20261003-c-r3](20261003-c-r3/)；各方法目录含正式 metrics、状态、run record 与审计结果。大规模原始生成日志保留在节点任务目录和恢复包内，不纳入 Git。

## 固定条件与原工作流

- 每种方法使用单节点四张 GB200（GPU 0–3）、7200 秒优化预算；模型为完整 DeepSeek-R1，FP8 dummy 权重，BF16 激活/KV。
- 模型形状：61 层、hidden size 7168、256 routed experts、top-k 8、128 attention heads。模型 config SHA-256：`79ddea672a62e95d3f0be27be434375538e1988975971cffcf87c96c1de84a65`。
- 场景 C：输入/输出均为 820–1024 tokens；temperature=0.3，ignore_eos=true。burst 并发上限 64；Poisson 32 req/s、并发上限 32；constant 16 req/s、并发上限 16。各 profile 256 请求。
- 开发请求 seed=21，正式请求 seed=1337。请求文件 SHA-256 分别为 `e3329da09722133ae0d3fd368b3438b2c8f230815ab32ee32cb77773e71f3ba5` 与 `08fb8644ae4fee2939187086103a9a310866be4b9bd5f50b9c2aef2a343d6cae`。
- SGLang 0.5.15.post1、Torch 2.11.0+cu130、Transformers 5.12.1、Codex 0.156.1、SMAC 2.3.1、ConfigSpace 1.2.1，ARM64。普通 NCCL；NVLS/multicast/custom/fused allreduce 按 GB200 准备条件关闭。
- 运行源码冻结为 `67a9cf389bd7568afac740e0e7cb8b6edcddc59b`。原版 exec/resume 和 HPO quick/full 工作流保持不变。两次新增 Sol 均从空白任务、独立 Codex home、空白 launcher 开始，RUN_INDEX=1，没有提供首轮解法或最终镜像。
- 原代码测试在两节点各为 18 passed。实验结束后，在 CPU Docker 内再次逐文件对照冻结 Git blob：两节点各 96 个已部署源码、配置、测试均一致。唯一未部署文件是非运行文档 PREFLIGHT_20260930.md，详见 [runtime_source_final_verification.json](runtime_source_final_verification.json)。
- 分离搜索种子的三个可选草稿仅在 devbox，未同步、未用于本轮。不下载模型权重或数据集，不进行准确性测试。

## Sol 选定配置与审计

两次新增服务的实际配置保存于各 `formal_server_info.json`。均为 TP4/EP1/DP1、KV 上限 147456、请求上限 64、TRT-LLM MLA 与 FlashInfer TRT-LLM MoE；radix cache 关闭，stream interval=16，server seed=42，sampling-defaults=model。首轮 Sol 为 sampling-defaults=openai。

- 第 2 次：prefill chunk 8192；decode CUDA Graph buckets 1/2/4/8/16/24/32/48/64，prefill 图关闭。任务局部 `inference_tuning/sitecustomize.py` 对 TP prefill 入队做等待，最多 80 次调度、divisor=4；实际执行继续调用原 scheduler 方法。
- 第 3 次：prefill chunk 4096；decode buckets 1–64；prefill 为 tc_piecewise，buckets 1024/2048/4096、eager compiler；scheduler receive interval=8。
- 两次均无推测解码。已检查冻结容器的差异，安装的 SGLang（editable 源码树）、Torch、Transformers 源码未修改；镜像配置中没有 API key。

`verify_sol_results.py` 在两个节点 CPU Docker 内均通过。逐条核验 held-out 请求字段、三种流量各 256 个不同请求、成功且非空的生成记录、server-reported 输入/输出 token 数；每个 profile 要求生成 235820 output tokens。审计证明记录完整性和所核对的服务配置，不证明生成质量或所有数值路径的正确性。

Sol 第 2 次首次 exec 返回 0，使用剩余 48 秒 resume 后返回 124；第 3 次 exec 用满预算返回 124。这是预期预算超时。两个预览容器及两个独立正式容器均退出 0。

## 机械搜索与审计

| 方法 | 尝试次数 | 有效快速评测 | 不同有效配置 | 最佳 trial_idx | 实际搜索计时，秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random 第 2 次 | 9 | 8 | 8 | 6 | 7200.673079 |
| SMAC 第 2 次 | 10 | 8 | 4 | 3 | 7211.103637 |

实际计时来自原始 run_end.budget_used_s，包括搜索结束时的原版清理耗时；没有补时或重启搜索。HPO 计时开始前的模块加载/初始化约 35 / 37 秒不属于搜索预算。

| 方法 | 请求上限 | prefill chunk | 显存比例 | 调度 |
| --- | ---: | ---: | ---: | --- |
| Random，首轮与本轮相同 | 256 | 4096 | 0.9 | lpm |
| SMAC 首轮 | 128 | 2048 | 0.9 | fcfs |
| SMAC 本轮 | 32 | 8192 | 0.92 | lpm |

均为 TP4/EP1/DP1、FP8 dummy、BF16 激活/KV、图和 torch compile 关闭、无推测解码。实际 `cuda_graph_config.decode/prefill.backend=disabled` 已核对；SGLang 会把旧 CLI 图开关规范化为该配置。

原版 HPO 搜索空间为 90 点、四个可变参数，quick 每个 profile 仅 4 请求，小于所有请求上限候选 16–256。两节点 HPO 源码 SHA-256 都为 `be9b857c8b1ef0b036e12af406439d250ca8b6d3da01dc329ae62eb1875c75e8`。

失败均保留原 server_start_failed 标签并计入预算：Random trial 8 在服务启动时预算到期，只留下模块导入日志，不能推断已触发端口校验；SMAC trial 1 的 traceback 确认 HTTP 55603 派生 gRPC 65603 越界，trial 9 在启动时预算到期。没有人工重选最佳配置。

`collect_hpo_search.py` 的搜索/选定配置核验与 `verify_hpo_results.py` 的全量正式审计均通过。后者核对 3×256 记录、精确 token 数、固定输入、完整模型形状、实际配置、冻结源码和由 profile 重算的正式分数。两项正式容器退出码均为 0。

原版 run_scenario_eval 只在读取 metrics 后的内存对象中增加 wall_s，磁盘 evaluator JSON 没有此字段。两项旧审计器因此首次比较失败；兼容该唯一包装字段后，全部 evaluator 字段仍严格相等，审计均返回 0。实验没有重跑。metadata 采集器早期 server_args 包装兼容问题也只影响观察，不改变搜索或请求。

## 实际耗时与恢复归档

| 节点 | Agent 至机械正式评测结束的流程时长 | 配置的优化预算 | 预算外流程耗时 |
| --- | ---: | ---: | ---: |
| gb200-1：Sol 第 2 次 + Random 第 2 次 | 5.879762 h | 4 h | 1.879762 h |
| gb200-2：Sol 第 3 次 + SMAC 第 2 次 | 6.278437 h | 4 h | 2.278437 h |
| 合计 | **12.158199 节点小时** | **8 节点小时** | **4.158199 节点小时** |

两节点并行，单节点流程最长约 6.28 h；4 h 只覆盖优化预算。启动、预览、全量正式评测额外计时：Random 正式 evaluator 为 5326.372351 秒，SMAC 为 6813.920721 秒。流程时长由各节点自己的 Docker timestamps 相减，不跨节点直接相减，也不包含随后恢复归档。原始计时见各机械目录的 pipeline_timing.json。

两节点原生 Docker save 都因既有 Ubuntu 基础层缺失 /home/ubuntu/.bash_logout 失败。使用持久磁盘上既有基础镜像层及本轮 committed delta 的 tar-split 原始字节重建完整包：**两份原镜像 ID 均保持不变，每份全部 71 层 diff ID 匹配，Docker load 均成功**。重建只读 Docker storage，随后以正常 Docker load 验证，未手动修改 storage 或实验源码；详见各运行目录的 image_archive_rebuild.json 和原生错误文本。

归档保存在每个节点 `/srv/shrinkwrap-data/inferencebench/images/`。CPU Docker 逐文件核验：gb200-1 798 个常规文件、gb200-2 807 个常规文件均与源内容一致。包含任务、日志、固定请求、模型配置及 tokenizer 元数据，不含 Codex home、权重或数据集。

完整包路径、大小、SHA-256 与镜像 ID 见 [recovery_archives.json](recovery_archives.json)。恢复需配套源码 commit 67a9cf3；机械方法使用基础镜像与归档 task，临时编译缓存可以重新生成。

初始化队列曾在优化开始前因 systemd 展开 Bash 数组失败，当时任务与 Codex 目录为空；日志保留于恢复包。后续使用原始预算与独立会话，无补时或替换。API 转发已在 Agent 优化结束后停止。最终资源状态以 resource_cleanup.json 为准。
