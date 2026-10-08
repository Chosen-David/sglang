# 因果任务拆分与资源并发核对_by_gpt

核对日期：2026-10-08。当前继续同一未完成 host-consumer 批次，不计完成轮次。

## 最新 main 与文档归属

固定 main：`e837d17c5c894ef468e301e4db966984c0be1ffb`。远端已采用 `<PROJECT_ROOT>/agent_doc/`，Project A 的指南、任务与结果留在 Project A；升级 agent 本身使用其自己的 agent_doc。普通 doc/docs 不自动迁移。人类指南目录保持只读。

本地仍有旧 `doc/` 路径未提交记录与 state/CODEMAP 修改，来源为此前恢复检查点。仅 fetch 和只读审计，未快进本地、未重写历史证据、未写指南。dirty-inventory.json 固定每个文件的路径与哈希；tracked-changes.patch 保留已跟踪差异。最新远端 state 不含本地 unfinished_batch_recovery，直接采用远端摘要会丢失当前未完成批次；必须显式保留并协调，不能重置论文或实验进度。

## 实际实现与范围

| 检查位置 | 实际行为 | §18 的缺口/边界 |
|---|---|---|
| core.py:339–430 | 每 tick 一个 claim，依赖/授权/租约门禁、持久化轮转游标 | 不等于每条链只能有一个外部作业；也未提供资源匹配优先队列 |
| core.py:431–468 | 同步 handler.run；token、取消、租约再核验后写入结果 | adapter 应短时 submit/poll；长时同步 handler 会阻塞该 worker |
| task_supervisor.py:113–127 | 每个 state DB 的唯一 managed worker，worker.lock | 应保留该互斥，不以新增独立 worker 规避所有权 |
| scheduler.py:drain_once | 定时 reservation 后调用 tick | 定时去重不是 GPU/显存/I/O 预约 |
| core.py:validate_plan | task_id、depends_on、估时、risk、尝试与观察预算 | 当前已读契约未要求冻结 shard 全集、资源分配、父子 ID 与汇总完整性 |

重要区别：异步 adapter 返回 pending 后可以提交其他独立作业，实际并发度依赖真实 adapter/资源；本次没有 adapter/jobID/live receipt，不能声称已有多 GPU 加速。原 host 执行者未知，未接管旧运行。

## 修改前冻结的最小验收提案

主要问题：就绪任务与实际可用资源没有可核验绑定，分片完整性与独立验收未成为汇总的显式合同。

机制候选：在原 Engine/adapter 生命周期内记录资源需求、已授权资源快照、预约/jobID、稳定子 ID 与不可变全集；先验分片后局部消费，全局汇总等待必需集合。主 AI 串行维护 TASK，不添加第二套 done。

反事实预期：现有 DAG 能控制依赖，却不能仅凭 core readiness 证明没有共卡、遗漏、重复、混入不同实验版本或违规拆分。

| 预定用例 | 硬验收 |
|---|---|
| 8 个独立分片、8 个实际获准合适资源 | 每片独立 jobID/输出/幂等键，先独立代码数据验收，再完整汇总 |
| 资源不足 | 有界排队；不超预约容量、不新增付费资源；取消释放所属预约 |
| 单分片失败/重启 | 仅重试受影响片，保留成功片和原预算；未知副作用先核实 |
| 共享拟合/训练状态不可拆 | 拒绝伪独立，记录原因；不能通过分片标签改变统计语义 |
| 7/8、重复片、版本/单位/权重漂移 | 全局结论与发布阻断；无关已验收局部消费保留 |
| 通信纠正与迟到旧结果 | 旧版本拒收、原 finding 保留、受影响消费者重新核验 |

主要指标：硬约束违反为 0；遗漏/重复/漂移必须全部拒绝；独立分片及汇总验收有真实证据。性能采用同条件 makespan，并分解排队/启动/计算/验证/汇总；收益阈值须由真实宿主基线后、候选运行前冻结，当前未知。停止条件：资源/权限不明、共享状态无法证明独立、预算耗尽或未知副作用，不增加资源、不绕过门禁。

这只是待复核方案，不是执行计划批准、程序测试通过或模型/多 GPU 效果证据。研究写作、图形、全部角色及批次发布门禁仍保留。

## 检查点与接续

本次新增全文精读 0；此前新证据累计 3 篇，历史报告 7 篇的原始笔记仍缺失，不能合计宣称 10/10 可验收。没有生产修改、程序实验、真实角色任务、真实交接、A/B 或外部集成。未 commit/push；原因是迁移整合和完整批次门禁尚未完成，不是已确认无 GitHub 写权限。

下个接续点：先逐文件协调旧检查点与 agent_doc 命名空间、保持冻结记录字节；在干净最新基线上检索已有结果，再冻结真实 submit/poll、资源预约与 fan-in 合同，经独立主控审核后实施/测试。不得自动恢复未知旧作业。

机器检查点：checkpoint.json；源码均由固定 commit 读取并存 SHA256。此报告与检查点保存在仓库外，避免与最新 main 的迁移及其他提交发生覆盖。

结束前远端再次更新为 `acf85d76f9aa1bcf6c0e657b50a46a4f225c09a8`（2026-10-08T03:04:11Z）。8 个文档文件新增目录职责说明；本次审计所用 runtime/workflow 字节未变，TASK 新增 DOC-ROLES-01。旧审计仍固定原 SHA，不把它视为新候选发布验收。

<details><summary>接续机器检查点</summary>

```json
{
  "batch_id": "2026-10-07-host-consumer",
  "wake_id": "W26",
  "observed_at": "2026-10-08T03:04:39.701942+00:00",
  "owner": "/root",
  "platform_run_id": null,
  "status": "checkpoint_saved_not_live",
  "baseline_commit": "e837d17c5c894ef468e301e4db966984c0be1ffb",
  "local_head": "b2889309810fe9ad808bfe4fef1cd218c6cbdabd",
  "sync": "fetch and pinned read-only audit only; dirty checkout NOT fast-forwarded; preserved without overwrite",
  "previous_checkpoint": "doc/results/host-consumer-w25/checkpoint.json",
  "historical_papers_reported": 7,
  "historical_evidence": "unavailable_not_revalidated",
  "new_fulltext_credit_this_wake": 0,
  "new_fulltext_credit_cumulative": 3,
  "required_papers": 10,
  "registered_roles": 15,
  "runtime_receipt": false,
  "local_runtime_directory_present": false,
  "prior_host_executor": "unknown; not taken over",
  "knowledge_status": "not_needed_for_static_control_flow_and_migration_audit; no theoretical claim or corpus-use claimed",
  "production_edits": 0,
  "program_tests": "not_executed",
  "role_tasks": "not_executed",
  "cross_role_handoff": "not_executed",
  "fair_ab": "not_executed",
  "external_integration": "not_executed",
  "publication_status": "not_committed_not_pushed",
  "counts_toward_convergence": false,
  "source_refs": [
    {
      "path": "AGENTS.md",
      "sha256": "c5b8cfd30519abba9ac29ccd4823e32f32ad3b128a5b874f6086037ba31771c1",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "prompts/decision_review.md",
      "sha256": "55b793450ff0e3d29696dae7626afab7f15d1ab843baa396dcc1d564e532c849",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "config/role_registry.json",
      "sha256": "c231d5031aab337845de3e970f0253fc8d5156be3a3bdff839f2ebf696f7c5eb",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "config/backend_registry.json",
      "sha256": "bb57ec64497a8d98da14b609132cd11f90e11bac9fb6fc3cce8f1cec8c2ac138",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "workflows/project_document_workflow.md",
      "sha256": "4248d340519a568cd3a2ff4d85e09ba3ca158ca81138ec47c4ab73ac48dfae08",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "workflows/project_memory_workflow.md",
      "sha256": "44f90314fd45c2e74c83b87a5bb56a4108045a6e44c60230d62e9530b2f61e9e",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "workflows/knowledge_access_workflow.md",
      "sha256": "61d702744b1c89b753defcb09ee5a727014bd3153aecca20166ae44c0c0fa091",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "workflows/agent_communication_workflow.md",
      "sha256": "349924855296cfa1f4982d8c7bae386ebf1c91036470c83f39706d36a6b6d99e",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "workflows/task_supervision_workflow.md",
      "sha256": "714c9ae213935a9280ba39e2822a74b151d57c4c77fa339963ddc603642cdd1d",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_runtime/core.py",
      "sha256": "198753673877360b828ba89bccc68bba1da70996a0f26fd8d2b5f7049ed94298",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_runtime/task_supervisor.py",
      "sha256": "fbff2de4f50a74a6d3e67ff28e4ac2182f245fcca666fe3a0069874e12d25c1a",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_runtime/scheduler.py",
      "sha256": "e4041354dab925c9a6363cee619ade06e94f941b44f101ccd39c8d33fafdaf42",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_runtime/communication.py",
      "sha256": "2df62b823606ca4d0ff1059c632dc567180ba816dd126f2ecd6ed84168963a7c",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "docs/continuous_optimization/state.json",
      "sha256": "3821bd35408866c04dd6a111ccc5e53a9bc78371cfd2f95f72e8a225348a79a9",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_doc/task/TASK.md",
      "sha256": "d8ed6391cdccceaa3ebde50c8e880cec607654aa7df101cc71b9182edec4603c",
      "commit": "e837d17c5c894ef468e301e4db966984c0be1ffb"
    },
    {
      "path": "agent_doc/task/TASK.md",
      "commit": "acf85d76f9aa1bcf6c0e657b50a46a4f225c09a8",
      "sha256": "197c8bf2c70ff4927d0c4a1f09a4b764cc6b104e592b431a04378218409bf46c"
    }
  ],
  "dirty_inventory": [
    {
      "path": "doc/results/host-consumer-w13/checkpoint.json",
      "sha256": "55943fb6b65a06cd5ec49c03824c95e39193d05bf6844f1ff0e5db7011985a90",
      "bytes": 3552
    },
    {
      "path": "doc/results/host-consumer-w13/document-inspection.json",
      "sha256": "6ea1969c392049fe2fdc5bf07dbe52bcc084915c6c09466d5b13c0f915df9827",
      "bytes": 23212
    },
    {
      "path": "doc/results/host-consumer-w13/independent-recovery-review.json",
      "sha256": "f9aafb26dc95f87ae3d13cf524c18be119297886910d5e85fc39329accf0aed4",
      "bytes": 20340
    },
    {
      "path": "doc/results/host-consumer-w13/knowledge-use.json",
      "sha256": "bc6d85c057d0ec30ce9b796cc0cc585be38566f491f753cb711296b98bb23e3e",
      "bytes": 407
    },
    {
      "path": "doc/results/host-consumer-w13/prior-result-review.json",
      "sha256": "00b9881ea3e1bd4ab9e14d113f7f90db363c7d05b37b0a4d6b1d9093f1138420",
      "bytes": 3871
    },
    {
      "path": "doc/results/host-consumer-w13/prior-search.json",
      "sha256": "42aad497cfdcc3db34a8a565a6f8d80786128fd521c1dd6cb470b2886b615e7b",
      "bytes": 4392
    },
    {
      "path": "doc/results/host-consumer-w13/recovery-plan.json",
      "sha256": "5a83f6965030a668b9c254d6c2d8431b8fee1a6c1d8b900bd85621d58268378f",
      "bytes": 5196
    },
    {
      "path": "doc/results/host-consumer-w13/state-preservation-check.json",
      "sha256": "4264e3d9c71ff45945bbe3947e836f64268678e8aef1c2657e3b9faffae7a7f6",
      "bytes": 224
    },
    {
      "path": "doc/results/host-consumer-w14/async-control-partial.md",
      "sha256": "fb38f758d8531c09a8b0eb9ed89992e4c04258ca712667745521ee3c3bf724df",
      "bytes": 4696
    },
    {
      "path": "doc/results/host-consumer-w14/checkpoint.json",
      "sha256": "fba76cd890d374ed2161f54a6208721dba044675f977cca2e707ac994b8275b8",
      "bytes": 2793
    },
    {
      "path": "doc/results/host-consumer-w15/checkpoint.json",
      "sha256": "f7ab00fa6fa5aaa1b8ff56bd676273b0b2450a0165126c102008b52493944c37",
      "bytes": 2209
    },
    {
      "path": "doc/results/host-consumer-w15/source-assessment.md",
      "sha256": "eebb316ae244122e8d99b9f6fea432a8e2853b68603493cabbce9c1806312466",
      "bytes": 6072
    },
    {
      "path": "doc/results/host-consumer-w16/checkpoint.json",
      "sha256": "3f5d8cc8e3fec073b70999a7580abd504767145dae90117dd4e196b676c9ad73",
      "bytes": 1673
    },
    {
      "path": "doc/results/host-consumer-w16/reading-checkpoint.md",
      "sha256": "eb2b10c8509547ad0447e0a4947830544d5a1c2e1934d67e13402d8ad71db74d",
      "bytes": 5399
    },
    {
      "path": "doc/results/host-consumer-w17/checkpoint.json",
      "sha256": "62fb3783898abb1bae93a8c1bc2cdd376427968c1a4bf12d02b22e8d316ad9ad",
      "bytes": 1806
    },
    {
      "path": "doc/results/host-consumer-w17/reading-checkpoint.md",
      "sha256": "6df2847b88a89974b5313b41236d798b58f43ef493efbf6395fc4991255df110",
      "bytes": 5584
    },
    {
      "path": "doc/results/host-consumer-w18/checkpoint.json",
      "sha256": "157ae911c01889f49b492f2a8e7dd08e0ff4b39a19aab2b9a270a267de7e51b3",
      "bytes": 1631
    },
    {
      "path": "doc/results/host-consumer-w18/reading-checkpoint.md",
      "sha256": "1a2cf1f6b6d8577755b2bb299ee9710c42ff14c6e69aea58d13a124e8287ae22",
      "bytes": 5244
    },
    {
      "path": "doc/results/host-consumer-w19/checkpoint.json",
      "sha256": "2a91a23edfbaa08fedd292a304dc080f222d3bf7dd5f67c5cdffdf65a7e36357",
      "bytes": 2258
    },
    {
      "path": "doc/results/host-consumer-w19/reading-checkpoint.md",
      "sha256": "6ca6b761b77ad7c1ce79406b3e9b4c650fe2e6c950943ae04f097dfab700f06f",
      "bytes": 3487
    },
    {
      "path": "doc/results/host-consumer-w20/checkpoint.json",
      "sha256": "e18c88eae297a08dfe45877c22ab918c199fe533357d3e1a6e8c827ece8e366f",
      "bytes": 2424
    },
    {
      "path": "doc/results/host-consumer-w20/paper.json",
      "sha256": "af2ab31174bef3d88662ce4cc95e12d4dbabc438bf234cc688f7df88dfada38e",
      "bytes": 2966
    },
    {
      "path": "doc/results/host-consumer-w20/reading-checkpoint.md",
      "sha256": "5f5e499706d1cef01e14c6e5b94c69c68a10d32d1258e12d60eaeed68158e03d",
      "bytes": 3314
    },
    {
      "path": "doc/results/host-consumer-w21/checkpoint.json",
      "sha256": "338abaa556091c93080f81d7a79f2f69fa3af69ed3782c66546bdbef660542a5",
      "bytes": 2541
    },
    {
      "path": "doc/results/host-consumer-w21/reading-checkpoint.md",
      "sha256": "458b6a4356ae7c8dd374c76c53ae678dc4e6977f8a4096ff42164b644d387cff",
      "bytes": 3245
    },
    {
      "path": "doc/results/host-consumer-w22/checkpoint.json",
      "sha256": "b324feb484490bc9b067ebd9c26ab0134059c924c1078f652614e76882ce59c1",
      "bytes": 3079
    },
    {
      "path": "doc/results/host-consumer-w22/paper.json",
      "sha256": "59e794f757330356aa809059e9a89387099a138abbd50324f2f6fe9709171c64",
      "bytes": 2102
    },
    {
      "path": "doc/results/host-consumer-w22/reading-checkpoint.md",
      "sha256": "d940fb3337dd871b737decdc7a0cf13e7c29528ea7fe7be745be77a54036fd48",
      "bytes": 6812
    },
    {
      "path": "doc/results/host-consumer-w23/checkpoint.json",
      "sha256": "7af5ff3f50e0610a9918eaa29d30feca8e8b07c31ae130fecd4d978dbebfc5cd",
      "bytes": 1781
    },
    {
      "path": "doc/results/host-consumer-w23/source-audit.md",
      "sha256": "70607c07adeeb1d5ec57400d42a95146e3c0849c02a30bff12797a419532e9e8",
      "bytes": 6203
    },
    {
      "path": "doc/results/host-consumer-w23/source-files.json",
      "sha256": "5b79a4580dea8583a6723a7a7e05eb87121f5b01dc95d808b44b833084374e35",
      "bytes": 1301
    },
    {
      "path": "doc/results/host-consumer-w24/checkpoint.json",
      "sha256": "7bcd1e760211da8a8c5deb563397f4ff3f8c7e821de7128d792566a37768a27a",
      "bytes": 1681
    },
    {
      "path": "doc/results/host-consumer-w24/paper.json",
      "sha256": "1872c87876a80d12817af7efe56e3819ad0b52df2f17d59b4674fe925b6bd20d",
      "bytes": 935
    },
    {
      "path": "doc/results/host-consumer-w24/reading-note.md",
      "sha256": "a6c8a28347748c08ddf9db87a1c51a2fa8af4ccf4be197b613b5604e60743c4d",
      "bytes": 7006
    },
    {
      "path": "doc/results/host-consumer-w25/audit.md",
      "sha256": "c1c774648e260529389e2ea9ed180c8830840ad8ca2b35e8ccf2fa023fcaf10c",
      "bytes": 3582
    },
    {
      "path": "doc/results/host-consumer-w25/baseline-plan.json",
      "sha256": "e224a7a410bdd8ed3a65477f5ea3c2943bc63d87e58653f478f756d9439a1a89",
      "bytes": 1094
    },
    {
      "path": "doc/results/host-consumer-w25/baseline-results.json",
      "sha256": "53bf86fbaf25f8f92db336ad7d2d93aadbf310b601dce8e20c4417ed50941e68",
      "bytes": 862
    },
    {
      "path": "doc/results/host-consumer-w25/checkpoint.json",
      "sha256": "367a8cc0b43a853f2a6d3769c82ede1e63b5e139d327327d58fdfb0379b579f2",
      "bytes": 1777
    },
    {
      "path": "doc/results/host-consumer-w25/communication-tests.log",
      "sha256": "52dcf96a44602c58c2eaa92b653ff071c5053e050b14c787f19fdc1bbf80a7b9",
      "bytes": 1803
    },
    {
      "path": "doc/results/host-consumer-w25/reproduce_recheck.py",
      "sha256": "f415bde3420350d5abb5587a6b1683b63afcbcee3b5a16ace2df48e9e62bd20c",
      "bytes": 2208
    },
    {
      "path": "doc/results/host-consumer-w25/source-bindings.json",
      "sha256": "69b4bae5ddf3781fa2706984896c2b7b27e3b72082b8b122aaf35e1744f4ecdc",
      "bytes": 1223
    },
    {
      "path": "doc/task/task_details/HOST-COMM-01.md",
      "sha256": "83360c9cbf1135c02ebb87c33a2e7518ce26f8965cf7ac9a6a90546d3467d601",
      "bytes": 6131
    },
    {
      "path": "doc/task/task_details/HOST-COMM-02.md",
      "sha256": "971974b2f4bc4457425e0c3fc212336fe660674153700197048205faecaa5387",
      "bytes": 1291
    },
    {
      "path": "doc/task/task_details/HOST-COMM-03.md",
      "sha256": "ef76dcfb6f7ac80a7e7694b4da1f688a25a758164d6cce538b0a988484030b05",
      "bytes": 872
    },
    {
      "path": "doc/task/task_details/HOST-WRITE-TEMPLATE-01.md",
      "sha256": "e4262cc89a8259dd721697e7fad1ddd656d45815d5858a82885ce1e98f3e06a3",
      "bytes": 993
    },
    {
      "path": "CODEMAP.md",
      "sha256": "4ba88da685bee4357b4506616c505aa14c8d72ccbf3a1d6fddfaca4077a09aa3",
      "bytes": 17846
    },
    {
      "path": "doc/task/TASK.md",
      "sha256": "d9473678208f6956e88144fb4fd4768fca6a16b84302fe9034e04e2631cf3d78",
      "bytes": 30977
    },
    {
      "path": "docs/continuous_optimization/state.json",
      "sha256": "65257c91bbcfbb9b90009773b9ff1e75ba3db7c29d31c1c4f973dffb54cf090f",
      "bytes": 13631
    }
  ],
  "next": "Reconcile dirty recovery records with agent_doc namespace without rewriting frozen evidence; freeze resource-aware submit/poll contract and independent review before implementation",
  "latest_remote_observed": "acf85d76f9aa1bcf6c0e657b50a46a4f225c09a8",
  "remote_drift": "8 documentation files, runtime and workflow audited bytes unchanged; no evidence transferred to a release candidate"
}
```
</details>
