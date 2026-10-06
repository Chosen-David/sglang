# E104 RULER 32K 双语论文修订模板（槽位待六臂全齐后填；源 e104_ruler_32k.json）

## 已锁定判决（FullKV 锚点齐后可写）
1. **32K 全族退化非稀疏伤害**：fwe FullKV 7.0 < TLI 11.0/12.67；vt FullKV 31.0 < TLI 32.0/34.6；
   multikey_3 FullKV 4.0 最崩；niah_single 系全族全满证明数据管线健康
2. **主臂-参照臂差距长度稳定**：32K −2.15 vs 16K 及以下 −2.10；cwe −18.2 占 77%（γ 伤害不随长度放大）
3. **cwe 唯一真稀疏分化点**：FullKV 83.5（持平 16K 83.6）/ PSI-ref 80.1（差收窄 −5.0→−3.4）/ PSI-main 61.9 / Quest 24.5
4. **参照臂 32K 反超 FullKV**：60.78 vs 59.09 = +1.69
5. **Quest 第四崩塌点**：87.91(4K)→81.76(8K)→70.18(16K)→{QUEST_AVG}（崩在 multikey_2 2.0/multikey_3 0.0/cwe 24.5）

## 待填槽位
QUEST_AVG / TIA_AVG / C0_AVG（单池）/ QUEST_vt / TIA 各任务关键值

## CN 修订点（TLI_paper.tex）
1. **L17 摘要**：「$-$2.10 的差距随长度集中于 far-dense 词表任务 cwe」→ 改长度稳定口径 + 32K 全族退化注
2. **L214-225 tab:ruler**：加 32K 行（FullKV 59.09 / TIA {TIA_AVG} / 单池 {C0_AVG} / PSI 60.78 / Quest {QUEST_AVG}）+ 总 AVG 口径注（3 长度 vs 4 长度两种算法）
3. **L248 §4.2 三观察段**：加 32K 第四点判读（全族退化绝对值以同长度 FullKV 为基线；cwe 差距收窄；Quest 第四崩塌点）
4. **L368 Limitations**：「伤害集中于 far-dense 词表任务 cwe 随长度放大（82.0/68.6/55.5）」→ 改为长度稳定 + 32K 补点判决

## EN 修订点（TLI_paper_en.tex）：与 CN 逐处对应（L216 表 / 摘要 / §4.2 / Limitations）

## 收割 checklist
1. 等 E104B_RULER_32K_DONE → python -u exp/trace/analyze_e104_ruler_32k.py（六臂补全）
2. 填槽位 → 双语替换 → grep「随长度」旧口径残留
3. 编译双语 + pypdfium2 抽验
4. two-level commit（analyze 脚本 + e104_ruler_32k.json）+ sglang commit（tex）+ 双 push
5. TaskUpdate #111 completed + 记忆更新

## 已知口径坑
- 主表「总 AVG 88.71」是 3 长度均值；加 32K 行后总 AVG 算法变化（4 长度）或注明 32K 行单列不入总——**倾向：32K 行入表但总 AVG 保持 3 长度口径加脚注**（16K 及以下与 32K 档 n=100 同但 32K 是自生成数据，口径须注明）
- fwe 32K 全族混乱区（Quest 56.14 > TLI 12.67 > FullKV 7.0）：如实报不解释过度
- 32K 数据为官方生成器自产（niah_single 全满校验过），与 4K/8K/16K KVCache-Factory 官方预生成数据不同源——表 caption 须注明
