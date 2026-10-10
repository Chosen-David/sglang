# E116d：RULER 生产级正式打分入口（两阶段，fail-closed）
# 阶段一「冻结」：遍历 root/L*/pred{postfix}/，逐 (L, task, method) 格从
#   原始候选全集（排除 *-merged.jsonl 派生物）仲裁 best-file（行数最多，
#   并列取文件名时间戳最新）；
# 阶段二「评分」：以 --manifest --expect-tasks --merge-best 子进程调用
#   score_ruler.py，退出码非零 → 本脚本以同码失败（不吞 scorer 的门禁）。
#
# ===== E116e（GPT 1033 审计 030-036 七项修复，本版全落地）=====
#   030 min-samples 升格为发布硬门禁：任一 (L, method, task) 格 n <
#       min-samples → 非零退出、不发布任何产物、清理 staging；
#   031 staging 原子发布：全部校验（manifest + scorer + JSON/MD + receipt）
#       先在 {out}.staging-{run_id}/ 临时目录完成，成功后逐文件 os.replace
#       原子发布（receipt 最后落盘=提交信号）；失败写独立 failure receipt
#       （{out}.failure-{run_id}.json，明确「旧产物属于上一轮成功」），
#       不覆盖不删除旧成功产物，只清理未发布 staging；
#   032 身份扩展：manifest 逐格绑定 length（L 目录档位 + 行内实际 token 数
#       统计）、method、源数据文件 SHA256（{data_root}/{L}/{task}.jsonl）、
#       model_path、关键参数（--yarn/--yarn-factor/--extra-param）；
#       长度身份门禁 = ① 行 length ≤ L 档位（128K 数据混入 32K 目录拒）
#       ② 同 task 跨 method 逐行 length 一致（同 row index/answers 不同
#       length 拒）；legacy 事后行号身份标 identity_mode=legacy-partial，
#       原生 _id 数据标 native；
#   033 --out 强制小写 .json 后缀（无后缀/.JSON/路径中间含 .json 全拒），
#       MD 路径用后缀精确推导（不用 replace），并断言 JSON/MD/manifest/
#       receipt 四路径两两不同；
#   036 不改原始文件：原始 pred 文件全程只读；legacy 补刻写到 staging 派生
#       副本（成功后保留在 {out}.run-{run_id}/ 版本化派生目录）；receipt 记录
#       source_sha256 → derived_sha256 与逐字段不变量断言（pred/answers/
#       length/budget 逐位不变）；失败保留源文件原样。
# ===== E116f（GPT 1228 审计 038/039 两项 P1 修复，发布事务重构）=====
#   TL-RULER-PUBLISH-ATOMICITY-038（四个公开文件逐个 os.replace 不是跨
#       文件事务 + except 只捕 SystemExit，中途失败留下混合代际）→
#       重构为「generation 目录 + 单指针提交」协议：result/md/manifest/
#       receipt 四个规范文件连同派生副本全部留在同一不可变 generation
#       （{out}.run-{run_id}/，即原 staging 整目录）内，fsync 后用一次
#       原子 os.rename 落位；四个公开固定路径（{out}/{out 去后缀}.md/
#       {out}.manifest.json/{out}.receipt.json）降级为兼容镜像，逐个经
#       「在目标文件系统写临时文件 + 原子替换」安装（临时文件与目标同
#       目录必然同设备，跨设备 --manifest-out 的 EXDEV 路径天然不可达，
#       且安装前做 st_dev 防御断言）；receipt 镜像最后落盘 = 唯一提交
#       信号；发布段任一步 OSError → 从备份回滚已替换镜像（旧产物逐位
#       还原；首轮无旧产物则删除新镜像）+ failure receipt + 清理
#       generation——「旧 receipt 配新结果」的混合代际状态不可达；
#   TL-RULER-DERIVED-COMMIT-039（success receipt 公开早于 derived_dir
#       存在）→ generation rename 前置于一切公开发布，receipt 声明的
#       derived_dir 在 receipt 发布前必须已存在并通过四规范文件 SHA
#       与 receipt 声明值的逐位校验；
#   except 范围扩大为 (SystemExit, OSError)；
#   receipt 新增 publish_protocol="e116f-generation-v2" 版本字段与
#   outputs.generation_files（消费者可从 receipt 单指针解析同一
#   generation 内的全部文件）；旧格式四件套（32K/64K 已发布产物）仍
#   按原路径直接可读，读取兼容不受影响。
# ===== E116g（GPT 1326 审计 040 修复，同 --out 并发发布互斥）=====
#   TL-RULER-CONCURRENT-PUBLISH-040（两个并发发布者对同一 --out 的
#       备份/四镜像安装/清理可交错，留下固定 JSON 与 receipt 的持久
#       混合代际且两进程均返回 0；run_id 只隔离 staging/generation，
#       不隔离固定 aliases）→ 发布事务整体移入进程间独占锁：以
#       realpath(abspath(--out)) 为键建立 fcntl.flock(LOCK_EX) 排他锁
#       （锁文件 {out}.lock 与目标同目录），锁覆盖「读取/建立备份 →
#       四镜像安装 → SHA 终验 → 备份清理或回滚」全段，备份必须在获得
#       锁后才创建；安装后新增 SHA 终验（镜像 ↔ generation 规范文件
#       逐位一致，锁内失败走锁内回滚）。锁语义边界如实声明：flock 为
#       咨询锁，仅约束同样走本入口的发布者；进程退出（含 SIGKILL）时
#       由内核自动释放，无陈旧锁需人工清理；仅验证过本机文件系统，
#       跨宿主共享文件系统（NFS 等）的 flock 语义未验证，不构成跨宿主
#       互斥承诺。
# ===== E116h（GPT 1429 审计 042/043/044 三项修复）=====
#   TL-RULER-LOCK-SYMLINK-042（锁键 realpath(out) 在 out 初始为 symlink 时
#       首次 replace 后漂移，第二发布者取得另一把锁）→ ①四条目标
#       （out/MD/manifest/receipt）任一为 symlink 即 fail-closed（锁内
#       lstat 复核二次）；②锁键改为 realpath(父目录) + lexical basename
#       ——最终分量会被 replace 改写，只解析父目录，键不随发布漂移；
#   TL-RULER-LOCK-WRITESET-043（锁只按 out，--manifest-out 可独立共享
#       → 不同 out 的两次发布并发覆盖同一 manifest 且均返回成功）→
#       完整写集锁：out/md/manifest/receipt 四条目标各取一把锁，全局
#       按锁键排序有序获取、逆序释放（任意两个发布写集只要共享任一
#       目标即共享该目标的锁 → 串行化；统一排序序获取避免死锁）；
#       写集内两条目标解析到同一锁键（symlink 父目录别名）→ fail-closed；
#   TL-RULER-ROLLBACK-RECOVERY-044（备份清理在 committed=True 之前，
#       清理 OSError 触发锁外部分回滚删除仅存备份 → 混合代际不可恢复）→
#       时序重排：四镜像安装 + SHA 终验完成后先在锁内置 committed=True
#       再做备份清理；清理降级为提交后 best-effort GC（失败只记
#       rollback_state["gc_pending"] + stderr warning，不回滚、不影响
#       提交语义；进程后续若落 failure receipt 则 gc_pending 一并记录）。
# ===== #194（GPT 2026-10-10 0330 审计 TL-E119-YARN-IDENTITY-057）=====
#   run_identity.yarn_factor 此前无条件等于操作者事后 CLI 声明——E119
#   128K 批次派单只传 --yarn 未传显式 factor（生成进程按
#   YARN_FACTOR_AUTO[131072]=4.0 自动档路径解析），三份正式 manifest 却
#   声明 2.0，effective factor 未闭合。修复（producer-yarn-config-v1）：
#   ① 生成侧 pred_ruler.py 每次运行把 effective 配置原子写入预测产物旁挂
#      receipt（{pred 基名}-yarn_receipt.json，见 yarn_receipt.py）；
#   ② 本入口 freeze_and_stage 对每个 best-file 探查旁挂 receipt：全部
#      格子有证据 → run_identity.yarn/yarn_factor 取生产者 effective 值
#      （yarn_factor_provenance="producer_receipt"，receipt path+SHA 逐格
#      冻结进 manifest cells 与 producer_evidence）；操作者 CLI 声明与
#      生产者证据冲突 → fail-closed（_fail 走 SystemExit，python -O 不
#      失效）；全部格子无证据（legacy）→ CLI 值如实降级
#      yarn_factor_provenance="operator_declared"（实际生效值未闭合，
#      不冒充 effective 值）；部分有部分无 → fail-closed（单次
#      run_identity 不得混合两种口径）；receipt 存在但半写/损坏/自相矛盾
#      /档位不符 → fail-closed（存在即证据）；
#   ③ 既有 128K 三臂 manifest 字节不动，旁挂版本化 correction JSON
#      （.manifest.yarn_correction.json）把 2.0 降级为 operator_declared
#      并记录「E190 启动链佐证强支持 4.0 自动档路径、生产者原生证据缺失、
#      effective factor 未闭合」——不脑补 actual=4.0（佐证≠同代哈希绑定，
#      与 051 同纪律）。64K 无冲突（auto 65536=2.0 与声明一致），不动。
# ===== E116j（GPT 2026-10-10 0125 审计 TL-E119-GEN-CONTENT-BINDING-052）=====
#   generation 的 md 与 generation receipt 此前只被消费者做「存在性检查」
#   （v2 receipt 只声明 result/manifest 的 SHA）→ 两者发布后被篡改不可
#   检出。修复：协议升级为 e116i-generation-v3——generation rename 冻结
#   后、任何公开镜像安装前，对 generation 内 result.md 与 receipt.json
#   计算 SHA256，写入「entry receipt」（gen_md_sha256 / gen_receipt_sha256）
#   并以 entry receipt 作为公开 {out}.receipt.json 镜像的安装源。receipt
#   不自哈希（避免自引用）：entry receipt 与 generation 内 receipt.json
#   是两个不同文件，前者哈希后者。既有 64K/128K 生产 receipt（v2/legacy）
#   一律不动，消费者按协议版本分流（见两个 analyzer 的 ⑨c）。
# ===== #196（GPT 2026-10-10 0635 二轮复审 062/063/064/065）=====
#   TL-E119-YARN-SNAPSHOT-RACE-062（P1）：staging 快照与源目录验证的
#       时序竞态——freeze_and_stage 先把源预测复制到 staging 派生副本
#       （best-file 仲裁用 staging），之后 _load_producer_yarn_receipt
#       才重新打开【源路径】验证当前源文件与回执，且无「staging SHA ==
#       receipt.prediction_sha256」断言。生产者在复制后、验证前原子提交
#       B 代 → formal 评 staging A、manifest 标 B 的 source SHA 与回执
#       verified_same_generation=true（GPT CPU 复现实锤）。修复（GPT 方案
#       2，无锁 bytes 快照语义重排，选择见 _load_producer_yarn_receipt
#       docstring）：①回执读成单一 bytes 快照（解析与 receipt SHA 同源，
#       防读-算之间被替换）；②以 staging 副本（评分对象）的 SHA/行数对
#       回执 prediction_sha256/prediction_lines 校验（失配 fail-closed）；
#       ③三方一致（staging == receipt == 源当前字节，源失配重读一次仍
#       失配则 fail-closed 拒绝并重试）才 verified_same_generation=true；
#       ④源路径之后被生产者推进不改变已冻结 generation 的身份——身份
#       断言绑定 staging bytes，manifest 的 source_sha256 取冻结窗口内
#       三方一致时刻的值，不再事后重算（事后重算会捡到新一代造成
#       manifest 自相矛盾）。
#   TL-E119-YARN-CONFIG-PARTIAL-064①（P2）：v2 schema 允许
#       model_config_sha256=None（远端模型 ID / config.json 不在
#       model_path 时生产者写 None）→「完整模型配置闭包」不成立。
#       修复：config_fingerprint 如实标注 config_identity=missing +
#       producer_evidence.model_config_closure 降级标注（不强行做
#       config.to_dict() 哈希——引入 transformers 违反 yarn_receipt
#       零重依赖设计原则，选诚实降级方案）。064②（max_num 入指纹）在
#       yarn_receipt.effective_config_sha256 落地。
#   063（纠偏 sidecar 严格 schema）与 065（测试 oracle 显式化）分别在
#       yarn_receipt.py 与测试套件落地。
# ===== #195（GPT 2026-10-10 0428 审计 059/060/061）=====
#   TL-E119-YARN-RECEIPT-BINDING-059（P1）：_load_producer_yarn_receipt
#       此前按 basename 找同名回执、分别记 SHA，不做同代交叉验证——
#       「A 进程的预测配 B 进程的回执」可达（同名重跑/中断重跑/并发/
#       事后改写）。修复：v2 回执（producer-yarn-config-v2，生成侧见
#       pred_ruler.py）校验 status=complete + prediction_basename/
#       SHA256/行数与 best-file 现算逐位一致，不一致/缺完成标记 →
#       _fail；v1 回执降级 provenance=producer_receipt_v1_partial（只证
#       factor 口径不证同代）；v1/v2 混装 → _fail（不混合证据代际）；
#   TL-E119-YARN-RECEIPT-CLOSURE-060（P2）：validate_producer_receipt
#       只校验开关与 factor——beta_fast/seed/模型 config hash/生产脚本
#       hash 写入零校验。修复：严格 schema（yarn_receipt.py）+ 摘要保留
#       完整配置指纹 + _check_producer_config_consistency 跨格一致性
#       门禁（按 context_length 分组，同档内 model/rope/seed/脚本 hash
#       经 effective_config_sha256 逐位一致；task/method/t/pred_postfix
#       为分组自由字段）；
#   TL-E119-YARN-CORRECTION-DISCOVERY-061（P2）：三份 128K
#       .manifest.yarn_correction.json 的机器消费入口
#       yarn_receipt.resolve_manifest_yarn_identity（同路径旁挂纠偏 →
#       旧 yarn_factor 不再当 effective，返回 operator_declared_
#       not_effective + null + 纠偏绑定哈希；target hash 失配 fail
#       closed）；正式消费接线在 analyze_e119_ruler128k_formal.py。
# ===== #197（GPT 2026-10-10 0834 审计 TL-E119-YARN-SAME-BYTES-PROVENANCE-066）=====
#   062 的三方 SHA 一致只证「内容等价」不证「运行同代」：两个配置不同
#   的运行产生【字节完全相同】的预测 JSONL 时（GPT 两份独立复现：直接
#   调 build/write/_load 三函数与走正式 stage→commit 生产提交函数，均
#   得到 staging=A 配 run-B/seed=99/max_num=100 回执且 verified_same_
#   generation=true），「复制到 staging 后、读回执前」的 B 代两步提交
#   穿插三方校验全过——manifest 的 run_id/seed/max_num/model 归属错挂
#   （provenance 破坏；评分数值不变，故 P2；据此声称 treatment 因果闭包
#   则升 P1）。修复（GPT 方案 2 共享锁）：freeze_and_stage 对每个
#   (task, method) 格的完整冻结窗口（候选复制 → best-file 仲裁 → 回执
#   bytes 快照 → 三方校验）全程持有与生产者同键的 flock
#   （yarn_receipt.acquire_output_lock，056/059 realpath(父目录)+basename
#   键口径；候选多键按锁键排序获取、逆序释放，043 写集锁同款防死锁纪律）
#   ——生产者的生成→提交生命周期同样持锁（pred_ruler.py），窗口内两步
#   os.replace 提交不可穿插，run_id/config 归属闭合；锁释放后的新一代
#   提交不改变已冻结 generation 的身份（062④）。062 三方 SHA 校验保留
#   为纵深防御：锁防【活进程】穿插，SHA 防【死亡中间态】（crash-between
#   失配代际仍由三方校验拒收）。锁不可用/获取失败 → fail-closed 不静默
#   降级（同代冻结窗口无锁即不可闭合；只读 root 须先复制到可写位置）；
#   acquire 等待时长逐格记入 manifest cells freeze_lock 轻量审计字段
#   （长时间持锁对补跑吞吐的影响据此可审计）。
# 用法（单臂单 root）：
#   python -u benchmark/RULER/score_ruler_formal.py \
#     --root exp/results_ruler/e109_full_Qwen3-8B/L32768 \
#     --pred-postfix _E109_FULLKV \
#     --data-root /home/wangyuanshuo02/sparse-bench/third_party/KVCache-Factory/data/RULER \
#     --out /tmp/ruler_formal_fullkv.json \
#     [--manifest-out ...] [--expect-tasks 11] [--min-samples 100] \
#     [--model-path ~/Qwen3-8B] [--yarn] [--extra-param k=v ...]
# 三臂各跑一次（postfix 分别为 _E109_mavg_a0.25_b0.125_g0.625 /
#   _E109_aavg_a0_b0_g0 / _E109_FULLKV）。
# 发布产物（成功时）：{out}（结果 JSON）、{out 去后缀}.md（表格）、
#   {out}.manifest.json（冻结身份 manifest）、{out}.receipt.json（entry
#   receipt=提交信号，含 gen_md_sha256/gen_receipt_sha256 内容绑定）、
#   {out}.run-{run_id}/（版本化派生目录：补刻副本 + merged 规范文件 +
#   scorer manifest + generation receipt.json + entry_receipt.json）。
# 失败产物：{out}.failure-{run_id}.json（独立 failure receipt，不动旧产物）。
import argparse
import fcntl
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from benchmark.RULER.score_ruler import (  # noqa: E402
    TASKS, _answers_sha16, _file_sha256, _nlines, _ts_of,
    _validate_manifest_schema,
)
# 057：生产者原生 yarn receipt 探查/自洽校验（协议与生成侧共享同一口径）
# 059/060（#195）：v2 同代绑定校验 + 严格 schema + 跨格配置一致性门禁
# 066（#197）：acquire/release_output_lock——freeze 冻结窗口与生产者
#   同键互斥（056/059 口径 realpath 输出路径锁原语，见 yarn_receipt.py）
# 066/crash-recovery（#198）：resolve_generation_pointer——指针协议产物
#   从 {pred}.tli_gen 解析不可变 generation（预测与回执同目录同源），
#   混合代不可达；无指针 → legacy-direct 直接读路径（既有产物零改动）
from benchmark.RULER.yarn_receipt import (  # noqa: E402
    GENERATION_BINDING_LEGACY, GENERATION_BINDING_POINTER,
    GENERATION_POINTER_SUFFIX, RECEIPT_V1_VERSION, RECEIPT_VERSION,
    acquire_output_lock, effective_config_sha256,
    generation_pointer_path, producer_receipt_path_for,
    release_output_lock, resolve_generation_pointer,
    validate_committed_generation,
)

FORMAL_PATH = os.path.abspath(__file__)
SCORER_PATH = os.path.join(os.path.dirname(FORMAL_PATH), "score_ruler.py")
# 036 逐字段不变量：legacy 补刻只允许新增 _id/_answers_sha 两键，
# 这四个业务字段必须逐位不变
INVARIANT_FIELDS = ("pred", "answers", "length", "budget")


def _fail(msg):
    raise SystemExit(f"[GATE-FAIL] {msg}")


# ---- E116f：generation + 单指针发布协议辅助 ----

def _fsync_file(path):
    """best-effort 文件落盘（发布协议的持久化边界；平台不支持时静默）"""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _fsync_dir(path):
    """best-effort 目录项落盘（保证 rename/replace 后目录项可见）"""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _backup_public(dst, run_id):
    """E116f（038）：替换公开镜像前先备份旧代际。
    优先硬链接（同目录必然同设备、零拷贝、保旧 inode），不支持时回退
    拷贝。旧产物不存在（首轮）→ 返回 None（回滚语义=删除新镜像）。
    备份路径 {dst}.bak-{run_id} 与目标同目录 → 回滚 os.replace 原子。"""
    if not os.path.exists(dst):
        return None
    bak = f"{dst}.bak-{run_id}"
    try:
        os.link(dst, bak)
    except OSError:
        shutil.copyfile(dst, bak)
    return bak


def _atomic_install(src, dst, run_id):
    """E116f（038）：把 generation 内规范文件安装到公开固定路径。
    直接 os.replace(src→dst) 在 --manifest-out 指向另一挂载点时抛
    EXDEV（审计 038 的复现路径）——统一改为「在目标文件系统写临时
    文件 + 原子替换」：临时文件与目标同目录（必然同设备），安装前做
    st_dev 防御断言（fail-closed），替换瞬间原子。失败时自清理临时
    文件后向上抛 OSError（由发布段统一回滚）。"""
    d = os.path.dirname(dst)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{dst}.tmp-{run_id}"
    try:
        shutil.copyfile(src, tmp)
        _fsync_file(tmp)
        # st_dev 防御校验（审计建议 2：跨设备须走目标文件系统临时文件；
        # tmp 与 dst 同目录理论上必同设备——不等则 fail closed 不安装）
        if os.stat(tmp).st_dev != os.stat(d or ".").st_dev:
            raise OSError(f"临时文件 {tmp} 与目标 {dst} 跨设备"
                          f"（st_dev 不等，理论不可达）——fail closed")
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _lock_file_for(target):
    """E116h（042/043）：单目标的稳定锁键 = realpath(父目录) + lexical
    basename + ".lock"。最终分量（文件名）会被发布的 os.replace 改写，
    realpath(整个路径) 在「目标初始是 symlink / 不存在」与「发布后是
    普通文件」两种状态之间会漂移（042 的锁键漂移根因），故只解析父
    目录——symlink 父目录的别名路径（alias/out.json 与 real/out.json）
    归一到同一把锁，而文件名本身不受任何发布操作影响。"""
    return os.path.join(
        os.path.realpath(os.path.dirname(os.path.abspath(target)) or "."),
        os.path.basename(target)) + ".lock"


def _locked_publish(mirrors, run_id, out_path, backups, published,
                    rollback_state):
    """E116g（040）+ E116h（042/043/044）：完整写集的并发发布互斥。

    锁协议（040 起覆盖、042/043 修正键与写集）：
      - 锁覆盖完整发布事务——「读取/建立备份 → 四镜像安装 → SHA 终验
        → 提交置位 → 备份 GC」；备份必须在获得锁之后才创建（审计建议
        1：备份与安装不可分割）；
      - 042：四条目标（out/MD/manifest/receipt）任一为符号链接 →
        fail-closed（获锁前拒绝一次，获锁后 lstat 复核一次）；锁键只
        解析父目录（见 _lock_file_for），不随首次 replace 漂移；
      - 043：完整写集锁——每条目标各取一把 fcntl.flock LOCK_EX 排他
        锁（阻塞等待），全局按锁键排序有序获取、逆序释放。任意两个
        发布写集只要共享任一目标（同 --out、不同 --out 共享
        --manifest-out、symlink 父目录别名）即共享该目标的锁 → 串行
        化；统一排序序获取避免死锁。写集内两条目标解析到同一锁键 →
        别名冲突，fail-closed；
      - 044：四镜像安装 + SHA 终验成功即事务提交（committed=True 在
        锁内、备份清理之前置位）；备份清理是提交后的 best-effort GC，
        失败只记 gc_pending + stderr warning，不回滚已提交代际。

    安装后 SHA 终验：每个已安装镜像与 generation 规范文件逐位一致；
    终验失败 → 锁内回滚（旧产物逐位还原，首轮则删除新镜像；回滚二次
    失败的镜像记入 rollback_state["errors"] 并使整体以非零退出 loudly
    失败——不静默声称成功）后向上抛出，由外层 except 统一写 failure
    receipt。

    锁语义边界（如实声明，不冒充跨宿主保障）：
      - flock 是咨询锁，仅约束同样走本入口的发布者；
      - 锁在进程退出（含被 SIGKILL）时由内核自动释放——不存在需要
        人工清理的陈旧锁（.lock 文件本身可长期留存，无锁持有状态）；
      - 仅验证过本机文件系统的 flock 语义；跨宿主共享文件系统
        （NFS 等）的 flock 语义有坑且未验证，本锁不构成跨宿主互斥
        承诺（多机发布到共享路径须另行协调）。
    """
    # ---- 042：发布前 symlink 拒绝（任何一条目标是符号链接都不允许
    #      发布——锁键 canonicalization 依赖「最终分量不是 symlink」，
    #      且对 symlink 的 replace 语义本身有歧义）----
    for _, dst in mirrors:
        if os.path.islink(dst):
            _fail(f"发布目标 {dst} 是符号链接——fail closed（042：symlink "
                  f"目标会使锁键漂移且替换语义歧义；请发布到普通文件路径）")
    # ---- 043：完整写集锁键（out/md/manifest/receipt 各一把，去重排序）----
    lock_paths = sorted({_lock_file_for(dst) for _, dst in mirrors})
    if len(lock_paths) != len(mirrors):
        _fail(f"发布写集存在路径别名：{len(mirrors)} 条目标解析到 "
              f"{len(lock_paths)} 把锁键（symlink 父目录/路径别名指向同一"
              f"物理文件）——两条镜像会写同一目标，fail closed（043）")
    fds = []
    try:
        for lp in lock_paths:
            d = os.path.dirname(lp)
            if d:
                os.makedirs(d, exist_ok=True)
            fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
            fds.append(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)   # 阻塞等待；内核级自动释放
        # ---- 042：获锁后 lstat 复核（锁外窗口内外部实体可能已把目标
        #      换成 symlink）----
        for _, dst in mirrors:
            if os.path.islink(dst):
                _fail(f"获锁后复核发现发布目标 {dst} 是符号链接——"
                      f"fail closed（042）")
        # ---- 备份必须在获得锁后创建（040）----
        for _, dst in mirrors:
            backups[dst] = _backup_public(dst, run_id)
        try:
            for src, dst in mirrors:
                _atomic_install(src, dst, run_id)
                published.append(dst)
            # ---- SHA 终验（锁内）：安装后的镜像与 generation 规范
            #      文件逐位一致；失败走锁内回滚 ----
            for src, dst in mirrors:
                if _file_sha256(dst) != _file_sha256(src):
                    raise OSError(
                        f"镜像 {dst} 安装后 SHA 终验失败（与 generation "
                        f"规范文件 {src} 不一致）——锁内回滚")
        except BaseException:
            # ---- 锁内回滚（040：失败回滚不得与并发发布者交错）----
            errors = []
            for dst in published:
                bak = backups.get(dst)
                try:
                    if bak is None:
                        try:
                            os.remove(dst)
                        except FileNotFoundError:
                            pass
                    else:
                        os.replace(bak, dst)
                except OSError as oe:
                    errors.append(f"{dst}: {oe}")
            rollback_state.update(
                done=True, n_restored=len(published) - len(errors),
                errors=errors)
            raise
        # ---- 044：四镜像安装 + SHA 终验成功 → 事务即提交。committed
        #      在锁内、备份清理之前置位——备份删除属提交后的垃圾回收，
        #      其失败不得触发回滚（旧顺序：清理失败 → committed 仍
        #      False → 锁外部分回滚删除仅存备份 → 混合代际不可恢复）。
        #      同时该标记使锁释放后 main() 的报告型 I/O 失败（stdout
        #      broken pipe 等）不会被误判为发布失败（kimi3 1404）----
        rollback_state["committed"] = True
        # ---- 备份清理 = 提交后 best-effort GC（044）：失败只记
        #      gc_pending + stderr warning，不回滚、不影响提交语义 ----
        gc_pending = []
        for dst, bak in backups.items():
            if bak is not None:
                try:
                    os.remove(bak)
                except FileNotFoundError:
                    pass
                except OSError as oe:
                    gc_pending.append(f"{bak}: {oe}")
        if gc_pending:
            rollback_state["gc_pending"] = gc_pending
            print(f"[formal] WARN 备份清理失败（发布已提交，残留待 GC）: "
                  f"{gc_pending}", file=sys.stderr)
    finally:
        # ---- 043：逆序释放全部写集锁 ----
        for fd in reversed(fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


def _stamp_or_copy(src, dst, task, stamp):
    """036：legacy 补刻写到派生副本 dst（源文件 src 全程只读）。
    返回 (stamped, rows)。stamped=True 表示发生了补刻（dst 含新增
    _id/_answers_sha 两字段）。写后重读做逐字段不变量断言。
    混合状态（部分行有 _id）→ fail closed（score_ruler 门禁 1 同款）。"""
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    if not rows:
        _fail(f"{src}: 空预测文件——fail closed")
    have = sum(1 for r in rows if r.get("_id") is not None)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if have == len(rows):
        shutil.copyfile(src, dst)   # native：原样拷贝
        return False, rows
    if have > 0:
        _fail(f"{src}: 部分行有 _id（{have}/{len(rows)}）——混合状态非法，"
              f"fail closed")
    if not stamp:
        _fail(f"{src}: legacy 数据缺 _id 且 --no-stamp-legacy-ids——无法做"
              f"manifest 校验，fail closed")
    out_rows = []
    for i, r in enumerate(rows):
        r2 = dict(r)
        # 与 pred_ruler.py E116c 原生口径一致：_id = {task}:{row_index}
        r2["_id"] = f"{task}:{i}"
        r2.setdefault("_answers_sha", _answers_sha16(r["answers"]))
        out_rows.append(r2)
    with open(dst, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # 逐字段不变量断言（写后重读，防转换脚本自身破坏业务字段）
    dst_rows = [json.loads(l) for l in open(dst, encoding="utf-8")]
    if len(dst_rows) != len(rows):
        _fail(f"{src} → {dst}: 补刻后行数变化（{len(rows)}→{len(dst_rows)}）"
              f"——不变量破坏，fail closed")
    for a, b in zip(rows, dst_rows):
        for k in INVARIANT_FIELDS:
            if a.get(k) != b.get(k):
                _fail(f"{src} → {dst}: 补刻后字段 {k} 发生变化"
                      f"（{a.get(k)!r} → {b.get(k)!r}）——不变量破坏，"
                      f"fail closed")
    return True, out_rows


def _load_producer_yarn_receipt(staged_pred, source_pred, task, Lnum,
                                 source_final_path=None):
    """057+059/060/062+066/crash-recovery：探查 best-file 旁挂生产者
    receipt，做自洽 + 同代绑定 + 严格 schema + staging 快照三方一致
    校验。

    参数（062 重排 + 066/crash-recovery 扩展）：staged_pred = staging
    派生副本路径（评分对象，freeze_and_stage 已复制完成）；source_pred
    = 实际被复制进 staging 的源文件路径（指针协议下位于 gen 目录内，
    basename 与 staging 副本相同）；source_final_path = 指针探查基准
    （best-file 最终路径 {task}-*.jsonl；None = 退回 source_pred——
    直查/legacy 场景两者本就同一路径）。

    返回 receipt 摘要 dict；文件不存在 → None（legacy，如实标 missing）。
    存在但损坏/自相矛盾/协议不符/档位不符 → _fail（存在即证据：半写或
    被篡改的 receipt 比缺失更危险，fail-closed 拒收整次发布）。

    059（producer-yarn-config-v2）同代绑定校验——修复前 formal 分别
    冻结「当前预测 SHA」与「当前回执 SHA」却不验证二者同代，「A 进程
    的预测配 B 进程的回执」可达：
      - v2：status 必须 complete（无完成标记=中断代际拒收）；
        prediction_sha256 / prediction_lines 与预测字节逐位比对
        （不一致 → fail closed——预测与回执不同代）；
      - v1（057 时代协议）无绑定字段 → 摘要 prediction_binding=None，
        provenance 降级 producer_receipt_v1_partial（只证 factor 口径
        不证同代），不冒充完整绑定证据。

    062（TL-E119-YARN-SNAPSHOT-RACE，GPT 方案 2 无锁 bytes 快照）：
    修复前先把源预测复制到 staging、best-file 仲裁用 staging 副本，
    本函数却重新打开【源路径】验证当前源文件与回执，且 manifest 分别
    现算 source/staging SHA 互不比对——生产者在复制后、验证前原子提交
    B 代 → formal 评 staging A、manifest 标 B 的 source SHA 与回执
    verified=true（GPT CPU 复现：staged_derived_sha256 !=
    receipt_prediction_sha256 但 freeze 成功返回）。修复语义重排：
      ① 回执读成单一 bytes 快照——解析与 receipt SHA 都基于同一 bytes
         （防「读-算之间被生产者 os.replace 推进」造成内容 A 配
         SHA B 的混合态回执摘要）；
      ② 同代绑定比对对象改为 staging 副本（评分对象）——staged SHA/
         行数 对回执 prediction_sha256/prediction_lines 校验，失配
         fail-closed（覆盖「复制完成后、回执读取前生产者提交 B 代」
         的 barrier 窗口：staging=A、receipt=B → 拒绝）；
      ③ 三方一致才 verified_same_generation=true：staging SHA ==
         回执 SHA == 源当前字节 SHA；源失配重读一次仍失配 →
         fail-closed（源在冻结窗口内被推进到新一代，拒绝并重试——
         不发布 staging=A/receipt=B 的混合代际，也不发布自洽但
         source_sha256 指向新一代的 manifest）；
      ④ 冻结完成后源路径再被生产者推进不改变已冻结 generation 的
         身份——身份断言绑定 staging bytes；manifest 的 source_sha256
         取本函数三方一致时刻的值（调用方不得事后重算源 SHA，事后
         重算会捡到新一代造成 manifest 自相矛盾）。
    快照顺序的设计选择：保留「先复制（freeze_and_stage 候选循环）→
    后读回执」的既有流程（不重排为先读回执再复制）——两种顺序下所有
    混合代际态都会被 ②/③ 拒收，区别仅在「提交落在回执读取之前还是
    之后」各对应 ② 或 ③ 触发；保留既有流程使改动面最小、且 staging
    复制循环无需感知回执。

    060 新增：摘要保留完整配置指纹（model_path/model_config_sha256/
    完整 rope_scaling/native_mpe/seed/max_gen/max_num/生产脚本/
    effective_config_sha256），供 _check_producer_config_consistency
    跨格门禁比较。

    064① 新增：model_config_sha256=None（远端模型 ID / config.json
    不在 model_path）→ config_fingerprint.config_identity=missing 如实
    降级——「完整模型配置闭包」不成立，不冒充（producer_evidence.
    model_config_closure 由 _resolve_yarn_identity 汇总降级标注）。

    066（TL-E119-YARN-SAME-BYTES-PROVENANCE）锁上下文声明：本函数的
    回执 bytes 快照与三方校验运行在调用方 freeze_and_stage 持有的【与
    生产者同键输出路径锁】窗口内——活进程的两步提交穿插（同字节不同
    配置的 B 代换代）不可达，锁窗口内读取的 receipt 与源/staging 属同
    一代际；三方 SHA 校验仍保留为纵深防御（锁防活进程穿插，SHA 防死亡
    中间态——crash-between 留下的「新预测配旧回执」失配代际不经过锁，
    由 ②③ 拒收）。单独调用本函数（无调用方锁）时只有内容等价语义，
    不得据此宣称运行同代（测试直查场景 staged==source 同文件退化）。

    066/crash-recovery（#198）：上述 066 锁上下文声明成立的前提是
    059 的【两步提交】——但两步之间死亡且 B 与 A 字节完全相同时，
    盘上留下「B 物理写入的预测 + A 旧回执」，三方 SHA 全等，②③ 全过，
    SHA 校验只能拒绝内容失配的坏态、不能证明运行同代。指针协议下
    （yarn_receipt.resolve_generation_pointer）：生产者把预测与完成
    回执同置不可变 generation 目录 {out}.gen-{attempt_id}/，提交 =
    单次原子切指针 {out}.tli_gen。本函数先以 best-file 最终路径解析
    指针：
      - 有指针 → generation_binding="pointer-v1"：source_pred 必须
        位于指针所指 gen 目录内（否则指针在发现与冻结之间被切换，
        fail-closed），回执取 gen 目录内与预测同源的那份——物理来源
        = 指针所指目录，混合代不可达；
      - 无指针 → generation_binding="legacy-direct"（既有产物零改动，
        维持 062①②③ 的锁 + SHA 双层兜底口径）。
    两种绑定下 ②③ 三方一致校验保留为纵深防御（防同代文件被事后改写）。

    070（TL-E119-PROBE-RECEIPT-VALIDATION，#200）：回执 bytes 快照（062①）
    + JSON 解析 + validate_producer_receipt（057/059/060）+ v2 staged 字节
    绑定（062②）整体重构为调用 yarn_receipt.validate_committed_generation
    共享校验器——完成探针 gen_completion_probe.py 与本正式入口单口径，
    消灭「调度判 complete/SKIP 但正式评分 fail-closed 拒收」的完成定义
    分裂；062③ 三方一致与 v1 降级分支保留在本函数（formal 专属语义）。"""
    # ---- 066/crash-recovery：先以 best-file 最终路径解析 generation 指针 ----
    # 有指针 = pointer-v1（预测与回执同置指针所指 gen 目录，物理来源同源，
    # 混合代不可达）；无指针 = legacy-direct（既有产物按旁挂约定路径直接读，
    # 059 v2 及更早协议——062①②③ 锁 + SHA 双层兜底口径不变）。
    final_path = (source_final_path if source_final_path is not None
                  else source_pred)
    gen_info = resolve_generation_pointer(final_path)
    if gen_info is None:
        binding_mode = GENERATION_BINDING_LEGACY
        rcp_path = producer_receipt_path_for(source_pred)
    else:
        binding_mode = gen_info["binding"]  # = GENERATION_BINDING_POINTER
        # 指针发现与冻结窗口之间的换代由调用方锁 + ②③ 三方校验兜底；
        # source_pred 不在指针所指 gen 目录内 = 指针已被切换或 source
        # 非 gen 代成员 —— fail-closed，不发布混合代际
        if os.path.abspath(os.path.dirname(os.path.abspath(source_pred))) \
                != os.path.abspath(gen_info["gen_dir"]):
            _fail(f"{gen_info['pointer_path']}: staging 源 {source_pred} 不在"
                  f"指针所指 generation 目录 {gen_info['gen_dir']} 内——"
                  f"指针在解析与冻结之间被切换，或源非该 generation 成员，"
                  f"fail closed（066/crash-recovery）")
        rcp_path = gen_info["rcp_path"]
    if not os.path.isfile(rcp_path):
        return None
    # ---- 070：回执内容 + v2 同代字节绑定统一走共享校验器 ----
    # yarn_receipt.validate_committed_generation（probe 与 formal 单口径，
    # 防第三套回执判断漂移）。绑定比对对象 = staging 评分副本（062②
    # 语义逐位保持）；共享校验器内部承载 062① bytes 快照、057/059/060
    # schema 与格式层、v2 字节级同代绑定（staged SHA/行数 vs 回执声明）。
    # v1 无绑定字段 → 共享校验器跳过字节比对，与本函数下方 v1 降级
    # 分支同口径（producer_receipt_v1_partial，只证 factor 口径）。
    v = validate_committed_generation(staged_pred, rcp_path)
    rcp = v["receipt"]
    rcp_sha = v["rcp_sha256"]
    if rcp["context_length"] != Lnum:
        _fail(f"{rcp_path}: receipt.context_length={rcp['context_length']} "
              f"与所在目录档位 L{Lnum} 不一致——生产者证据与数据档位"
              f"冲突，fail closed（057）")
    version = rcp["receipt_version"]
    binding = None
    # 062③：source_sha256 在三方一致窗口内取值（调用方复用，不重算）
    source_sha = None
    if version == RECEIPT_VERSION:
        # 062② 的 staged SHA/行数比对已由共享校验器完成（失败即
        # SystemExit fail-closed，不再重复计算——staged_sha 直接取共享
        # 校验器对 staging 副本的字节快照结果；行数一致性同验）
        staged_sha = v["binding_sha256"]
        # ---- 062③：三方一致（staging == 回执 == 源当前字节）----
        # 源失配重读一次（防瞬时读异常误杀）；仍失配 = 源在冻结窗口内
        # 被推进到新一代 → fail-closed 拒绝并重试，不发布混合代际
        source_sha = _file_sha256(source_pred)
        if source_sha != staged_sha:
            source_sha = _file_sha256(source_pred)
            if source_sha != staged_sha:
                _fail(f"{rcp_path}: 源预测 {source_pred} 当前字节 SHA256="
                      f"{source_sha} 与已冻结 staging 评分副本 SHA256="
                      f"{staged_sha}（=回执声明）不一致——源在冻结窗口内"
                      f"被生产者推进到新一代（062 快照竞态），拒绝本次"
                      f"发布并重试；已冻结 generation 的身份绑定 staging"
                      f" bytes，不得用新一代源 SHA 发布旧一代评分，"
                      f"fail closed（062）")
        binding = {
            "status": rcp["status"],
            "run_id": rcp["run_id"],
            "prediction_basename": rcp["prediction_basename"],
            "prediction_sha256": rcp["prediction_sha256"],
            "prediction_lines": rcp["prediction_lines"],
            # 062：verified 只在 staging == 回执 == 源 三方一致时为真
            "verified_same_generation": True,
            "staged_sha256": staged_sha,
            "source_sha256_at_freeze": source_sha,
        }
    else:
        # v1 无绑定字段：source SHA 仍按当前字节记录（无三方断言语义）
        source_sha = _file_sha256(source_pred)
    # v1：binding 保持 None（provenance 由 _resolve_yarn_identity 降级标注）
    gp = rcp["generation_params"]
    return {
        "path": os.path.abspath(rcp_path),
        # 062①：receipt SHA 与解析同源（同一 bytes 快照）
        "sha256": rcp_sha,
        "receipt_version": version,
        # 066/crash-recovery：generation 绑定口径——pointer-v1 = 从
        # {out}.tli_gen 指针解析后同时读预测与回执（物理来源同源）；
        # legacy-direct = 无指针产物按旁挂约定直接读（既有口径零改动）
        "generation_binding": binding_mode,
        # 066/crash-recovery：指针协议产物附带 generation 身份闭包
        # （指针路径/gen 目录名/预测 SHA256/回执 SHA256——manifest 级
        # 证据：本格评分对象的物理来源是指针所指不可变 gen 目录）
        "generation": (
            {"pointer_path": gen_info["pointer_path"],
             "gen_dir_name": os.path.basename(gen_info["gen_dir"]),
             "gen_pred_sha256": _file_sha256(gen_info["pred_path"]),
             "gen_rcp_sha256": rcp_sha}
            if gen_info is not None else None),
        "yarn_enabled": rcp["yarn_enabled"],
        "effective_yarn_factor": rcp["effective_yarn_factor"],
        "yarn_factor_source": rcp["yarn_factor_source"],
        "context_length": rcp["context_length"],
        "task": task,
        # 059：None = v1 回执无同代绑定（只证 factor 口径不证同代）
        "prediction_binding": binding,
        # 062③：冻结窗口内的源 SHA（三方一致；调用方复用，不重算）
        "source_sha256": source_sha,
        # 060：配置指纹（跨格一致性门禁输入；task/method/t/pred_postfix
        # 为分组自由字段，不进指纹——见 effective_config_sha256）
        "config_fingerprint": {
            "model_path": rcp["model_path"],
            "model_config_sha256": rcp["model_config_sha256"],
            "rope_scaling": rcp["rope_scaling"],
            "native_max_position_embeddings":
                rcp["native_max_position_embeddings"],
            "seed": gp["seed"],
            "max_gen": gp["max_gen"],
            # 064②：max_num 入指纹（yarn_receipt.effective_config_sha256
            # 同步收录；此前「必需+类型校验但不进指纹」形成闭包缺口）
            "max_num": gp["max_num"],
            "producer_script": rcp["producer_script"],
            "effective_config_sha256": effective_config_sha256(rcp),
            # 064①：model_config_sha256=None（远端模型 ID / config.json
            # 不在 model_path）→ 如实标注 missing，不得称完整配置闭包；
            # 有值 = config.json 字节已 SHA 绑定
            "config_identity": ("config_json_sha256_bound"
                                if rcp["model_config_sha256"] is not None
                                else "missing"),
        },
    }


def freeze_and_stage(root, postfix, expect_tasks, stamp, staging,
                     min_samples, data_root):
    """阶段一（E116e 重构）：原始候选只读拷贝进 staging 派生目录 →
    legacy 补刻（派生副本上）→ 长度身份门禁 → min-samples 硬门禁 →
    best-file 仲裁 → 跨 method 身份（ids/answers_sha/lengths 逐行）一致性
    → 源数据 SHA256 绑定 → 生产者 yarn receipt 探查（057）→ 冻结 manifest。
    返回 (tasks_manifest, cells_info, src_data_sha, producer_yarn)；
    producer_yarn = {"cells": {格键: receipt 摘要}, "missing": [格键]}——
    best-file 旁挂 receipt 的存在性即生产者证据（存在即证据，半写/损坏/
    自相矛盾一律 fail-closed，见 _load_producer_yarn_receipt）。

    066（TL-E119-YARN-SAME-BYTES-PROVENANCE）：每个 (task, method) 格的
    完整冻结窗口（候选复制 → best-file 仲裁 → 回执 bytes 快照 → 三方
    校验）在【与生产者同键的输出路径锁】内执行（acquire_output_lock，
    候选多键按锁键排序获取、逆序释放）——生产者生成/提交生命周期同样
    持锁（pred_ruler.py 059 口径），窗口内 B 代两步提交不可穿插，「同
    字节不同配置」的回执/预测换代不可达；锁获取失败 fail-closed；
    acquire 等待时长逐格记入 cells[].tasks[].freeze_lock。"""
    staged_root = os.path.join(staging, "pred_root")
    tasks_manifest = {}   # {task: {ids, answers_sha, lengths, identity_mode}}
    cells_info = {}       # {key: {length_dir, identity_mode, tasks: {...}}}
    src_data_sha = {}     # {"{Lnum}/{task}": {path, sha256}}
    producer_yarn = {"cells": {}, "missing": []}   # 057
    n_stamped = 0
    for L_dir in sorted(glob.glob(os.path.join(root, "L*"))):
        Lname = os.path.basename(L_dir)
        m = re.fullmatch(r"L(\d+)", Lname)
        if not m:
            _fail(f"非法长度目录名 {Lname}（须形如 L32768）——fail closed")
        Lnum = int(m.group(1))
        pred_dir = os.path.join(L_dir, f"pred{postfix}")
        if not os.path.isdir(pred_dir):
            continue
        for task in TASKS:
            # ---- 066/crash-recovery：候选发现 = legacy 直写文件 + 指针代预测 ----
            # 指针协议（#198）下预测不再落最终路径 {out}.jsonl，而是与完成
            # 回执同置不可变 gen 目录 {out}.gen-{attempt_id}/，提交 = 切指针
            # {out}.tli_gen。候选发现双通道：
            #   ① legacy 直写文件（{task}-*.jsonl glob，既有口径零改动）；
            #   ② 指针文件（{task}-*.jsonl.tli_gen glob）→ resolve_
            #      generation_pointer 解析出 gen 目录内预测作为复制源
            #      （指针存在但损坏/缺件 → resolve 内部 fail-closed，不静默
            #      回退 legacy 直读）。同基名双通道并存时指针代优先——指针
            #      切换 = 唯一提交信号，直写残留只可能来自更早的旧协议运行
            #      （新代码不写直写文件）；-merged 派生物两通道一致排除。
            legacy_files = [f for f in sorted(glob.glob(
                os.path.join(pred_dir, f"{task}-*.jsonl")))
                if not os.path.basename(f).endswith("-merged.jsonl")]
            cand = {}   # basename -> {source, final, binding}
            for f in legacy_files:
                cand[os.path.basename(f)] = {
                    "source": f, "final": f,
                    "binding": GENERATION_BINDING_LEGACY}
            for p in sorted(glob.glob(
                    os.path.join(pred_dir,
                                 f"{task}-*{GENERATION_POINTER_SUFFIX}"))):
                final = p[: -len(GENERATION_POINTER_SUFFIX)]
                base = os.path.basename(final)
                if base.endswith("-merged.jsonl"):
                    continue
                gen_info = resolve_generation_pointer(final)
                if gen_info is None:
                    _fail(f"{p}: 指针文件存在但 resolve 返回 None——"
                          f"指针协议解析内部矛盾，fail closed"
                          f"（066/crash-recovery）")
                cand[base] = {
                    "source": gen_info["pred_path"], "final": final,
                    "binding": GENERATION_BINDING_POINTER}
            files = sorted(cand)   # 候选基名（排序确定性，既有口径）
            if not files:
                continue
            # 032 源数据身份绑定：{data_root}/{L}/{task}.jsonl 必须存在
            data_file = os.path.join(data_root, str(Lnum), f"{task}.jsonl")
            if not os.path.isfile(data_file):
                _fail(f"源数据文件缺失: {data_file}（task={task}, L={Lname}）"
                      f"——身份闭包不完整，fail closed")
            src_data_sha.setdefault(f"{Lnum}/{task}", {
                "path": data_file, "sha256": _file_sha256(data_file)})
            groups = {}
            for f in files:
                method = os.path.basename(f).replace(f"{task}-", "") \
                    .rsplit("-", 1)[0]
                groups.setdefault(method, []).append(f)
            for method, gfiles in groups.items():
                # ---- 066（TL-E119-YARN-SAME-BYTES-PROVENANCE，GPT 方案 2 共享锁）----
                # 062 的三方 SHA 一致只证明内容等价（staging == 回执声明 ==
                # 源当前字节）；两个配置不同的运行产生【字节完全相同】的
                # 预测时（GPT 复现 run-A/seed=42/max_num=1 vs run-B/seed=99/
                # max_num=100），「复制到 staging 后、读回执前」的 B 代两步
                # 提交穿插三方校验全过 → staging 配 A 代字节、manifest 归属
                # B 代 run_id/config 且 verified_same_generation=true——运行
                # provenance 错挂（数值不变故 P2；当 treatment 因果闭包证据
                # 用则升 P1）。修复：本格完整冻结窗口（候选复制 → best-file
                # 仲裁 → 回执 bytes 快照 → 三方校验）全程持有与生产者同键
                # 的 flock（yarn_receipt.acquire_output_lock，056/59 键口径：
                # realpath(父目录) + basename + .attempt.lock）——生产者的
                # 生成→提交生命周期同样持锁（pred_ruler.py），窗口内两步
                # os.replace 提交不可穿插；锁释放后的新一代提交不改变已冻结
                # generation 的身份（062④ 绑定 staging bytes）。纵深防御：
                # 锁防【活进程】穿插（run_id/config 归属闭合），062 三方
                # SHA 校验保留防【死亡中间态】（crash-between 失配代际不经
                # 锁，仍由三方校验拒收）。候选多键按锁键排序获取、逆序释放
                # （043 写集锁同款纪律；生产者单键持有、formal 同序获取 →
                # 无死锁环）。锁不可用/获取失败 → fail-closed 不静默降级
                #（同代冻结窗口无锁即不可闭合）；acquire 等待时长逐格记入
                # manifest（cells[].tasks[].freeze_lock 轻量审计字段，长
                # 时间 formal 持锁对补跑吞吐的影响据此可审计）。----
                # 066/crash-recovery：锁键 = 候选【最终路径】——生产者锁
                # 在 {out} 最终路径上（pred_ruler acquire_output_lock
                # (out_path)），指针代候选的 source 在 gen 目录内但互斥
                # 键必须与生产者同键，否则 B3 活进程互斥失效。
                lock_keys = sorted(cand[f]["final"] for f in gfiles)
                lock_fds = []
                _lock_t0 = time.perf_counter()
                for _lf in lock_keys:
                    try:
                        lock_fds.append(acquire_output_lock(_lf))
                    except OSError as e:
                        # 已获取的锁逆序释放后再 fail-closed（不留半持有态）
                        for _fd in reversed(lock_fds):
                            try:
                                release_output_lock(_fd)
                            except OSError:
                                pass
                        _fail(f"{_lf}: 冻结窗口输出路径锁获取失败（{e}）——"
                              f"与生产者的同代互斥不可建立（066），run_id/"
                              f"config 归属无法闭合，fail closed；只读/权限"
                              f"受限的 root 须先复制到可写位置再跑 formal")
                _lock_wait_s = time.perf_counter() - _lock_t0
                try:
                    # ---- 036：只读拷贝 + legacy 补刻（派生副本）----
                    staged_files = []
                    stamped_flags = {}
                    for f in gfiles:
                        # 066/crash-recovery：复制源 = 候选实际源（legacy =
                        # 直写文件本身；指针代 = gen 目录内预测）
                        src_f = cand[f]["source"]
                        dst = os.path.join(staged_root, Lname, f"pred{postfix}",
                                           f)
                        stamped, rows = _stamp_or_copy(src_f, dst, task,
                                                       stamp=stamp)
                        staged_files.append(dst)
                        stamped_flags[dst] = stamped
                        if stamped:
                            n_stamped += 1
                            print(f"[formal] legacy 补刻 _id（派生副本）: {f}")
                        # ---- 032 长度身份门禁 ①：行 length ≤ L 档位 ----
                        # （length=实际 token 数；128K 数据混入 32K 目录 → 拒）
                        for r in rows:
                            if r.get("length") is None:
                                _fail(f"{os.path.basename(f)}: 行缺 length 字段"
                                      f"——无法做长度身份门禁，fail closed")
                            if r["length"] > Lnum:
                                _fail(
                                    f"{Lname}/{method}/{task}: 行 length="
                                    f"{r['length']} > 目录档位 {Lnum}（疑似 "
                                    f"{r['length']} 档数据混入 {Lname} 目录）"
                                    f"——长度身份门禁 fail closed")
                    # ---- best-file 仲裁（行数最多，并列取时间戳最新）----
                    best = max(staged_files, key=lambda f: (
                        _nlines(f), _ts_of(os.path.basename(f))))
                    rows = [json.loads(l)
                            for l in open(best, encoding="utf-8")]
                    ids = [r["_id"] for r in rows]
                    shas = {r["_id"]: r["_answers_sha"] for r in rows}
                    lengths = [r["length"] for r in rows]
                    if len(ids) != len(set(ids)):
                        _fail(f"{os.path.basename(best)}: _id 重复——fail closed")
                    # ---- 030：min-samples 发布硬门禁 ----
                    if len(rows) < min_samples:
                        _fail(
                            f"{Lname}/{method}/{task}: n={len(rows)} < "
                            f"--min-samples {min_samples}——样本数不完整，"
                            f"拒绝发布（min-samples 是发布门禁不是展示开关），"
                            f"不发布任何产物")
                    identity_mode = ("native" if not stamped_flags[best]
                                     else "legacy-partial")
                    key = f"{Lname}/{method}"
                    print(f"[formal] {key}/{task}: best-file "
                          f"{os.path.basename(best)} (n={len(rows)}) 冻结进 "
                          f"manifest（identity_mode={identity_mode}）")
                    # ---- 跨 method/L 身份一致性（032：ids + answers_sha +
                    # 逐行 length 三重比对）----
                    if task in tasks_manifest:
                        prev = tasks_manifest[task]
                        if set(prev["ids"]) != set(ids) or \
                                prev["answers_sha"] != shas:
                            _fail(f"task={task}: 跨方法/跨 L 目录的样本身份不一致"
                                  f"（ids 或 answers_sha 漂移）——fail closed，"
                                  f"manifest 拒绝生成")
                        if prev["lengths"] != lengths:
                            _fail(
                                f"task={task}: 跨方法逐行 length 不一致（同 "
                                f"row index/answers 而 length 漂移——输入身份"
                                f"不闭合）——fail closed（032 长度身份门禁）")
                    else:
                        tasks_manifest[task] = {
                            "ids": ids, "answers_sha": shas,
                            "lengths": lengths,
                            "identity_mode": identity_mode,
                        }
                    # ---- 057：生产者 yarn receipt 探查（best-file 旁挂）----
                    # best 是 staging 派生副本（与原始文件同名）；receipt 在
                    # 原始源旁挂（legacy）/ gen 目录内（指针代）探查。缺
                    # receipt = legacy（missing 如实记录，由
                    # _resolve_yarn_identity 裁决 provenance）。
                    # 062：staged=评分对象、source=回执旁挂探查对象——同代
                    # 绑定与三方一致校验在 _load_producer_yarn_receipt 内
                    # 以 staging bytes 为锚完成。
                    # 066/crash-recovery：src_best = 实际复制源（legacy 直写
                    # 文件 / 指针代 gen 目录内预测），source_final_path =
                    # best-file 最终路径（指针解析基准，回执与代归属从
                    # {final}.tli_gen 解析——物理来源同源，混合代不可达）。
                    best_base = os.path.basename(best)
                    src_best = cand[best_base]["source"]
                    best_final = cand[best_base]["final"]
                    cell_rcp = _load_producer_yarn_receipt(
                        best, src_best, task, Lnum,
                        source_final_path=best_final)
                    cell_key = f"{key}/{task}"
                    if cell_rcp is None:
                        producer_yarn["missing"].append(cell_key)
                    else:
                        producer_yarn["cells"][cell_key] = cell_rcp
                    # 062③：source_sha256 复用冻结窗口内的三方一致值
                    # （v2 时 = staging SHA = 回执 SHA；v1/legacy 无绑定语义
                    # 则为当前字节现算）——不得事后重算源 SHA：生产者在
                    # 冻结后推进源路径不改变已冻结 generation 的身份，
                    # 事后重算会捡到新一代造成 manifest 自相矛盾
                    if cell_rcp is not None:
                        cell_source_sha = cell_rcp["source_sha256"]
                    else:
                        cell_source_sha = _file_sha256(src_best)
                    cells_info.setdefault(key, {
                        "length_dir": Lnum,
                        "identity_mode": identity_mode,
                        "tasks": {},
                    })["tasks"][task] = {
                        "n": len(rows),
                        "best_file": best_base,
                        "source_path": os.path.abspath(
                            # best 与 staged 同名，映射回实际复制源
                            # （legacy 直写文件 / 指针代 gen 内预测）
                            src_best),
                        # 066/crash-recovery：best-file 最终路径（指针解析
                        # 基准；legacy 候选 = source_path 本身）
                        "source_final_path": os.path.abspath(best_final),
                        # 066/crash-recovery：generation 绑定口径——
                        # pointer-v1（预测+回执从 {final}.tli_gen 指针解析，
                        # 同 gen 目录同源）/ legacy-direct（直写路径直接读）
                        "generation_binding": cand[best_base]["binding"],
                        "source_sha256": cell_source_sha,
                        # v2 时三者已断言一致（staged == receipt == source）
                        "derived_sha256": _file_sha256(best),
                        # 057：生产者证据按格绑定（null=missing，legacy）；
                        # path+sha256 把 receipt 字节冻结进 manifest
                        "producer_yarn_receipt": cell_rcp,
                        # 066：冻结窗口锁审计字段（轻量）——acquire
                        # 等待秒数（生产者持锁时 formal 串行化等待的
                        # 吞吐影响据此可审计）
                        "freeze_lock": {
                            "n_files": len(gfiles),
                            "acquire_wait_seconds": round(_lock_wait_s, 6),
                        },
                    }
                finally:
                    # ---- 066：逆序释放本格全部冻结窗口锁 ----
                    for _fd in reversed(lock_fds):
                        release_output_lock(_fd)
    if expect_tasks > 0 and len(tasks_manifest) < expect_tasks:
        missing = sorted(set(TASKS[:expect_tasks]) - set(tasks_manifest))
        _fail(f"root={root} postfix={postfix}: manifest 只覆盖 "
              f"{len(tasks_manifest)}/{expect_tasks} 个任务（缺 {missing}）"
              f"——fail closed")
    if n_stamped:
        print(f"[formal] 共 {n_stamped} 个 legacy 文件在派生副本上补刻身份"
              f"（_id/_answers_sha；源文件零改动，逐字段不变量已断言）")
    _validate_manifest_schema(tasks_manifest)
    return tasks_manifest, cells_info, src_data_sha, producer_yarn


def _check_producer_config_consistency(found):
    """060（TL-E119-YARN-RECEIPT-CLOSURE）：跨格完整配置一致性门禁。

    修复前只比较 yarn_enabled/effective_factor 两项——回执声称保存的
    完整 rope_scaling、模型 config hash、seed、生产脚本 hash 写入零
    校验（GPT 审计 060 复现：beta_fast=999 / seed=999 / 模型与脚本假
    hash 均通过）。

    显式分组规则：按 receipt.context_length 分组（档位）；同一档内以下
    字段必须逐格逐位一致——model_path、model_config_sha256、完整
    rope_scaling、native_max_position_embeddings、producer_script
    （路径+SHA）、generation_params.seed、generation_params.max_gen、
    yarn 开关/effective factor/source（经 effective_config_sha256 规范
    化摘要统一比较）。task / method / t / pred_postfix 为分组自由字段
    （逐格允许变化，不进指纹）。跨档不比较——factor 等按档自动
    （65536→2.0、131072→4.0）；跨档的 factor 全局唯一性由
    _resolve_yarn_identity 的既有门禁（单一 effective 配置）承载。
    不一致 → _fail 报告首个漂移字段链（fail closed，python -O 不失效）。"""
    groups = {}
    for key, cell in sorted(found.items()):
        groups.setdefault(cell["context_length"], {})[key] = cell
    for cl, cells in sorted(groups.items()):
        hashes = {k: c["config_fingerprint"]["effective_config_sha256"]
                  for k, c in cells.items()}
        if len(set(hashes.values())) <= 1:
            continue
        ref_key = sorted(cells)[0]
        ref = cells[ref_key]["config_fingerprint"]
        # 先报具体字段（seed/model/rope/脚本 hash 等，诊断价值高），
        # effective_config_sha256 是派生摘要留作兜底——按字母序先撞 SHA
        # 会把「seed=42 vs 999」这类可定位漂移吞成不可读的 hex 失配；
        # config_identity（064①）同为派生标注（由 model_config_sha256
        # 是否为 None 派生），一并排除，漂移仍落在源字段上
        concrete = [f for f in sorted(ref)
                    if f not in ("effective_config_sha256",
                                 "config_identity")]
        for k in sorted(cells):
            cur = cells[k]["config_fingerprint"]
            for field in concrete:
                if ref[field] != cur[field]:
                    _fail(f"060 跨格配置漂移（L{cl} 档内）：{ref_key} 与 "
                          f"{k} 的 config_fingerprint.{field} 不一致"
                          f"（{ref[field]!r} vs {cur[field]!r}）——同一"
                          f"运行同档的完整生产配置必须逐位一致，"
                          f"fail closed（060）")
        _fail(f"060 跨格配置漂移（L{cl} 档内）：effective_config_sha256 "
              f"不一致（指纹差异来自 config_fingerprint 未展开收录的"
              f"字段，如 yarn 开关/factor/source）{hashes}——"
              f"fail closed（060）")


def _resolve_yarn_identity(args, producer_yarn):
    """057+059/060：run_identity 的 yarn 字段裁决（生产者证据优先）。

    路径：
      ① 全部格子有生产者 receipt → yarn/yarn_factor 取生产者 effective
        值；provenance 按协议代际分流（059）：全部 v2（同代绑定已逐格
        校验）→ "producer_receipt"；全部 v1 → 降级
        "producer_receipt_v1_partial"（只证 factor 口径不证回执与预测
        同代——057 时代产物不撤销、不冒充）；操作者 CLI 声明与生产者
        证据冲突 → _fail（事后声明不得覆盖实际值）；
      ② 全部格子无 receipt（legacy）→ CLI 值如实降级
        provenance="operator_declared"，不冒充实际生效值；
      ③ 部分有部分无 / v1 与 v2 混合 → _fail（单次 run_identity 不得
        混合两种口径/两种协议代际，fail closed）。
    060：进入裁决前先做跨格完整配置一致性门禁（同档内模型/rope/seed/
    脚本 hash 逐位一致，见 _check_producer_config_consistency）。
    _fail 走 SystemExit（非 assert），python -O 下门禁不失效。"""
    found = producer_yarn["cells"]
    missing = producer_yarn["missing"]
    n_total = len(found) + len(missing)
    if not found:
        return {
            "yarn": bool(args.yarn),
            "yarn_factor": args.yarn_factor,
            "yarn_factor_provenance": "operator_declared",
            "yarn_factor_operator_declared": args.yarn_factor,
            "producer_evidence": {
                "status": "missing",
                "cells_with_receipt": 0,
                "cells_total": n_total,
                "note": ("legacy 产物无生产者 yarn receipt：yarn_factor 为"
                         "操作者事后 CLI 声明（operator_declared），实际"
                         "生效值未闭合——不冒充 effective 值（057）"),
            },
        }
    if missing:
        _fail(f"生产者 yarn receipt 覆盖不全：{len(found)}/{n_total} 格有"
              f"证据（缺 {sorted(missing)}）——单次 formal 的 run_identity "
              f"不能混合『生产者证实』与『操作者声明』两种口径，"
              f"fail closed（057）")
    # 059：协议代际一致性——v1（只证 factor）与 v2（同代绑定）不得混装
    versions = {c["receipt_version"] for c in found.values()}
    if len(versions) > 1:
        _fail(f"生产者 yarn receipt 协议代际混合：{sorted(versions)}——"
              f"单次 run_identity 不得混合 v1（factor 口径）与 v2（同代"
              f"绑定）两种证据代际，fail closed（059）")
    version = versions.pop()
    # 060：跨格完整配置一致性门禁（同档内应逐位一致的字段）
    _check_producer_config_consistency(found)
    enableds = {c["yarn_enabled"] for c in found.values()}
    factors = {c["effective_yarn_factor"] for c in found.values()}
    if len(enableds) > 1 or len(factors) > 1:
        _fail(f"生产者 yarn receipt 之间不一致：yarn_enabled="
              f"{sorted(enableds, key=str)}，effective_factor="
              f"{sorted(factors, key=str)}——同一 run_identity 无法声明"
              f"单一 effective 配置（混合档位/factor 的 root 须按档拆分"
              f"formal 运行），fail closed（057）")
    enabled = enableds.pop()
    factor = factors.pop()
    # 操作者 CLI 声明 vs 生产者证据冲突 → fail-closed（057 核心门禁：
    # 修复前 128K 批次正是「派单未传 factor → 4.0 自动档生效，formal
    # 事后声明 2.0 原样入 manifest」才产生身份冲突）
    if bool(args.yarn) != enabled:
        _fail(f"yarn 声明冲突：CLI --yarn={bool(args.yarn)} vs 生产者 "
              f"receipt yarn_enabled={enabled}（{len(found)} 格证据一致）"
              f"——生产者证据优先，事后声明不得覆盖实际值，"
              f"fail closed（057）")
    if args.yarn_factor is not None and (not enabled or
                                          factor != args.yarn_factor):
        _fail(f"yarn factor 声明冲突：CLI --yarn-factor={args.yarn_factor} "
              f"vs 生产者 receipt effective={factor!r}（yarn_enabled="
              f"{enabled}）——生产者证据优先，fail closed（057）")
    provenance = ("producer_receipt" if version == RECEIPT_VERSION
                  else "producer_receipt_v1_partial")
    # 064①：模型配置闭包状态如实汇总——任一格 model_config_sha256=None
    # （远端模型 ID / config.json 不在 model_path）→ 闭包不完整，降级
    # 标注；不得称「完整模型配置闭包」（不强行做 config.to_dict() 哈希
    # ——引入 transformers 违反 yarn_receipt 零重依赖设计原则，选诚实
    # 降级；逐格 config_identity 见 manifest cells 的 config_fingerprint）
    config_missing_cells = sorted(
        k for k, c in found.items()
        if c["config_fingerprint"]["model_config_sha256"] is None)
    model_config_closure = not config_missing_cells
    evidence = {
        "status": "present",
        "protocol": version,
        # 059：v2 才声称同代绑定（prediction SHA/行数已逐格比对）；
        # v1 只证 factor 口径，不冒充回执与预测同代
        "same_generation_bound": version == RECEIPT_VERSION,
        # 060：完整配置跨格一致性门禁已执行（此前只比开关+factor）
        "config_consistency_enforced": True,
        # 064①：True = 全部格 config.json 字节已 SHA 绑定；False = 存在
        # missing 格（model_config_sha256=None），完整模型配置闭包不成立
        "model_config_closure": model_config_closure,
        # 066/crash-recovery：generation 绑定口径聚合（pointer-v1 = 预测与
        # 回执从 {out}.tli_gen 指针解析（同 gen 目录同源，混合代不可达）；
        # legacy-direct = 直写路径直接读（059 v2 及更早，锁+SHA 双层兜底）。
        # 逐格明细见 cells[].tasks[].generation_binding；过渡期混装如实
        # 记录两种口径（协议代际一致性仍由上方 versions 门禁承载）
        "generation_binding_modes": sorted({
            c.get("generation_binding", GENERATION_BINDING_LEGACY)
            for c in found.values()}),
        "cells_with_receipt": len(found),
        "cells_total": n_total,
        "yarn_enabled": enabled,
        "effective_yarn_factor": factor if enabled else None,
        "receipts": {k: {"path": v["path"], "sha256": v["sha256"]}
                    for k, v in sorted(found.items())},
    }
    if not model_config_closure:
        evidence["model_config_closure_note"] = (
            f"以下 {len(config_missing_cells)} 格生产者回执的 "
            f"model_config_sha256=None（远端模型 ID / config.json 不在 "
            f"model_path，生产侧 pred_ruler 无法本地取 config.json 字节）："
            f"{config_missing_cells}——模型配置身份未逐位闭合，"
            f"config_identity=missing（逐格见 cells[].tasks[]."
            f"producer_yarn_receipt.config_fingerprint），不得宣称"
            f"「完整模型配置闭包」（064 降级标注）")
    if version == RECEIPT_V1_VERSION:
        evidence["note"] = ("v1 回执（057 时代协议）只证 factor 口径，"
                            "不证回执与预测同代（无 prediction SHA/行数"
                            "绑定字段）——producer_receipt_v1_partial 降级"
                            "标注（059）；完整配置 schema 与跨格一致性"
                            "门禁仍已执行（060）")
    return {
        "yarn": enabled,
        "yarn_factor": factor if enabled else None,
        "yarn_factor_provenance": provenance,
        "yarn_factor_operator_declared": args.yarn_factor,
        "producer_evidence": evidence,
    }


def main():
    ap = argparse.ArgumentParser(
        description="RULER 生产级正式打分入口（E116e：staging 原子发布 + "
                    "min-samples 硬门禁 + 身份闭包 + 源文件只读）")
    ap.add_argument("--root", required=True,
                    help="结果根目录（其下 L*/pred{postfix}/，全程只读）")
    ap.add_argument("--pred-postfix", required=True,
                    help="pred 目录后缀（区分臂，如 _E109_FULLKV）")
    ap.add_argument("--out", required=True,
                    help="结果 JSON 输出路径（必须以小写 .json 结尾）")
    ap.add_argument("--manifest-out", default="",
                    help="冻结 manifest 落盘路径（默认 <out>.manifest.json）")
    ap.add_argument("--expect-tasks", type=int, default=11,
                    help="预期任务闭包数（默认 11=RULER 全任务）")
    ap.add_argument("--min-samples", type=int, default=100,
                    help="发布硬门禁：任一格 n 低于该值 → 非零退出不发布")
    ap.add_argument("--data-root", required=True,
                    help="RULER 源数据根目录（其下 {L}/{task}.jsonl；"
                         "manifest 绑定逐任务源数据 SHA256）")
    ap.add_argument("--model-path", default="",
                    help="声明生成该预测所用模型路径（身份绑定；legacy 数据"
                         "可留空，manifest 记 null 不冒充）")
    ap.add_argument("--yarn", action="store_true",
                    help="声明生成时启用 YaRN（057：有生产者 receipt 时"
                         "须与其一致，冲突 fail-closed；legacy 无证据时"
                         "记为操作者声明）")
    ap.add_argument("--yarn-factor", type=float, default=None,
                    help="声明 YaRN factor（057：有生产者 receipt 时必须"
                         "与其 effective 值一致，冲突 fail-closed；legacy "
                         "无证据时降级 operator_declared，不冒充实际"
                         "生效值）")
    ap.add_argument("--extra-param", action="append", default=[],
                    metavar="K=V",
                    help="附加身份参数（可重复，如 --extra-param "
                         "tia_level1_topk=1024）")
    ap.add_argument("--no-stamp-legacy-ids", action="store_true",
                    help="禁用 legacy _id 补刻（缺 _id 数据将 fail closed）")
    args = ap.parse_args()

    # ---- 033：--out 后缀门禁 + 四路径两两不同（在任何文件创建之前）----
    basename = os.path.basename(args.out)
    if not basename.endswith(".json") or basename == ".json":
        _fail(f"--out 必须以非空小写 .json 后缀结尾，得到 {args.out!r}"
              f"（无后缀/大写 .JSON/路径中间含 .json 均拒绝——防止 MD "
              f"覆盖 JSON 同一文件）")
    md_path = args.out[:-len(".json")] + ".md"   # 后缀精确推导，不用 replace
    manifest_path = args.manifest_out or (args.out + ".manifest.json")
    receipt_path = args.out + ".receipt.json"
    four = [os.path.abspath(p) for p in
            (args.out, md_path, manifest_path, receipt_path)]
    if len(set(four)) != len(four):
        _fail(f"JSON/MD/manifest/receipt 路径必须两两不同，得到 "
              f"{four}（--manifest-out 不得与 --out 或其派生路径相同）")

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{ts}-{os.getpid()}-{random.randint(1000, 9999)}"
    # staging/run_dir 绝对化：scorer 子进程以 cwd=REPO 运行，相对路径会
    # 相对 REPO 解析而非调用者 cwd——统一用绝对路径消除歧义
    staging = os.path.abspath(f"{args.out}.staging-{run_id}")
    # E116f：run_dir 即不可变 generation 目录（四规范文件 + 派生副本 +
    # merged 文件同目录共存），staging 校验完成后一次原子 rename 落位
    run_dir = os.path.abspath(f"{args.out}.run-{run_id}")

    # E116f（038）发布事务状态：在 try 之前定义，失败路径据此回滚
    backups = {}    # 公开路径 → 备份路径（None = 旧产物不存在/首轮）
    published = []  # 已成功替换的公开镜像（回滚清单，按替换顺序）
    # E116g（040）：锁内回滚状态（_locked_publish 在锁内完成回滚后置
    # done=True，外层 except 据此跳过重复回滚——锁释放后的二次回滚会
    # 与下一个发布者交错，正是 040 要消除的窗口）
    # committed（kimi3 1404 审查）：发布事务成功提交后置 True——锁释放
    # 后的 print 等报告型 I/O 失败（如 stdout broken pipe）不得回滚已
    # 发布产物、不得删除 generation 目录
    rollback_state = {"done": False, "committed": False,
                      "n_restored": 0, "errors": []}

    extra_params = {}
    for kv in args.extra_param:
        if "=" not in kv:
            _fail(f"--extra-param 须为 K=V 形式，得到 {kv!r}")
        k, v = kv.split("=", 1)
        extra_params[k] = v

    try:
        if not os.path.isdir(args.data_root):
            _fail(f"--data-root 不是目录: {args.data_root}")
        os.makedirs(staging, exist_ok=True)

        # ---- 阶段一：staging 派生 + 冻结 manifest（含身份扩展 032 +
        #      057 生产者 yarn 证据消费）----
        tasks_manifest, cells_info, src_data_sha, producer_yarn = \
            freeze_and_stage(
                args.root, args.pred_postfix, args.expect_tasks,
                stamp=not args.no_stamp_legacy_ids, staging=staging,
                min_samples=args.min_samples, data_root=args.data_root)

        # ---- 057：run_identity 的 yarn 字段裁决（生产者证据优先；
        #      冲突 fail-closed；legacy 无证据 → operator_declared；
        #      059：v1 回执 → producer_receipt_v1_partial 降级标注）----
        yarn_id = _resolve_yarn_identity(args, producer_yarn)
        _prov = yarn_id["yarn_factor_provenance"]
        if _prov == "producer_receipt":
            pe = yarn_id["producer_evidence"]
            print(f"[formal] yarn 身份：producer_receipt（v2 同代绑定）"
                  f"证实 yarn={yarn_id['yarn']} effective_factor="
                  f"{yarn_id['yarn_factor']!r}"
                  f"（{pe['cells_with_receipt']}/{pe['cells_total']} 格"
                  f"生产者证据闭合，prediction SHA/行数逐格比对通过，"
                  f"057+059）")
        elif _prov == "producer_receipt_v1_partial":
            pe = yarn_id["producer_evidence"]
            print(f"[formal] yarn 身份：v1 回执只证 factor 口径不证同代 → "
                  f"producer_receipt_v1_partial 降级（yarn="
                  f"{yarn_id['yarn']} effective_factor="
                  f"{yarn_id['yarn_factor']!r}，"
                  f"{pe['cells_with_receipt']}/{pe['cells_total']} 格，"
                  f"057+059）")
        else:
            print(f"[formal] yarn 身份：生产者证据缺失 → CLI 声明降级 "
                  f"operator_declared（yarn={yarn_id['yarn']} "
                  f"yarn_factor={yarn_id['yarn_factor']!r}，实际生效值"
                  f"未闭合，不冒充 effective 值，057）")

        manifest_full = {
            "manifest_version": 2,
            "run_id": run_id,
            "generated": datetime.now().isoformat(),
            "root": os.path.abspath(args.root),
            "pred_postfix": args.pred_postfix,
            "expect_tasks": args.expect_tasks,
            "min_samples": args.min_samples,
            "run_identity": {
                "data_root": os.path.abspath(args.data_root),
                "model_path": args.model_path or None,
                "yarn": yarn_id["yarn"],
                # 057：yarn/yarn_factor 不再无条件等于 CLI 声明——
                # provenance=producer_receipt 时为生产者 receipt 证实的
                # 实际生效值；operator_declared 时为操作者事后声明
                # （实际生效值未闭合，不冒充）
                "yarn_factor": yarn_id["yarn_factor"],
                "yarn_factor_provenance":
                    yarn_id["yarn_factor_provenance"],
                "yarn_factor_operator_declared":
                    yarn_id["yarn_factor_operator_declared"],
                "producer_evidence": yarn_id["producer_evidence"],
                "extra_params": extra_params,
                "formal_script_sha256": _file_sha256(FORMAL_PATH),
                "scorer_sha256": _file_sha256(SCORER_PATH),
                # legacy 声明口径：model/yarn 等为操作者事后声明，
                # 可能不可恢复——以 receipt/manifest 记录为准，不冒充完整
                "note": ("model_path 为操作者声明值；yarn_factor 按 "
                         "yarn_factor_provenance 区分：producer_receipt="
                         "生成进程旁挂 v2 回执证实的实际生效值（回执与"
                         "best-file 经 prediction SHA/行数逐位同代绑定，"
                         "059），producer_receipt_v1_partial=v1 回执只证 "
                         "factor 口径不证回执与预测同代（059 降级标注），"
                         "operator_declared=legacy 产物无生产者证据时的"
                         "操作者事后声明（实际生效值未闭合，不冒充）；"
                         "legacy 数据无法从文件恢复完整输入身份"
                         "（identity_mode=legacy-partial），native 数据"
                         "由 pred_ruler.py 落盘"),
            },
            "source_data_sha256": src_data_sha,
            "cells": cells_info,
            # score_ruler.py 兼容子集（ids + answers_sha；lengths/
            # identity_mode 为 E116e 身份扩展，scorer 调用时剥离）
            "tasks": tasks_manifest,
        }
        staged_manifest = os.path.join(staging, "manifest.json")
        json.dump(manifest_full, open(staged_manifest, "w"), indent=1,
                  ensure_ascii=False)

        # ---- 阶段二：scorer 子进程（root=staging 派生副本；merged 规范
        # 文件也只落在 staging 内，原始目录零写入）----
        scorer_manifest = os.path.join(staging, "scorer.manifest.json")
        json.dump({t: {"ids": m["ids"], "answers_sha": m["answers_sha"]}
                   for t, m in tasks_manifest.items()},
                  open(scorer_manifest, "w"))
        staged_result = os.path.join(staging, "result.json")
        argv = [sys.executable, "-u", "-m", "benchmark.RULER.score_ruler",
                "--root", os.path.join(staging, "pred_root"),
                "--pred-postfix", args.pred_postfix,
                "--out", staged_result,
                "--min-samples", str(args.min_samples),
                "--manifest", scorer_manifest,
                "--expect-tasks", str(args.expect_tasks), "--merge-best"]
        r = subprocess.run(argv, cwd=REPO,
                           env={**os.environ, "PYTHONPATH": REPO})
        if r.returncode != 0:
            _fail(f"scorer 退出码 {r.returncode} —— 正式打分失败（scorer "
                  f"阶段门禁触发，不发布）")

        # ---- 发布前终验（staging 内完成；031 原子发布前置条件）----
        staged_md = os.path.join(staging, "result.md")
        if not os.path.isfile(staged_result) or not os.path.isfile(staged_md):
            _fail("scorer 成功但 staging 产物缺失（result.json/result.md）"
                  "——不发布")
        res = json.load(open(staged_result))
        for key, tasks in res["n"].items():
            for t, n in tasks.items():
                if n < args.min_samples:   # 030 复核（双保险）
                    _fail(f"{key}/{t}: scorer 结果 n={n} < min-samples "
                          f"{args.min_samples}——拒绝发布")
        # 每格源→派生 sha 追溯 + 逐字段不变量已由 _stamp_or_copy 断言；
        # receipt 汇总 cells 级 source→derived 映射
        avgs = {k: round(sum(s.values()) / len(s), 2)
                for k, s in res["scores"].items()}

        receipt = {
            "run_id": run_id,
            "status": "success",
            "generated": datetime.now().isoformat(),
            # E116f/E116j：发布协议版本字段（旧格式 receipt 缺该键 =
            # E116e 及之前的多文件逐个替换协议；e116f-generation-v2 =
            # generation 单指针 + 四角色存在性；e116i-generation-v3 =
            # v2 + entry receipt 对 generation md/receipt 的内容绑定）
            "publish_protocol": "e116i-generation-v3",
            "formal": {"script": "benchmark/RULER/score_ruler_formal.py",
                       "sha256": _file_sha256(FORMAL_PATH)},
            "scorer": {"script": "benchmark/RULER/score_ruler.py",
                       "sha256": _file_sha256(SCORER_PATH)},
            "inputs": {
                "root": os.path.abspath(args.root),
                "pred_postfix": args.pred_postfix,
                "data_root": os.path.abspath(args.data_root),
                "expect_tasks": args.expect_tasks,
                "min_samples": args.min_samples,
                "run_identity": manifest_full["run_identity"],
            },
            "outputs": {"json": os.path.abspath(args.out),
                        "md": os.path.abspath(md_path),
                        "manifest": os.path.abspath(manifest_path),
                        "derived_dir": os.path.abspath(run_dir),
                        # 消费者可从 receipt 单指针解析同一 generation
                        # 内的全部规范文件（不跨固定别名拼装）
                        "generation_files": {
                            "json": "result.json",
                            "md": "result.md",
                            "manifest": "manifest.json",
                            "receipt": "receipt.json"}},
            "manifest_sha256": _file_sha256(staged_manifest),
            "result_sha256": _file_sha256(staged_result),
            "source_data_sha256": src_data_sha,
            "cells": cells_info,
            "avg": avgs,
            "legacy_stamp": {
                "n_stamped": sum(
                    1 for c in cells_info.values()
                    for t in c["tasks"]
                    if c["identity_mode"] == "legacy-partial"),
                "invariant_fields": list(INVARIANT_FIELDS),
                "invariant_asserted": True,
                "note": ("补刻只发生在 staging 派生副本；源文件 SHA256 记录"
                         "于 cells[].tasks[].source_sha256，全程只读"),
            },
        }
        staged_receipt = os.path.join(staging, "receipt.json")
        json.dump(receipt, open(staged_receipt, "w"), indent=1,
                  ensure_ascii=False)

        # ==== E116f：generation 目录 + 单指针提交发布协议 ====
        # （修复审计 038/039；CLI 与产物路径完全不变，仅事务语义重构）
        # ① 预建四个公开目标的父目录（此时尚未触碰任何旧产物）；
        for p in (args.out, md_path, manifest_path, receipt_path):
            d = os.path.dirname(p)
            if d:
                os.makedirs(d, exist_ok=True)
        # ② st_dev 校验：generation rename 的源（staging）与目标
        #    （run_dir）同父目录必然同设备——防御性 fail-closed（审计
        #    建议 2 的 rename 对校验；公开镜像的跨设备安全由
        #    _atomic_install 的目标文件系统临时文件保证）
        _parent = os.path.dirname(staging) or "."
        if os.stat(staging).st_dev != os.stat(_parent).st_dev:
            _fail(f"staging {staging} 与其父目录跨设备（理论不可达）"
                  f"——generation rename 无法原子，fail closed")
        # ③ 整个 staging（result/md/manifest/receipt 四规范文件 +
        #    pred_root 派生副本 + merged 规范文件）fsync 后一次原子
        #    rename 成不可变 generation——receipt 声明的 derived_dir
        #    在任何公开文件发布前已存在（修复 039）
        for p in (staged_result, staged_md, staged_manifest,
                  staged_receipt):
            _fsync_file(p)
        _fsync_dir(staging)
        os.rename(staging, run_dir)   # 唯一的 rename：generation 提交
        _fsync_dir(_parent)
        # ④ generation 完整性校验（receipt 的 derived_dir 必须已存在且
        #    四规范文件 SHA 与 receipt 声明逐位一致，否则不发布）
        if not os.path.isdir(run_dir):
            _fail(f"receipt 声明的 derived_dir 尚不存在: {run_dir}"
                  f"——不允许发布（039）")
        gen = {
            "json": os.path.join(run_dir, "result.json"),
            "md": os.path.join(run_dir, "result.md"),
            "manifest": os.path.join(run_dir, "manifest.json"),
            "receipt": os.path.join(run_dir, "receipt.json"),
        }
        for name, p in gen.items():
            if not os.path.isfile(p):
                _fail(f"generation 目录缺规范文件 {name}: {p}——不发布")
        if _file_sha256(gen["manifest"]) != receipt["manifest_sha256"] or \
                _file_sha256(gen["json"]) != receipt["result_sha256"]:
            _fail("generation 规范文件 SHA 与 receipt 声明不一致——不发布")
        # ==== E116j（052）：entry receipt——generation 内容哈希绑定 ====
        # 对已冻结 generation 目录内的 result.md 与 receipt.json 计算
        # SHA256，作为 gen_md_sha256 / gen_receipt_sha256 写入 entry
        # receipt。receipt 不自哈希（避免自引用）：entry receipt（安装
        # 到公开 {out}.receipt.json）与 generation 内 receipt.json 是两个
        # 不同文件，前者哈希后者。entry receipt 落在 generation 内
        # （entry_receipt.json），作为公开 receipt 镜像的安装源——锁内
        # SHA 终验保证公开 receipt 与 generation 副本逐位一致。
        gen_md_sha = _file_sha256(gen["md"])
        gen_rc_sha = _file_sha256(gen["receipt"])
        entry_receipt = dict(receipt)
        entry_receipt["gen_md_sha256"] = gen_md_sha
        entry_receipt["gen_receipt_sha256"] = gen_rc_sha
        entry_p = os.path.join(run_dir, "entry_receipt.json")
        json.dump(entry_receipt, open(entry_p, "w"), indent=1,
                  ensure_ascii=False)
        _fsync_file(entry_p)
        _fsync_dir(run_dir)
        # 写后重读自校验：绑定字段必须与 generation 当前内容逐位一致
        # （防序列化异常悄悄破坏内容绑定声明）
        _chk = json.load(open(entry_p))
        if (_chk.get("gen_md_sha256") != _file_sha256(gen["md"]) or
                _chk.get("gen_receipt_sha256") != _file_sha256(
                    gen["receipt"])):
            _fail("entry receipt 落盘重读的 gen_*_sha256 与 generation "
                  "实际内容不一致——内容绑定声明不可信，不发布（052）")
        # ⑤ 公开固定路径作为兼容镜像逐个安装（JSON → MD → manifest →
        #    entry receipt；receipt 镜像最后落盘 = 唯一提交信号）。E116g
        #    （040）：整个安装事务在进程间独占锁内执行——「备份建立 →
        #    四镜像安装 → SHA 终验 → 备份清理或锁内回滚」，同一 --out
        #    的并发发布者串行化，混合代际不可达
        mirrors = [
            (gen["json"], args.out),
            (gen["md"], md_path),
            (gen["manifest"], manifest_path),
            (entry_p, receipt_path),
        ]
        _locked_publish(mirrors, run_id, args.out, backups, published,
                        rollback_state)
        print(f"[formal] 发布成功（generation 原子提交 + 兼容镜像）："
              f"{args.out} / {md_path} / {manifest_path} / {receipt_path}")
        print(f"[formal] 派生目录（generation，含四规范文件+补刻副本"
              f"+merged 规范文件）: {run_dir}")
        print(f"DONE {args.out}")
    except (SystemExit, OSError) as e:
        # ---- 031/030/036/E116f：失败路径——回滚已替换镜像 + 独立 failure
        # receipt + 清理 staging/generation，不触碰（或逐位还原）旧成功
        # 产物与源文件 ----
        if isinstance(e, SystemExit):
            msg = str(e.code) if e.code is not None and str(e.code) else \
                f"exit({e.code})"
        else:
            msg = f"{type(e).__name__}: {e}"
        print(msg, file=sys.stderr)   # 拒绝原因同时输出到 stderr（可观测）
        # E116f（038）回滚：published 中的公开镜像已换成新代际 → 从备份
        # 逐位还原（os.replace，与备份同目录必同设备、原子）；旧产物原本
        # 不存在（首轮）→ 删除新镜像。回滚后不存在「旧 receipt 配新结果」
        # E116g（040）：镜像安装段的失败已在 _locked_publish 锁内完成
        # 回滚（rollback_state.done）——锁释放后的二次回滚会与下一个
        # 发布者交错（正是 040 的竞态窗口），此处跳过；未进锁即失败的
        # 路径（staging/scorer/generation rename）published 为空，回滚
        # 天然为 no-op
        # kimi3（1404）：发布事务已成功提交（committed）后发生的报告型
        # I/O 错误（stdout broken pipe 等）不是发布失败——不回滚、不删
        # generation，failure receipt 如实记录 publish_committed
        if rollback_state.get("committed"):
            rollback_errors = []
        elif rollback_state["done"]:
            rollback_errors = rollback_state["errors"]
        else:
            rollback_errors = []
            for dst in published:
                bak = backups.get(dst)
                try:
                    if bak is None:
                        try:
                            os.remove(dst)
                        except FileNotFoundError:
                            pass
                    else:
                        os.replace(bak, dst)
                except OSError as oe:
                    rollback_errors.append(f"{dst}: {oe}")
        # 残留备份/临时文件清理（_atomic_install 失败时已自清理 tmp，
        # 这里对全部公开路径兜底）。044：回滚存在错误（混合代际被
        # loudly 拒绝）时，剩余备份是旧代际的唯一恢复路径——保留并在
        # failure receipt 记录 retained_backups，不得删除
        keep_backups = bool(rollback_errors)
        for dst in backups:
            bak = backups[dst]
            if bak is not None and os.path.exists(bak):
                if keep_backups:
                    rollback_state.setdefault(
                        "retained_backups", []).append(bak)
                    continue
                try:
                    os.remove(bak)
                except OSError:
                    pass
            tmp = f"{dst}.tmp-{run_id}"
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        fpath = f"{args.out}.failure-{run_id}.json"
        try:
            d = os.path.dirname(fpath)
            if d:
                os.makedirs(d, exist_ok=True)
            json.dump({
                "run_id": run_id,
                "status": "failed",
                "generated": datetime.now().isoformat(),
                "error": msg,
                "argv": sys.argv[1:],
                "publish_protocol": "e116i-generation-v3",
                # kimi3（1404）：committed=True 表示发布事务已原子完成，
                # 本 failure receipt 记录的是发布后的报告型错误而非发布
                # 失败——产物与 generation 均已保留，消费以 receipt 为准
                "publish_committed": bool(rollback_state.get("committed")),
                "gc_pending": rollback_state.get("gc_pending", []),
                # 044：回滚有错误时保留的备份（旧代际唯一恢复路径）
                "retained_backups": rollback_state.get("retained_backups", []),
                "rollback": {
                    "attempted": bool(published) and
                    not rollback_state.get("committed"),
                    "n_restored": (
                        0 if rollback_state.get("committed")
                        else rollback_state["n_restored"]
                        if rollback_state["done"]
                        else len(published) - len(rollback_errors)),
                    "errors": rollback_errors,
                    # E116g（040）：True = 回滚发生在发布锁内（镜像
                    # 安装段失败）；False = 未进锁即失败（无镜像被触碰）
                    "in_publish_lock": rollback_state["done"],
                },
                "note": (
                    "发布事务已原子提交（generation + 四兼容镜像在位），"
                    "本 failure 记录的是发布后的报告阶段错误（如 stdout "
                    "broken pipe）——产物已保留，请以最新 *.receipt.json "
                    "为准"
                    if rollback_state.get("committed") else
                    ("本轮失败且未完整回滚，以下公开镜像未还原到旧代际"
                     f"（混合代际，loudly 拒绝，须人工核对）: {rollback_errors}"
                     if rollback_errors else
                     "本轮失败，未发布任何产物（发布阶段失败时已回滚"
                     "全部已替换的公开镜像至旧代际）；输出路径下既有"
                     "产物（如有）属于上一轮成功运行，请以最新 "
                     "*.receipt.json 为准")),
                "staging_cleaned": True,
                "generation_cleaned": not bool(
                    rollback_state.get("committed")),
            }, open(fpath, "w"), indent=1, ensure_ascii=False)
            print(f"[formal] failure receipt 落盘: {fpath}", file=sys.stderr)
        except OSError:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        # kimi3（1404）：committed 后 run_dir 是 receipt 的 derived_dir
        # 单指针目标（补刻副本/scorer manifest），不得删除
        if not rollback_state.get("committed"):
            shutil.rmtree(run_dir, ignore_errors=True)
        if rollback_state.get("committed"):
            # 发布已成功，报告型错误不影响发布有效性——rc=0（failure
            # receipt 仍落盘备查）
            sys.exit(0)
        if isinstance(e, SystemExit):
            code = e.code if isinstance(e.code, int) and e.code != 0 else 1
        else:
            code = 1
        sys.exit(code)


if __name__ == "__main__":
    main()
