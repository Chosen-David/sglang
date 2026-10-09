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
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from benchmark.RULER.score_ruler import (  # noqa: E402
    TASKS, _answers_sha16, _file_sha256, _nlines, _ts_of,
    _validate_manifest_schema,
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


def freeze_and_stage(root, postfix, expect_tasks, stamp, staging,
                     min_samples, data_root):
    """阶段一（E116e 重构）：原始候选只读拷贝进 staging 派生目录 →
    legacy 补刻（派生副本上）→ 长度身份门禁 → min-samples 硬门禁 →
    best-file 仲裁 → 跨 method 身份（ids/answers_sha/lengths 逐行）一致性
    → 源数据 SHA256 绑定 → 冻结 manifest。
    返回 (tasks_manifest, identity_info)。"""
    staged_root = os.path.join(staging, "pred_root")
    tasks_manifest = {}   # {task: {ids, answers_sha, lengths, identity_mode}}
    cells_info = {}       # {key: {length_dir, identity_mode, tasks: {...}}}
    src_data_sha = {}     # {"{Lnum}/{task}": {path, sha256}}
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
            files = sorted(glob.glob(
                os.path.join(pred_dir, f"{task}-*.jsonl")))
            files = [f for f in files
                     if not os.path.basename(f).endswith("-merged.jsonl")]
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
                # ---- 036：只读拷贝 + legacy 补刻（派生副本）----
                staged_files = []
                stamped_flags = {}
                for f in gfiles:
                    dst = os.path.join(staged_root, Lname, f"pred{postfix}",
                                       os.path.basename(f))
                    stamped, rows = _stamp_or_copy(f, dst, task, stamp=stamp)
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
                cells_info.setdefault(key, {
                    "length_dir": Lnum,
                    "identity_mode": identity_mode,
                    "tasks": {},
                })["tasks"][task] = {
                    "n": len(rows),
                    "best_file": os.path.basename(best),
                    "source_path": os.path.abspath(
                        # best 与 staged 同名，映射回原始源文件
                        os.path.join(pred_dir, os.path.basename(best))),
                    "source_sha256": _file_sha256(os.path.join(
                        pred_dir, os.path.basename(best))),
                    "derived_sha256": _file_sha256(best),
                }
    if expect_tasks > 0 and len(tasks_manifest) < expect_tasks:
        missing = sorted(set(TASKS[:expect_tasks]) - set(tasks_manifest))
        _fail(f"root={root} postfix={postfix}: manifest 只覆盖 "
              f"{len(tasks_manifest)}/{expect_tasks} 个任务（缺 {missing}）"
              f"——fail closed")
    if n_stamped:
        print(f"[formal] 共 {n_stamped} 个 legacy 文件在派生副本上补刻身份"
              f"（_id/_answers_sha；源文件零改动，逐字段不变量已断言）")
    _validate_manifest_schema(tasks_manifest)
    return tasks_manifest, cells_info, src_data_sha


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
                    help="声明生成时启用 YaRN（身份绑定）")
    ap.add_argument("--yarn-factor", type=float, default=None,
                    help="声明 YaRN factor（身份绑定）")
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

        # ---- 阶段一：staging 派生 + 冻结 manifest（含身份扩展 032）----
        tasks_manifest, cells_info, src_data_sha = freeze_and_stage(
            args.root, args.pred_postfix, args.expect_tasks,
            stamp=not args.no_stamp_legacy_ids, staging=staging,
            min_samples=args.min_samples, data_root=args.data_root)

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
                "yarn": bool(args.yarn),
                "yarn_factor": args.yarn_factor,
                "extra_params": extra_params,
                "formal_script_sha256": _file_sha256(FORMAL_PATH),
                "scorer_sha256": _file_sha256(SCORER_PATH),
                # legacy 声明口径：model/yarn 等为操作者事后声明，
                # 可能不可恢复——以 receipt/manifest 记录为准，不冒充完整
                "note": ("model_path/yarn 等为操作者声明值；legacy 数据无法"
                         "从文件恢复完整输入身份（identity_mode=legacy-"
                         "partial），native 数据由 pred_ruler.py 落盘"),
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
