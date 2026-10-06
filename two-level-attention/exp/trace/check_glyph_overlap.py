# glyph_overlap 检测器：论文图/页面字符级 bbox 重叠检测（读者 agent L1 版式检查的自动化升级）
# 背景：用户 2026-10-06 发现架构图与可视化图存在字重叠/图例遮挡，此前读者 agent 只查
# 文本层（tofu/badref/数字计数），未做 glyph 级检测。本脚本固化为读者 agent 每轮必跑工具。
# 用法：
#   python3 check_glyph_overlap.py <file.pdf> [file2.pdf ...]   # 单/多文件
#   python3 check_glyph_overlap.py --figs                       # 扫 sglang/paper/figures/*.pdf
#   python3 check_glyph_overlap.py --paper                      # 扫双语论文 PDF 全页
# 判定：两不同 span 的字符 bbox 重叠面积占较小者 >45% 记为重叠对（排除同行相邻字符微碰）。
# 退出码：发现重叠返回 1（供 CI/agent 判定），干净返回 0。
# 注意：Poisson 假阳性主要来自 legend 多行换行——报告 span 全文供人工/agent 分辨；
#      修图后必须重跑本脚本至 ov 阈值以下才算闭环。
# --paper 全页模式的已知假阳性（2026-10-06 定案）：中文版正文「CJK×拉丁」同行相邻对的
#      advance 框交叠（'：'x'sink' 等）与数学下标紧排（'∑'x'd'）——英文版同图同内容
#      0 对即证明非视觉压字；正文报警不构成修复项，图件（--figs/显式文件）才是权威口径。
import sys
import glob
import pymupdf

THRESH = 0.45  # 重叠面积占较小字符的比例阈值


def glyph_overlaps(path, thresh=THRESH):
    doc = pymupdf.open(path)
    issues = []
    for pno, page in enumerate(doc):
        d = page.get_text("rawdict")
        chars = []
        for block in d["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    txt = "".join(c["c"] for c in span["chars"])
                    for ch in span["chars"]:
                        if ch["c"].strip():
                            chars.append((pymupdf.Rect(ch["bbox"]), ch["c"], txt))
        rs = sorted(chars, key=lambda s: (s[0].x0, s[0].y0))
        for i in range(len(rs)):
            r1, c1, t1 = rs[i]
            for j in range(i + 1, min(i + 80, len(rs))):
                r2, c2, t2 = rs[j]
                if r2.x0 > r1.x1:
                    break
                inter = r1 & r2
                if inter.is_empty:
                    continue
                a1, a2 = r1.get_area(), r2.get_area()
                if a1 <= 0 or a2 <= 0:
                    continue
                ov = inter.get_area() / min(a1, a2)
                if ov > thresh and t1 != t2:
                    issues.append((pno, c1, c2, round(ov, 2), t1[:14], t2[:14]))
    # 按 span 对去重（保留最大重叠度）
    uniq = {}
    for pno, c1, c2, ov, t1, t2 in issues:
        key = (t1, t2) if t1 <= t2 else (t2, t1)
        if key not in uniq or ov > uniq[key][3]:
            uniq[key] = (pno, c1, c2, ov, t1, t2)
    return list(uniq.values())


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    if args[0] == "--figs":
        files = sorted(glob.glob("/home/wangyuanshuo02/sglang/paper/figures/*.pdf"))
    elif args[0] == "--paper":
        files = ["/home/wangyuanshuo02/sglang/paper/TLI_paper.pdf",
                 "/home/wangyuanshuo02/sglang/paper/TLI_paper_en.pdf"]
    else:
        files = args
    total = 0
    for f in files:
        iss = glyph_overlaps(f)
        if iss:
            total += len(iss)
            print(f"== {f} ({len(iss)} 对) ==")
            for pno, c1, c2, ov, t1, t2 in iss[:10]:
                print(f"  p{pno+1}: '{t1}' x '{t2}' ov={ov}")
    print(f"总计: {total} 对字重叠（阈值 {THRESH}，{len(files)} 文件）")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
