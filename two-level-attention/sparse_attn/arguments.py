import argparse

def _add_tia_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='Twi-Index Sparse Attention')
    group.add_argument('--tia_block_size', type=int, default=64)
    group.add_argument('--tia_level1_topk', type=int, default=128)
    group.add_argument('--tia_level2_topk', type=int, default=1024)
    group.add_argument('--tia_level2_cmp_ratio', type=int, default=2, choices=[2, 4, 8])
    group.add_argument('--tia_enable_async_topk', action="store_true")
    return parser

def _add_twilight_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='Twilight Sparse Attention')
    group.add_argument('--twi_block_size', type=int, default=64)
    group.add_argument('--twi_level1_topk', type=int, default=128)
    group.add_argument('--twi_level2_topp', type=float, default=0.95)
    return parser

def _add_quest_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='quest')
    group.add_argument('--quest_block_size', type=int, default=64)
    group.add_argument('--quest_topk', type=int, default=16)
    return parser

def _add_tli_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='TLI (Two-Level Indexer)')
    group.add_argument('--tli_enable_subspace', type=lambda x: str(x).lower() != 'false', default=True)
    group.add_argument('--tli_enable_kmeans', type=lambda x: str(x).lower() != 'false', default=True)
    # ---- E110：ccluster / sim_greedy（用户 2026-10-07 定义，TASK.md「关于cluster的方法」节）----
    #   far_select/near_select 各自独立可选 L2 选择方式：
    #   4bit       = 细筛分数 topk（现状默认，回归保护）
    #   cluster    = kmeans 簇代表打分 topk（ccluster = (cluster, cluster)；cavg = far 侧已有）
    #   sim_greedy = 增量贪心聚类（余弦相似度 >= --tli_sim 归并，簇心=算术均值增量维护）
    group.add_argument('--tli_far_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='far 侧 L2 选择: 4bit=细筛分数(默认) / cluster=kmeans 簇代表 / '
                            'sim_greedy=增量贪心聚类簇代表')
    group.add_argument('--tli_near_select', type=str, default='4bit',
                       choices=['4bit', 'cluster', 'sim_greedy'],
                       help='near 侧 L2 选择: 4bit=细筛分数(默认,现状) / cluster=kmeans 簇代表 / '
                            'sim_greedy=增量贪心聚类簇代表（ccluster 组合 = near 侧开 cluster）')
    group.add_argument('--tli_sim', type=float, default=0.9,
                       help='sim_greedy 余弦相似度归并阈值（簇心-running-mean 与新 token 的 cos >= sim 则归并）')
    group.add_argument('--tli_sim_dims', type=str, default='subspace',
                       choices=['subspace', 'nope', 'tail', 'full'],
                       help='sim_greedy 聚类维度: subspace=与 cluster/kmeans 路径同一子空间(默认,含压缩维) / '
                            'nope=后 64 非旋转维 / tail=低频尾维(cmp_ratio 控宽) / full=全 128 维')
    group.add_argument('--tli_enable_layer_skip', type=lambda x: str(x).lower() != 'false', default=True)
    group.add_argument('--tli_far_clusters', type=int, default=256)
    group.add_argument('--tli_far_niter', type=int, default=10)
    group.add_argument('--tli_far_blocks', type=int, default=16)
    group.add_argument('--tli_far_tokens', type=int, default=512)
    group.add_argument('--tli_layer_skip_path', type=str, default=None)
    # ---- E64 框架参数化（B 配置帕累托点：bp128+sup_wsvd d8+alpha0.125/beta0.25/gamma0.5）----
    group.add_argument('--tli_far_method', type=str, default='minmax', choices=['minmax', 'avg'],
                       help='far 池 L1 粗筛分数源: minmax=块上界(默认) / avg=块均值')
    group.add_argument('--tli_near_method', type=str, default='avg', choices=['avg', 'minmax'],
                       help='near 池 L1 粗筛分数源: avg=块均值(默认) / minmax=块上界')
    group.add_argument('--tli_alpha', type=float, default=0.0,
                       help='near 区占 mid 长度比（E64 alpha；0=单池老逻辑）')
    group.add_argument('--tli_beta', type=float, default=0.0,
                       help='near 块预算占 K1 比（E64 beta）')
    group.add_argument('--tli_gamma', type=float, default=1.0,
                       help='near 细筛 token 折扣（E64 gamma）')
    group.add_argument('--tli_subspace', type=str, default='full',
                       choices=['full', 'rope', 'nope', 'tail', 'random', 'highfreq'],
                       help='子空间选择: full=全128维(默认) / rope=前64旋转维 / nope=后64非旋转维 / tail=旧口径低频尾维32(E71/B7/C0复现)')
    group.add_argument('--tli_proj_basis', type=str, default=None,
                       help='sup_wsvd 投影基 .pt（裸 tensor [n_layers,Hkv,128,r]）')
    # ---- E85f：per-layer 静态 pair 选取（E85e 重放 0.849 vs tail32 0.811 的 e2e 判决）----
    group.add_argument('--tli_static_pair', action='store_true',
                       help='prefill 尾 256 q 统计的 pair 幅值 top-16 完整对作 per-layer '
                            '固定子空间（需 --tli_subspace tail；pair 未定前回退 tail32）')
    # ---- E87：top-σ 选择（用户 2026-10-02 定义：区不做两级，细筛分 ≥ sink 分数 max − σ 即选中）----
    # ---- E89：MoBA 复现臂（统一 harness 内公平实现，training-free chunk gate）----
    group.add_argument('--tli_moba', action='store_true',
                       help='MoBA 复现臂：全维 chunk-mean gate 选 top-K2/BS 块全展开，'
                            '无两级无量化；sink/swa 同口径保送')
    group.add_argument('--tli_sigma_select', type=str, default='none',
                       choices=['none', 'far', 'near', 'mid'],
                       help='top-σ 侧: none=两侧两级(默认) / far=far 区 top-σ+near 两级 / '
                            'near=near 区 top-σ+far 两级 / mid=整个 mid 不分区 top-σ')
    group.add_argument('--tli_sigma', type=float, default=8.0,
                       help='top-σ 阈值宽度（细筛原始分尺度，锚 = sink 分数 max）')
    # ---- E103：kv-head 共享消融（审稿 MAJOR：§2.2 GQA 共享 vs per-q-head 独立选择）----
    group.add_argument('--tli_per_q_head', type=lambda x: str(x).lower() != 'false',
                       default=False,
                       help='E103 消融：旁路 GQA 组内 mean 聚合，per-q-head 独立选择'
                            '（mask [H,T]，索引量 ×G；关闭 = 论文 §2.2 kv-head 共享口径）')
    return parser

def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Attention Arguments',
                                     allow_abbrev=False)
    parser.add_argument('--method', type=str, choices=['quest', 'tia', 'twi', 'tli', 'none'])
    parser = _add_quest_args(parser)
    _add_twilight_args(parser)
    parser = _add_tia_args(parser)
    parser = _add_tli_args(parser)
    return parser.parse_args()

def add_sparse_attn_args(parser: argparse.ArgumentParser):
    parser.add_argument('--method', type=str, choices=['quest', 'tia', 'twi', 'tli', 'none'])
    parser = _add_quest_args(parser)
    parser = _add_twilight_args(parser)
    parser = _add_tia_args(parser)
    parser = _add_tli_args(parser)
    return parser
