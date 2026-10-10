import torch
from einops import rearrange, einsum

class Metrics:

    def __init__(self, threshold: int = 2048) -> None:
        self.threshold = threshold
        self.select_tokens = []
        self.k_max = []
        self.k_min = []
        self.k_delta_max = []
        self.k_delta_mean = []
    
    def clear(self):
        self.select_tokens.clear()
        self.k_delta_max.clear()
        self.k_delta_mean.clear()
    
    def add_select_result(self, 
        q_ids: torch.Tensor, 
        mask: torch.Tensor, 
        block_size: int,
    ):
        mask = mask.squeeze(0) # [tq, h, tk]
        for i in range(q_ids.shape[0]):
            num_tokens = q_ids[i].item() + 1
            if num_tokens >= self.threshold:
                b_mask = mask[i] # [h, tk]
                num_select_tokens = b_mask.sum().item() * block_size / b_mask.shape[0]
                self.select_tokens.append(num_select_tokens)
    
    def get_select_tokens(self):
        if len(self.select_tokens) > 0:
            result = sum(self.select_tokens) / len(self.select_tokens)
        else:
            result = None
        return result
    
    def add_k_delta(self, k: torch.Tensor, block_size: int):
        k = rearrange(k, 'b (t bs) h d -> b t bs h d', bs=block_size)
        k_max = k.amax(dim=2)
        k_min = k.amin(dim=2)
        # 【B6 修复（kimi3 清单 F10，2026-10-08）】块内极差应为减法：
        # 原 (k_max + k_min).abs() 是 min 与 max 绝对值之和（有符号下无
        # 几何意义），按块跨度语义应为 k_max - k_min（极差，与 L1 minmax
        # 上界分数所刻画的「块内散布」一致）。死代码防御修复：当前无
        # 生产者调用本方法，无历史数据口径需要迁移。
        k_delta = k_max - k_min
        k_delta_max  = k_delta.amax(dim=0).amax(dim=1)
        k_delta_mean = k_delta.mean(dim=0).mean(dim=1)
        self.k_delta_max.append(k_delta_max.cpu())
        self.k_delta_mean.append(k_delta_mean.cpu())

    def get_k_delta_max(self):
        if len(self.k_delta_max) > 0:
            k_delta_max = torch.stack(self.k_delta_max).amax(dim=0)
            return k_delta_max.amax()
        else:
            return None

    def get_k_delta_mean(self):
        if len(self.k_delta_mean) > 0:
            k_delta_mean = torch.stack(self.k_delta_mean).mean(dim=0)
            return k_delta_mean.mean()
        else:
            return None

GLOBAL_METRICS = Metrics()

def get_metrics():
    return GLOBAL_METRICS
