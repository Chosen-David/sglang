import torch
from einops import rearrange

class AttnWeightsRecorder:

    def __init__(self) -> None:
        self.layer_dict = {}
        self.max_layer_idx = -1
        self.min_stride = 16
        self.max_stride = 64
    
    def update(self, layer_idx: int, attn_weights: torch.Tensor, group_size: int):
        # attn_weights: [b, h, qt, kt]
        b, h, qt, kt = attn_weights.shape
        attn_weights = rearrange(attn_weights, 'b (h g) qt kt -> b h g qt kt', g=group_size).mean(dim=2)
        self.max_layer_idx = max(self.max_layer_idx, layer_idx)

        assert b == 1

        # 生成 [qt, kt] 的差分矩阵 diff = i_q - i_k
        q_idx = torch.arange(qt, device=attn_weights.device).view(qt, 1)      # [qt, 1]
        k_idx = torch.arange(kt, device=attn_weights.device).view(1, kt)      # [1, kt]
        diff = q_idx - k_idx                                             # [qt, kt]

        for stride in range(self.min_stride, self.max_stride + 1):
            if (layer_idx, stride) not in self.layer_dict[layer_idx]:
                self.layer_dict[(layer_idx, stride)] = []

            # 构造 mask: (i_q - i_k) % stride == 0
            # 注意 PyTorch 中 % 在负数下也定义良好，对这个条件是合理的
            mask = (diff % stride == 0) & (q_idx >= k_idx)

            # 扩展到 [1, 1, qt, kt]，以便广播到 [b, h, qt, kt]
            mask = mask.to(attn_weights).unsqueeze(0).unsqueeze(0)          # [1, 1, qt, kt]

            # 被选中的元素加总
            masked_weights = attn_weights * mask                            # [b, h, qt, kt]
            sum_vals = masked_weights.sum(dim=(0, -1, -2))                  # [h]

            # 选中元素的个数（对每个 (b,h) 一样，直接用 mask_sum）
            mask_sum = mask.sum(dim=(0, -1, -2))                            # [1]
            # 为安全起见，防止除 0
            eps = 1e-9
            mean_vals = sum_vals / (mask_sum + eps)                         # [h]
            self.layer_dict[(layer_idx, stride)].append(mean_vals.cpu())
    
    def get_record(self):
        results = []
        for layer_idx in range(self.max_layer_idx + 1):
            all_strides = []
            for stride in range(self.min_stride, self.max_stride):
                attn_weights = torch.stack(self.layer_dict[(layer_idx, stride)]).mean(dim=0)
                all_strides.append(attn_weights)
            stride_id = torch.stack(all_strides).argmax(dim=0) + self.min_stride # [h]
            results.append(stride_id)
        results = torch.stack(stride_id).numpy() # [num_layers, h]
        return results

    def clear(self):
        self.layer_dict.clear()
