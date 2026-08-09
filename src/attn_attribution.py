"""注意力权重归因 (创新点3, 修复 v1 失败的偏差向量归因)
用编码器内 VariableSelfAttention 的注意力权重做变量级归因,
用 SMD interpretation_label 验证."""
import torch
import numpy as np


def attention_attribution(model, x_instance, D):
    """对单个异常 instance 计算变量级归因.
    x_instance: (1, l, D) 单个 instance
    返回: (D,) 每个变量的归因分数(越高越可疑)."""
    model.eval()
    with torch.no_grad():
        # 走前向, 提取变量自注意力的权重
        # InstanceEncoder 的 forward 里 VariableSelfAttention 输出 attn
        # 我们需要 model 返回注意力权重
        z, attn_weights = model.encode_with_attention(x_instance)
        # attn_weights: (1, D, D) 变量间注意力
        # 归因 = 每个变量被其他变量关注的总和 (入度)
        attribution = attn_weights[0].sum(dim=0)  # (D,) 列求和=入度
    return attribution.cpu().numpy()


def evaluate_attn_attribution(model, X_test, Y_test, events, D, top_k=3):
    """用真实标签(SMD interpretation_label)评估注意力归因 Hit@k.
    events: list of (start, end, [true_variable_indices]) from interpretation_label.
    返回 Hit@k."""
    hits = []
    for (start, end, true_vars) in events:
        # 取该事件区间内的 instance
        mid = (start + end) // 2
        if mid >= len(X_test):
            continue
        instance = X_test[mid:mid+1]  # (1, l, D) 假设已切成 instance
        if instance.shape[0] == 0:
            continue
        try:
            attr = attention_attribution(model, torch.from_numpy(instance).float(), D)
            topk_pred = set(np.argsort(-attr)[:top_k].tolist())
            hit = len(topk_pred & set(true_vars)) > 0
            hits.append(hit)
        except Exception as e:
            continue
    return float(np.mean(hits)) if hits else float('nan')
