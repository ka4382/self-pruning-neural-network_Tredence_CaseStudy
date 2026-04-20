import torch
from model import PrunableLinear

def compute_sparsity_loss(model):
    loss = 0
    count = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 2)
            loss += torch.mean(gates)
            count += 1

    return loss / count


def calculate_sparsity(model, threshold=0.05):
    total = 0
    pruned = 0

    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores * 2)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total
