import torch

def linf_proj(x_orig, x_adv, epsilon):
    return torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

def l2_proj(x_orig, x_adv, epsilon):
    delta = x_adv - x_orig
    norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
    norm = torch.max(norm, torch.ones_like(norm) * 1e-12)
    factor = torch.min(torch.ones_like(norm), epsilon / norm)
    delta = delta * factor.view(-1, 1, 1, 1)
    return x_orig + delta

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def pgd_attack(model, x, y, epsilon=0.03, step_size=0.01, nb_iter=40, norm='linf', loss_fn=None):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(nb_iter):
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        if norm == 'linf':
            x_adv = x_adv + step_size * grad.sign()
            x_adv = linf_proj(x, x_adv, epsilon)
        elif norm == 'l2':
            grad_norm = grad / (grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)
            x_adv = x_adv + step_size * grad_norm
            x_adv = l2_proj(x, x_adv, epsilon)
        else:
            raise ValueError("Unsupported norm type. Use 'linf' or 'l2'.")
        x_adv = x_adv.detach().requires_grad_(True)
    return x_adv.detach()

def pgd_attack_smile_masked(model, x, y_target, mask=None, epsilon=0.03, step_size=0.01, nb_iter=40, norm='linf'):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss_fn = torch.nn.BCELoss()
    target_value = y_target.item()
    for i in range(nb_iter):
        outputs = model(x_adv)[:, 31].view(-1, 1)
        loss = loss_fn(outputs, y_target)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        if mask is not None:
            grad = grad * mask
        if norm == 'linf':
            x_adv = x_adv - step_size * grad.sign()
            x_adv = linf_proj(x, x_adv, epsilon)
        elif norm == 'l2':
            grad_norm = grad / (grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(-1, 1, 1, 1) + 1e-12)
            x_adv = x_adv - step_size * grad_norm
            x_adv = l2_proj(x, x_adv, epsilon)
        else:
            raise ValueError("Unsupported norm type. Use 'linf' or 'l2'.")
        x_adv = x_adv.detach().requires_grad_(True)
        current_score = model(x_adv)[:, 31].item()
        if (target_value == 1.0 and current_score >= 0.5) or (target_value == 0.0 and current_score <= 0.5):
            break
    return x_adv.detach()