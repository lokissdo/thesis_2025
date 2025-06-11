import torch

def linf_proj(x_orig, x_adv, epsilon):
    return torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

def fgsm_attack(model, x, y, epsilon=0.03, loss_fn=None):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    outputs = model(x_adv)
    loss = loss_fn(outputs, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = x_adv + epsilon * grad.sign()
    x_adv = linf_proj(x, x_adv, epsilon)
    return x_adv.detach()

def fgsm_attack_smile_masked(model, x, y_target, mask=None, epsilon=0.03):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    loss_fn = torch.nn.BCELoss()
    target_value = y_target.item()

    outputs = model(x_adv)[:, 31].view(-1, 1)
    loss = loss_fn(outputs, y_target)
    grad = torch.autograd.grad(loss, x_adv)[0]
    if mask is not None:
        grad = grad * mask
    x_adv = x_adv - epsilon * grad.sign()
    x_adv = linf_proj(x, x_adv, epsilon)

    return x_adv.detach()