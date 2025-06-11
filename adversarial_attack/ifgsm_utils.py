import torch

def linf_proj(x_orig, x_adv, epsilon):
    return torch.clamp(x_adv, x_orig - epsilon, x_orig + epsilon)

def ifgsm_attack(model, x, y, epsilon=0.03, step_size=0.01, nb_iter=10, loss_fn=None):
    model.eval()
    x_adv = x.clone().detach()
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(nb_iter):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = loss_fn(outputs, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + step_size * grad.sign()
        x_adv = linf_proj(x, x_adv, epsilon)
        x_adv = x_adv.detach()
    return x_adv

def ifgsm_attack_smile_masked(model, x, y_target, mask=None, epsilon=0.03, step_size=0.01, nb_iter=10):
    model.eval()
    x_adv = x.clone().detach()
    loss_fn = torch.nn.BCELoss()
    target_value = y_target.item()

    for _ in range(nb_iter):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)[:, 31].view(-1, 1)
        loss = loss_fn(outputs, y_target)
        grad = torch.autograd.grad(loss, x_adv)[0]
        if mask is not None:
            grad = grad * mask
        x_adv = x_adv - step_size * grad.sign()
        x_adv = linf_proj(x, x_adv, epsilon)
        x_adv = x_adv.detach()

        current_score = model(x_adv)[:, 31].item()
        if (target_value == 1.0 and current_score >= 0.5) or (target_value == 0.0 and current_score <= 0.5):
            break

    return x_adv