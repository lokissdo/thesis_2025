import torch
import torch.nn.functional as F

def cw_attack(model, x, y, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=lr)

    for step in range(max_iter):
        outputs = model(x_adv)
        real = torch.sum(y * outputs, dim=1)
        other = torch.max((1 - y) * outputs - y * 1e4, dim=1)[0]
        loss1 = torch.clamp(other - real + kappa, min=0).sum()
        loss2 = torch.norm((x_adv - x).view(x.size(0), -1), p=2, dim=1).sum()
        loss = loss1 + c * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return x_adv.detach()

def cw_attack_smile_masked(model, x, y_target, mask=None, c=1e-4, kappa=0, max_iter=1000, lr=0.01):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=lr)
    target_value = y_target.item()
    loss_fn = torch.nn.BCELoss()

    for step in range(max_iter):
        outputs = model(x_adv)[:, 31].view(-1, 1)
        loss1 = loss_fn(outputs, y_target)
        loss2 = torch.norm((x_adv - x).view(x.size(0), -1), p=2, dim=1).sum()
        loss = loss1 + c * loss2

        optimizer.zero_grad()
        if mask is not None:
            grad = torch.autograd.grad(loss, x_adv, create_graph=True)[0]
            grad = grad * mask
            x_adv = x_adv - lr * grad
            x_adv = x_adv.detach().requires_grad_(True)
        else:
            loss.backward()
            optimizer.step()

        current_score = model(x_adv)[:, 31].item()
        if (target_value == 1.0 and current_score >= 0.5) or (target_value == 0.0 and current_score <= 0.5):
            break

    return x_adv.detach()