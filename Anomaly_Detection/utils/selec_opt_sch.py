import torch.optim as optim
import torch.optim.lr_scheduler as sch


def select_optimizer(model, opt, lr):
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
    elif opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0005)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise ValueError(f'Invalid optimizer: {opt}')

    return optimizer


def select_scheduler(optimizer, gamma, warmup_steps, scheduler_type, step_size):
    if scheduler_type == 'cosine_warmup':
        scheduler = sch.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=1, eta_min=0)
    elif scheduler_type == 'reduce':
        scheduler = sch.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
    elif scheduler_type == 'step':
        scheduler = sch.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'multi_step':
        scheduler = sch.MultiStepLR(optimizer, milestones=[150, 225], gamma=gamma)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f'Invalid scheduler: {scheduler_type}')

    return scheduler
