import os
import random
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from model.SimpleNet import SimpleNet
from scripts.train import train
from scripts.eval import test
from utils.Dataset import AnomalyDataset
from utils.selec_opt_sch import select_optimizer, select_scheduler

import matplotlib.pyplot as plt
os.environ["WANDB_PROJECT"] = "simplenet"
os.environ["WANDB_LOG_MODEL"] = "logs"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    random.seed(seed)


def main(args):
    lr = args.learning_rate
    model_type = args.model_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    sch = args.scheduler
    opt = args.optimizer
    warmup_steps = args.warmup_steps
    gamma = args.gamma
    dropout = args.drop_prob
    step_size = args.step_size
    train_data = args.train_data
    val_data = args.val_data
    test_data = args.test_data
    pretrained = args.pretrained
    model_path = args.model_path
    save_path = args.save_path

    seed_everything(args.seed)
    wandb.init(project="simplenet", name=f'simplenet_{num_epochs}_{lr}_{opt}_{model_type}')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform_train = v2.Compose([v2.Resize((224, 224)),
                                  v2.RandomHorizontalFlip(p=0.5),
                                  v2.RandomRotation(degrees=15),
                                  v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
                                  v2.ToImage(),
                                  v2.ToDtype(torch.float32, scale=True)
                                  ])

    transform_test = v2.Compose([v2.Resize((224, 224)),
                                 v2.ToImage(),
                                 v2.ToDtype(torch.float32, scale=True)
                                 ])

    train_dataset = AnomalyDataset(data_path=train_data, transform=transform_train)
    val_dataset = AnomalyDataset(data_path=val_data, transform=transform_test)
    test_dataset = AnomalyDataset(data_path=test_data, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f'Train data: {len(train_dataset)}')
    print(f'Val data: {len(val_dataset)}')
    print(f'Test data: {len(test_dataset)}')

    model = SimpleNet(pretrained=pretrained, path=model_path, dropout=dropout, model_type=model_type).to(device)
    num_of_data_false = len(os.listdir(os.path.join(train_data, 'False'))) + len(os.listdir(os.path.join(val_data, 'False')))
    num_of_data_true = len(os.listdir(os.path.join(train_data, 'True'))) + len(os.listdir(os.path.join(val_data, 'True')))

    pos_weight = torch.tensor([num_of_data_false / num_of_data_true], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = select_optimizer(model, opt, lr)
    scheduler = select_scheduler(optimizer, gamma, warmup_steps, sch, step_size)

    train_losses, val_losses = train(model=model,
                                     device=device,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     num_epochs=num_epochs,
                                     save_path=save_path
                                     )

    test(model=model,
         device=device,
         test_loader=test_loader,
         criterion=criterion,
         save_path=save_path
         )

    os.makedirs('logs', exist_ok=True)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.legend()
    plt.savefig('logs/loss.png')
    plt.close()
