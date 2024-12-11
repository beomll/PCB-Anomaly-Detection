import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import wandb

import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scripts.eval import evaluate


def train(model, device, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs, save_path):
    min_loss = np.Inf
    patience = 0

    os.makedirs(save_path, exist_ok=True)

    train_losses = list()
    train_pred = list()
    train_true = list()
    val_losses = list()

    for epoch in range(num_epochs):
        with tqdm(train_loader, unit='b', ascii=True, ncols=150, desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            model.train()
            train_loss = 0
            for (img, label) in train_loader:
                optimizer.zero_grad()

                img = img.to(device)
                label = label.to(device).float()

                outputs = model(img)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                if scheduler is not None and not (isinstance(scheduler, ReduceLROnPlateau) or isinstance(scheduler, MultiStepLR)):
                    scheduler.step()

                train_pred.extend(outputs.detach().cpu().numpy())
                train_true.extend(label.detach().cpu().numpy())

                train_loss += loss.item()

                pbar.set_postfix_str(f'loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]["lr"]:.2e}')
                pbar.update(1)

        train_losses.append(train_loss / len(train_loader))
        train_preds = [1 if pred > 0.5 else 0 for pred in train_pred]
        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds)
        train_precision = precision_score(train_true, train_preds)
        train_recall = recall_score(train_true, train_preds)

        wandb.log({'epoch': epoch + 1,
                   'train/loss': train_loss / len(train_loader),
                   'train/accuracy': train_acc,
                   'train/f1': train_f1,
                   'train/precision': train_precision,
                   'train/recall': train_recall
                   })

        val_loss, acc, f1, precision, recall = evaluate(model, device, criterion, val_loader)
        val_losses.append(val_loss)
        if val_loss < min_loss:
            print(f'Best model saved at epoch {epoch + 1}')
            print(f'Validation loss decreased ({min_loss:.4f} --> {val_loss:.4f})')
            print(f'Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

            min_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model_ep_{epoch + 1}.pth')
            patience = 0
        else:
            patience += 1
            if patience >= num_epochs * 0.1:
                print('Early stopping')
                break

        if scheduler is not None and isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None and isinstance(scheduler, MultiStepLR):
            scheduler.step()

        wandb.log({'epoch': epoch + 1,
                   'eval/loss': val_loss,
                   'eval/accuracy': acc,
                   'eval/f1': f1,
                   'eval/precision': precision,
                   'eval/recall': recall
                   })

    torch.save(model.state_dict(), f'{save_path}/last_model_ep_{num_epochs}.pth')

    return train_losses, val_losses
