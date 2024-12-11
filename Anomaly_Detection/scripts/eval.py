import os

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch

from tqdm import tqdm


def evaluate(model, device, criterion, val_loader):
    val_loss = 0
    preds = list()
    targets = list()

    with tqdm(val_loader, unit='b', ascii=True, ncols=150, desc=f'Validation') as pbar:
        model.eval()
        with torch.no_grad():
            for (img, label) in val_loader:
                img = img.to(device)
                label = label.to(device).float()

                outputs = model(img)
                loss = criterion(outputs, label)

                preds.extend(outputs.detach().cpu().numpy())
                targets.extend(label.detach().cpu().numpy())

                val_loss += loss.item()

                pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                pbar.update(1)

    preds = [1 if pred > 0.5 else 0 for pred in preds]
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    precision = precision_score(targets, preds)
    recall = recall_score(targets, preds)

    return val_loss / len(val_loader), acc, f1, precision, recall


def test(model, device, criterion, test_loader, save_path):
    last_name = [model_state for model_state in os.listdir(save_path) if 'last' in model_state]
    best_name = [model_state for model_state in os.listdir(save_path) if 'best' in model_state]

    model.load_state_dict(torch.load(f'{save_path}/{last_name[0]}'))
    test_loss, acc, f1, precision, recall = evaluate(model, device, criterion, test_loader)
    print(f'Last model test loss: {test_loss:.4f}, accuracy: {acc:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')

    model.load_state_dict(torch.load(f'{save_path}/{best_name[0]}'))
    test_loss, acc, f1, precision, recall = evaluate(model, device, criterion, test_loader)
    print(f'Best model test loss: {test_loss:.4f}, accuracy: {acc:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')
