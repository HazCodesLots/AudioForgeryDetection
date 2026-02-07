
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from rawgat_st_model import RawGAT_ST
from dataloader import get_dataloader


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ratio=9.0):
        '''
        Weighted Cross Entropy Loss
        Args:
            weight_ratio: bonafide:spoof ratio (default 9:1 as per paper)
        '''
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight_ratio = weight_ratio

    def forward(self, predictions, targets):
        weights = torch.ones_like(targets, dtype=torch.float)
        weights[targets == 1] = self.weight_ratio  # bonafide
        weights[targets == 0] = 1.0  # spoof

        loss = nn.functional.cross_entropy(predictions, targets, reduction='none')
        weighted_loss = (loss * weights).mean()

        return weighted_loss


def compute_eer(scores, labels):
    '''
    Compute Equal Error Rate (EER)
    Args:
        scores: prediction scores
        labels: ground truth labels (1=bonafide, 0=spoof)
    '''
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100


def compute_class_accuracies(predictions, labels):
    '''
    Compute separate accuracies for bonafide and spoof
    '''
    _, predicted = predictions.max(1)

    # Bonafide accuracy (label=1)
    bonafide_mask = labels == 1
    if bonafide_mask.sum() > 0:
        bonafide_acc = (predicted[bonafide_mask] == labels[bonafide_mask]).float().mean().item() * 100
    else:
        bonafide_acc = 0.0

    # Spoof accuracy (label=0)
    spoof_mask = labels == 0
    if spoof_mask.sum() > 0:
        spoof_acc = (predicted[spoof_mask] == labels[spoof_mask]).float().mean().item() * 100
    else:
        spoof_acc = 0.0

    return bonafide_acc, spoof_acc


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, accum_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_bonafide_acc = 0.0
    total_spoof_acc = 0.0
    batch_count = 0
    max_grad_norm = 0.0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (audio, labels, _) in enumerate(pbar):
        audio = audio.to(device)
        labels = labels.to(device)
        
        outputs = model(audio)
        loss = criterion(outputs, labels)
        
        loss = loss / accum_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f'\n  NaN/Inf loss detected at batch {batch_idx}, skipping')
            optimizer.zero_grad()
            continue
        
        loss.backward()
        
        if (batch_idx + 1) % accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f'\n  Gradient explosion at batch {batch_idx}! Skipping update.')
                optimizer.zero_grad()
                continue
            
            max_grad_norm = max(max_grad_norm, grad_norm.item())
            
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += loss.item() * accum_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        bonafide_acc, spoof_acc = compute_class_accuracies(outputs, labels)
        total_bonafide_acc += bonafide_acc
        total_spoof_acc += spoof_acc
        batch_count += 1
        
        pbar.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'bon': f'{bonafide_acc:.1f}%',
            'spf': f'{spoof_acc:.1f}%',
            'gnorm': f'{max_grad_norm:.2f}'
        })
    
    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    avg_bonafide_acc = total_bonafide_acc / batch_count
    avg_spoof_acc = total_spoof_acc / batch_count
    
    return epoch_loss, epoch_acc, avg_bonafide_acc, avg_spoof_acc



def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_scores = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for audio, labels, _ in tqdm(val_loader, desc='Validating'):
            audio = audio.to(device)
            labels = labels.to(device)

            outputs = model(audio)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            scores = outputs[:, 1] - outputs[:, 0]
            val_scores.extend(scores.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    eer = compute_eer(all_scores, all_labels)

    bonafide_mask = all_labels == 1
    spoof_mask = all_labels == 0

    bonafide_acc = (all_predictions[bonafide_mask] == all_labels[bonafide_mask]).mean() * 100 if bonafide_mask.sum() > 0 else 0
    spoof_acc = (all_predictions[spoof_mask] == all_labels[spoof_mask]).mean() * 100 if spoof_mask.sum() > 0 else 0

    return val_loss, val_acc, bonafide_acc, spoof_acc, eer


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = RawGAT_ST(num_classes=2).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1

    if args.resume and os.path.exists(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f'Resuming from epoch {start_epoch}')

    print('\nLoading training data...')
    train_loader = get_dataloader(
        data_dir=args.train_data,
        protocol_file=args.train_protocol,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        is_eval=False
    )

    print('\nLoading development data...')
    val_loader = get_dataloader(
        data_dir=args.dev_data,
        protocol_file=args.dev_protocol,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        is_eval=False
    )

    criterion = WeightedCrossEntropyLoss(weight_ratio=9.0)
    
    total_grad_norms = []


    metrics_log = {
        'config': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'weight_ratio': 9.0,
            'num_workers': args.num_workers,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'epochs': []
    }

    best_eer = float('inf')
    best_loss = float('inf')

    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f'\n{"="*70}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*70}')

        train_loss, train_acc, train_bonafide_acc, train_spoof_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, accum_steps=4
        )

        val_loss, val_acc, val_bonafide_acc, val_spoof_acc, eer = validate(
            model, val_loader, criterion, device
        )

        print(f'\n EPOCH {epoch} SUMMARY:')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Bonafide Acc: {train_bonafide_acc:.2f}%')
        print(f'Spoof Acc: {train_spoof_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%')
        print(f'Bonafide Acc: {val_bonafide_acc:.2f}%')
        print(f'Spoof Acc: {val_spoof_acc:.2f}%')
        print(f'EER: {eer:.2f}%')

        epoch_metrics = {
            'epoch': epoch,
            'train': {
                'loss': float(train_loss),
                'accuracy': float(train_acc),
                'bonafide_accuracy': float(train_bonafide_acc),
                'spoof_accuracy': float(train_spoof_acc)
            },
            'validation': {
                'loss': float(val_loss),
                'accuracy': float(val_acc),
                'bonafide_accuracy': float(val_bonafide_acc),
                'spoof_accuracy': float(val_spoof_acc),
                'eer': float(eer)
            }
        }
        metrics_log['epochs'].append(epoch_metrics)

        json_path = os.path.join(args.save_dir, 'training_metrics.json')
        with open(json_path, 'w') as f:
            json.dump(metrics_log, f, indent=2)


        if val_loss < best_loss:
            best_loss = val_loss
            best_eer = eer
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'eer': eer,
                'bonafide_acc': val_bonafide_acc,
                'spoof_acc': val_spoof_acc
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'\n Saved best model (loss: {val_loss:.4f}, EER: {eer:.2f}%)')

        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'eer': eer,
                'bonafide_acc': val_bonafide_acc,
                'spoof_acc': val_spoof_acc
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))
            print(f' Saved checkpoint at epoch {epoch}')

    metrics_log['config']['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metrics_log['config']['best_loss'] = float(best_loss)
    metrics_log['config']['best_eer'] = float(best_eer)

    with open(json_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)


    print(f' TRAINING COMPLETED!')
    print(f'Best validation loss: {best_loss:.4f}')
    print(f'Best EER: {best_eer:.2f}%')
    print(f'Metrics saved to: {json_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RawGAT-ST model')

    parser.add_argument('--train_data', type=str, 
                       default=r'N:\ASVspoof2019\LA\ASVspoof2019_LA_train\flac',
                       help='Path to training audio files')
    parser.add_argument('--train_protocol', type=str,
                       default=r'N:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt',
                       help='Path to training protocol file')
    parser.add_argument('--dev_data', type=str,
                       default=r'N:\ASVspoof2019\LA\ASVspoof2019_LA_dev\flac',
                       help='Path to development audio files')
    parser.add_argument('--dev_protocol', type=str,
                       default=r'N:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt',
                       help='Path to development protocol file')

    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')

    parser.add_argument('--save_dir', type=str, default='./checkpoints_aug', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='./logs_aug', help='Directory for tensorboard logs')
    parser.add_argument('--save_interval', type=int, default=1, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')

    args = parser.parse_args()


    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train(args)