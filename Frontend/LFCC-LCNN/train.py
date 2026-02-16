import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from LCNN import LCNN


class LFCCLCNNTrainer:
    def __init__(
        self,
        model,
        lfcc_extractor,
        train_loader,
        val_loader,
        device='cuda',
        lr=0.0001,
        weight_decay=0.0001,
        class_weights=None
    ):
        self.model = model.to(device)
        self.lfcc_extractor = lfcc_extractor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.history = {
            'epochs': [],
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_eer': [],
            'learning_rates': []
        }
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using CrossEntropyLoss with class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.scheduler = None
        
    def train_epoch(self, grad_clip=1.0):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for lfcc, labels in pbar:
            lfcc = lfcc.to(self.device)
            labels = labels.to(self.device)
            
            lfcc = lfcc.unsqueeze(1)
            outputs = self.model(lfcc)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, desc='Validation'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for lfcc, labels in tqdm(self.val_loader, desc=desc):
                lfcc = lfcc.to(self.device)
                labels = labels.to(self.device)
                
                lfcc = lfcc.unsqueeze(1)
                outputs = self.model(lfcc)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_scores)
        
        eer = self.calculate_eer(all_labels, all_scores)
        
        return avg_loss, accuracy, auc, eer
    
    def calculate_eer(self, labels, scores):
        """Calculate Equal Error Rate"""
        from scipy.optimize import brentq
        from scipy.interpolate import interp1d
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer * 100


    def save_checkpoint(self, epoch, val_eer, val_acc, save_path, is_best=False):
        import os
        checkpoint_dir = os.path.dirname(save_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'lfcc_extractor_state_dict': self.lfcc_extractor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_eer': val_eer,
            'val_acc': val_acc
        }
        
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1:03d}.pt')
        torch.save(state, epoch_path)
        print(f"Saved epoch {epoch+1} weights to {epoch_path}")
        
        if is_best:
            torch.save(state, save_path)
            print(f"Saved best model with EER: {val_eer:.4f}%")
            
    def save_metrics(self, save_path):
        import json
        import os
        
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        metrics_dir = os.path.dirname(save_path)
        
        simple_metrics = {
            'train_loss': self.history['train_loss'],
            'train_acc': self.history['train_acc'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc'],
            'val_auc': self.history['val_auc'],
            'val_eer': self.history['val_eer']
        }
        
        simple_path = os.path.join(metrics_dir, 'training_metrics.json')
        with open(simple_path, 'w') as f:
            json.dump(simple_metrics, f, indent=4)
        
        detailed_metrics = {
            "epochs": [
                {
                    "epoch": i+1,
                    "learning_rate": self.history['learning_rates'][i],
                    "train_accuracy": self.history['train_acc'][i],
                    "val_accuracy": self.history['val_acc'][i],
                    "train_loss": self.history['train_loss'][i],
                    "val_loss": self.history['val_loss'][i],
                    "val_auc": self.history['val_auc'][i],
                    "val_eer": self.history['val_eer'][i]
                }
                for i in range(len(self.history['train_acc']))
            ]
        }
        
        detailed_path = os.path.join(metrics_dir, f'{base_name}_detailed_metrics.json')
        with open(detailed_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=4)
    
    def train(self, num_epochs=100, save_path='best_lfcc_lcnn.pt', 
              grad_clip=1.0, scheduler_type='cosine'):
        """
        Train with improved scheduler and early stopping
        
        Args:
            num_epochs: Maximum number of epochs
            save_path: Path to save best model
            grad_clip: Gradient clipping threshold (0 to disable)
            scheduler_type: 'cosine' or 'step' or 'plateau'
        """
        
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
            print(f" Using Cosine Annealing LR: {self.optimizer.param_groups[0]['lr']:.6f} → 1e-6")
        elif scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                verbose=True,
                min_lr=1e-6
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
            print(f"Using StepLR (step_size=10, gamma=0.5)")
        else:
            print(f"No scheduler enabled")
        
        print(f"Gradient clipping: {'Enabled (%.1f)' % grad_clip if grad_clip > 0 else 'Disabled'}")
        
        best_eer = float('inf')
        
        for epoch in range(num_epochs):

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} | LR: {current_lr:.6f}")
            
            train_loss, train_acc = self.train_epoch(grad_clip=grad_clip)
            print(f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc*100:.2f}%")
            
            val_loss, val_acc, val_auc, val_eer = self.validate()
            print(f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc*100:.2f}% | "
                  f"Val AUC: {val_auc:.6f} | Val EER: {val_eer:.4f}%")
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_eer'].append(val_eer)
            
            is_best = val_eer < best_eer
            if is_best:
                best_eer = val_eer
                print(f" New best EER: {best_eer:.4f}%")
            
            self.save_checkpoint(epoch, val_eer, val_acc, save_path, is_best=is_best)
            self.save_metrics(save_path)
            
            if self.scheduler is not None:
                if scheduler_type == 'plateau':
                    self.scheduler.step(val_eer)
                else:
                    self.scheduler.step()
            
        
        print(f"\n Training complete! Best EER: {best_eer:.4f}%")
        return best_eer



def main():
    WAVEFAKE_ROOT = r"C:\Users\HazCodes\Documents\Datasets\generated_audio"
    LJSPEECH_ROOT = r"C:\Users\HazCodes\Documents\Datasets\generated_audio\LJSpeech-1.1\wavs"
    BATCH_SIZE = 64
    NUM_EPOCHS = 100
    SAMPLE_RATE = 16000
    MAX_LENGTH = 64000 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from WaveFakeLoader import create_loaders_from_splits
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    train_loader, test_loader = create_loaders_from_splits(
        splits_json=splits_json,
        vocoders_train=None, # All vocoders
        vocoders_test=None,  # All vocoders
        batch_size=BATCH_SIZE,
        num_workers=4,
        noise_std=0.001
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("Failed to initialize dataloaders or dataset is empty. Check your paths.")
        return


    print("\n[DATA SPLIT VERIFICATION]")
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + test_size
    train_ratio = train_size / total_size
    test_ratio = test_size / total_size
    print(f"Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Test:  {test_size} ({test_ratio*100:.1f}%)")
    
    if abs(train_ratio - 0.8) > 0.05 or abs(test_ratio - 0.2) > 0.05:
        print("WARNING: Not a proper 80/20 split!")



    train_dataset = train_loader.dataset
    fake_count = train_dataset.fake_count
    real_count = train_dataset.real_count
    total = fake_count + real_count
    
    print(f"\nCalculating class weights (Total: {total}):")
    print(f"  Fake samples: {fake_count}")
    print(f"  Real samples: {real_count}")
    
    weight_fake = total / (2 * fake_count)
    weight_real = total / (2 * real_count)
    class_weights = torch.tensor([weight_fake, weight_real], dtype=torch.float32).to(device)
    print(f"  Generated weights: {class_weights}")


    from feature_extraction import LFCCExtractor
    lfcc_extractor = LFCCExtractor(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=60,
        n_filter=60
    )
    
    model = LCNN(n_lfcc=60, num_classes=2)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = LFCCLCNNTrainer(
        model=model,
        lfcc_extractor=lfcc_extractor,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=0.0001,
        weight_decay=0.0001,
        class_weights=class_weights
    )
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'weights', 'best_lfcc_lcnn_wavefake.pt')

    print("TRAINING CONFIGURATION")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Scheduler: Cosine Annealing")
    print(f"Gradient Clipping: 1.0")
    print(f"Save Path: {save_path}")

    best_eer = trainer.train(
        num_epochs=NUM_EPOCHS, 
        save_path=save_path,
        grad_clip=1.0,
        scheduler_type='cosine'
    )
    
    print("TRAINING SUMMARY")
    print(f"Best EER achieved: {best_eer:.4f}%")
    print(f"Model saved to: {save_path}")
    print(f"Metrics saved to: {os.path.join(os.path.dirname(save_path), 'training_metrics.json')}")



if __name__ == '__main__':
    main()
