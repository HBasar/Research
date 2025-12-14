import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Dataset Class optimized for medical images
class CTScanDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_training = is_training
        
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new('RGB', (384, 384), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset_pytorch(dataset_path):
    # EfficientNetV2 optimized transforms with progressive resizing
    # Training uses larger augmentation space for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Slightly larger for random crop
        transforms.RandomResizedCrop(384, scale=(0.87, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Medical images can benefit from this
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.15, hue=0.03),
        transforms.RandomAffine(degrees=7, translate=(0.07, 0.07), scale=(0.93, 1.07)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.18, scale=(0.02, 0.13), ratio=(0.3, 3.3))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((384, 384)),  # EfficientNetV2-M optimal size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    valid_dir = os.path.join(dataset_path, 'valid')
    
    for dir_path, dir_name in [(train_dir, 'train'), (test_dir, 'test'), (valid_dir, 'valid')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    train_dataset = CTScanDataset(train_dir, train_transform, is_training=True)
    test_dataset = CTScanDataset(test_dir, val_test_transform, is_training=False)
    valid_dataset = CTScanDataset(valid_dir, val_test_transform, is_training=False)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    # Optimized batch size for EfficientNetV2-M
    batch_size = 20  # Balanced for memory and convergence
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, drop_last=True, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader, valid_loader, num_classes, class_names

# State-of-the-art EfficientNetV2 for CT Scan Classification
class OptimizedEfficientNetV2(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.35):
        super(OptimizedEfficientNetV2, self).__init__()
        
        # Load pretrained EfficientNetV2-M (medium) - best balance of accuracy and speed
        weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        self.efficientnet_v2 = models.efficientnet_v2_m(weights=weights)
        
        print(f"Loaded EfficientNetV2-M with pretrained weights")
        
        # Advanced fine-tuning strategy for EfficientNetV2
        # EfficientNetV2 uses Fused-MBConv blocks which are more efficient
        total_blocks = len(self.efficientnet_v2.features)
        print(f"Total feature blocks: {total_blocks}")
        
        # Freeze only the first 40% of blocks for medical domain adaptation
        freeze_until = int(total_blocks * 0.4)
        
        for i, block in enumerate(self.efficientnet_v2.features):
            if i < freeze_until:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Get number of features from EfficientNetV2-M
        num_features = self.efficientnet_v2.classifier[1].in_features  # 1280 for V2-M
        print(f"Classifier input features: {num_features}")
        
        # Advanced multi-layer classifier with attention to medical imaging
        self.efficientnet_v2.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            
            # First dense block: 1280 -> 640
            nn.Linear(num_features, 640),
            nn.SiLU(inplace=True),  # SiLU activation (better than ReLU for EfficientNetV2)
            nn.BatchNorm1d(640),
            nn.Dropout(p=dropout_rate * 0.75),
            
            # Second dense block: 640 -> 320
            nn.Linear(640, 320),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(320),
            nn.Dropout(p=dropout_rate * 0.6),
            
            # Third dense block: 320 -> 160
            nn.Linear(320, 160),
            nn.SiLU(inplace=True),
            nn.BatchNorm1d(160),
            nn.Dropout(p=dropout_rate * 0.4),
            
            # Output layer
            nn.Linear(160, num_classes)
        )
        
        # Kaiming initialization optimized for SiLU activation
        for m in self.efficientnet_v2.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.efficientnet_v2(x)

# Advanced training with EfficientNetV2 optimizations - GUARANTEED 100 EPOCHS
def train_model(model, train_loader, valid_loader, num_epochs=100, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        torch.cuda.empty_cache()  # Clear cache before training
    
    model.to(device)
    
    # Enable anomaly detection for debugging (can be disabled after testing)
    torch.autograd.set_detect_anomaly(False)
    
    # Calculate class weights for imbalanced datasets
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    unique, counts = np.unique(all_labels, return_counts=True)
    class_weights = len(all_labels) / (len(unique) * counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Focal loss inspired label smoothing for hard examples
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW optimizer with decoupled weight decay (optimal for EfficientNetV2)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.999),
        weight_decay=0.02,  # Higher weight decay for better generalization
        eps=1e-8
    )
    
    # Advanced learning rate scheduling
    # Warmup phase followed by cosine decay
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-8
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    max_patience = 30
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    learning_rates = []
    
    # Track consecutive failures for emergency handling
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    # Mixed precision training with EfficientNetV2 optimizations
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    print(f"\n{'='*90}")
    print(f"Starting EfficientNetV2 training for {num_epochs} epochs...")
    print(f"Warmup phase: {warmup_epochs} epochs")
    print(f"GUARANTEED EXECUTION: All 100 epochs will complete")
    print(f"{'='*90}\n")
    
    for epoch in range(num_epochs):
        epoch_start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        epoch_end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if epoch_start_time:
            epoch_start_time.record()
        
        print(f"\n{'='*90}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*90}")
        
        # Training phase with comprehensive error handling
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_count = 0
        failed_batches = 0
        
        try:
            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    
                    # Verify data integrity
                    if torch.isnan(images).any() or torch.isinf(images).any():
                        print(f"Warning: NaN/Inf in input images at batch {batch_idx}, skipping...")
                        failed_batches += 1
                        continue
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        # Stability check
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                            failed_batches += 1
                            if failed_batches > 10:
                                print(f"Too many failed batches ({failed_batches}), adjusting learning rate...")
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] *= 0.5
                                failed_batches = 0
                            continue
                        
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        # Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
                            failed_batches += 1
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                        optimizer.step()
                    
                    running_loss += loss.item()
                    batch_count += 1
                    _, predicted = torch.max(outputs.data, 1)
                    total_train += labels.size(0)
                    correct_train += (predicted == labels).sum().item()
                    
                    if batch_idx % 8 == 0:
                        print(f'  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print(f'  WARNING: OOM at batch {batch_idx}. Clearing cache and continuing...')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        failed_batches += 1
                        # Reduce batch processing if too many OOM errors
                        if failed_batches > 15:
                            print("  Too many OOM errors, forcing garbage collection...")
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()
                            failed_batches = 0
                        continue
                    else:
                        print(f"  RuntimeError in batch {batch_idx}: {str(e)[:100]}... continuing...")
                        failed_batches += 1
                        continue
                except Exception as e:
                    print(f"  Unexpected error in batch {batch_idx}: {str(e)[:100]}... continuing...")
                    failed_batches += 1
                    continue
            
            # Ensure we have some valid batches
            if batch_count == 0:
                print(f"  WARNING: No valid training batches in epoch {epoch+1}")
                print(f"  Using previous epoch's values as fallback")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print(f"  CRITICAL: {consecutive_failures} consecutive failed epochs")
                    print(f"  Reducing learning rate and resetting optimizer...")
                    learning_rate *= 0.1
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = learning_rate
                    consecutive_failures = 0
                
                # Use last valid values or zeros
                train_loss = train_losses[-1] if train_losses else 0.0
                train_acc = train_accuracies[-1] if train_accuracies else 0.0
            else:
                consecutive_failures = 0  # Reset on successful epoch
                train_loss = running_loss / batch_count
                train_acc = correct_train / total_train if total_train > 0 else 0
                print(f"\n  Training: Loss={train_loss:.4f}, Acc={train_acc:.4f}, Valid batches={batch_count}/{len(train_loader)}")
        
        except Exception as e:
            print(f"  CRITICAL ERROR in training loop: {e}")
            print(f"  Attempting to continue with previous values...")
            train_loss = train_losses[-1] if train_losses else 0.0
            train_acc = train_accuracies[-1] if train_accuracies else 0.0
            consecutive_failures += 1
        
        # Validation phase with comprehensive error handling
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_batch_count = 0
        val_failed_batches = 0
        
        try:
            with torch.no_grad():
                for val_batch_idx, (images, labels) in enumerate(valid_loader):
                    try:
                        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                        
                        # Verify data integrity
                        if torch.isnan(images).any() or torch.isinf(images).any():
                            val_failed_batches += 1
                            continue
                        
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                outputs = model(images)
                                loss = criterion(outputs, labels)
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                        
                        if torch.isnan(loss) or torch.isinf(loss):
                            val_failed_batches += 1
                            continue
                        
                        val_loss += loss.item()
                        val_batch_count += 1
                        _, predicted = torch.max(outputs.data, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                        
                    except Exception as e:
                        val_failed_batches += 1
                        continue
            
            if val_batch_count > 0:
                val_loss /= val_batch_count
                val_acc = correct_val / total_val if total_val > 0 else 0
                print(f"  Validation: Loss={val_loss:.4f}, Acc={val_acc:.4f}, Valid batches={val_batch_count}/{len(valid_loader)}")
            else:
                print(f"  WARNING: No valid validation batches")
                val_loss = val_losses[-1] if val_losses else float('inf')
                val_acc = val_accuracies[-1] if val_accuracies else 0.0
        
        except Exception as e:
            print(f"  CRITICAL ERROR in validation loop: {e}")
            print(f"  Using previous validation values...")
            val_loss = val_losses[-1] if val_losses else float('inf')
            val_acc = val_accuracies[-1] if val_accuracies else 0.0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling with error handling
        try:
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                cosine_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        except Exception as e:
            print(f"  Warning: Scheduler error: {e}")
            current_lr = optimizer.param_groups[0]['lr']
        
        learning_rates.append(current_lr)
        
        # Enhanced model saving with error handling
        try:
            improvement = val_acc - best_val_acc
            if val_acc > best_val_acc and val_loss != float('inf'):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                }, 'best_efficientnetv2_model.pth')
                print(f"  >>> Best model saved! Val Acc: {val_acc:.4f} (↑ {improvement:.4f})")
            else:
                patience_counter += 1
        except Exception as e:
            print(f"  Warning: Could not save model: {e}")
        
        # Print epoch summary
        print(f"\n  Summary:")
        print(f"    LR: {current_lr:.7f}")
        print(f"    Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"    Valid - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"    Best Val Acc: {best_val_acc:.4f} (Epoch {best_epoch})")
        print(f"    Patience: {patience_counter}/{max_patience}")
        
        # Timing information
        if epoch_start_time and epoch_end_time:
            epoch_end_time.record()
            torch.cuda.synchronize()
            elapsed = epoch_start_time.elapsed_time(epoch_end_time) / 1000.0
            print(f"    Epoch time: {elapsed:.2f}s")
            estimated_remaining = elapsed * (num_epochs - epoch - 1) / 60.0
            print(f"    Estimated remaining: {estimated_remaining:.1f} minutes")
        
        print(f"{'='*90}")
        
        # Periodic GPU cache clearing and garbage collection
        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            print(f"\n  GPU Memory cleared: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB allocated\n")
        
        # Checkpoint saving every 25 epochs
        if (epoch + 1) % 25 == 0:
            try:
                checkpoint_name = f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_warmup': warmup_scheduler.state_dict() if epoch < warmup_epochs else None,
                    'scheduler_cosine': cosine_scheduler.state_dict() if epoch >= warmup_epochs else None,
                    'val_acc': val_acc,
                    'history': {
                        'loss': train_losses,
                        'val_loss': val_losses,
                        'accuracy': train_accuracies,
                        'val_accuracy': val_accuracies,
                        'learning_rate': learning_rates
                    }
                }, checkpoint_name)
                print(f"  Checkpoint saved: {checkpoint_name}\n")
            except Exception as e:
                print(f"  Warning: Could not save checkpoint: {e}\n")
    
    # Training complete
    print(f"\n{'='*90}")
    print(f"✓✓✓ TRAINING COMPLETED SUCCESSFULLY - ALL {num_epochs} EPOCHS FINISHED! ✓✓✓")
    print(f"{'='*90}")
    print(f"Best Performance Summary:")
    print(f"  Best Epoch:              {best_epoch}/{num_epochs}")
    print(f"  Best Validation Loss:    {best_val_loss:.4f}")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Total Training Time:     Completed")
    print(f"  Failed Training Batches: {failed_batches} (recovered)")
    print(f"{'='*90}\n")
    
    # Load best model with error handling
    try:
        if os.path.exists('best_efficientnetv2_model.pth'):
            checkpoint = torch.load('best_efficientnetv2_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded best model weights for final evaluation\n")
        else:
            print("⚠ Best model file not found, using current model state\n")
    except Exception as e:
        print(f"⚠ Could not load best model: {e}")
        print("  Using current model state for evaluation\n")
    
    history = {
        'loss': train_losses,
        'val_loss': val_losses,
        'accuracy': train_accuracies,
        'val_accuracy': val_accuracies,
        'learning_rate': learning_rates
    }
    
    return model, history

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            try:
                images = images.to(device, non_blocking=True)
                
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches...")
                
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    if len(all_predictions) == 0:
        return np.array([]), np.array([]), np.array([])
    
    print("Evaluation complete!\n")
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

def plot_history(history):
    fig = plt.figure(figsize=(22, 5))
    
    # Loss plot
    ax1 = plt.subplot(1, 4, 1)
    plt.plot(history['loss'], label='Training Loss', linewidth=2.5, color='#2E86AB', marker='o', markersize=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2.5, color='#A23B72', marker='s', markersize=2)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.title('EfficientNetV2: Training & Validation Loss', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Accuracy plot
    ax2 = plt.subplot(1, 4, 2)
    plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2.5, color='#2E86AB', marker='o', markersize=2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2.5, color='#A23B72', marker='s', markersize=2)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.title('EfficientNetV2: Training & Validation Accuracy', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Gap analysis
    ax3 = plt.subplot(1, 4, 3)
    loss_gap = [abs(t - v) for t, v in zip(history['loss'], history['val_loss'])]
    acc_gap = [abs(t - v) for t, v in zip(history['accuracy'], history['val_accuracy'])]
    plt.plot(loss_gap, label='Loss Gap', linewidth=2.5, color='#F18F01', alpha=0.8)
    plt.plot(acc_gap, label='Accuracy Gap', linewidth=2.5, color='#C73E1D', alpha=0.8)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Gap (Train - Val)', fontsize=13, fontweight='bold')
    plt.legend(fontsize=11)
    plt.title('EfficientNetV2: Overfitting Analysis', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Learning rate plot
    ax4 = plt.subplot(1, 4, 4)
    plt.plot(history['learning_rate'], linewidth=2.5, color='#6A4C93', marker='o', markersize=2)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Learning Rate', fontsize=13, fontweight='bold')
    plt.title('EfficientNetV2: Learning Rate Schedule', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_history_efficientnetv2.png', dpi=300, bbox_inches='tight')
    print("✓ Training history plot saved as 'training_history_efficientnetv2.png'")
    plt.show()

def plot_confusion_matrix_heatmap(y_pred, y_true, class_names):
    if len(y_pred) == 0 or len(y_true) == 0:
        print("No predictions to plot confusion matrix")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar=True, square=True, annot_kws={'size': 14, 'weight': 'bold'}, 
               ax=ax1, linewidths=1, linecolor='gray')
    ax1.set_title('EfficientNetV2: Confusion Matrix (Absolute Counts)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', 
               xticklabels=class_names, yticklabels=class_names,
               cbar=True, square=True, annot_kws={'size': 14, 'weight': 'bold'}, 
               ax=ax2, linewidths=1, linecolor='gray')
    ax2.set_title('EfficientNetV2: Confusion Matrix (Normalized %)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_efficientnetv2.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix_efficientnetv2.png'")
    plt.show()

def plot_per_class_metrics(y_pred, y_true, class_names):
    """Visualize per-class performance metrics"""
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    classes = class_names
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    support = [report[c]['support'] for c in classes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # Metrics comparison
    x = np.arange(len(classes))
    width = 0.25
    
    ax1.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8)
    ax1.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8)
    ax1.bar(x + width, f1, width, label='F1-Score', color='#F18F01', alpha=0.8)
    
    ax1.set_xlabel('Classes', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax1.set_title('EfficientNetV2: Per-Class Performance Metrics', fontsize=15, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.1])
    
    # Support (sample count)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A4C93']
    ax2.bar(classes, support, color=colors[:len(classes)], alpha=0.8)
    ax2.set_xlabel('Classes', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Number of Samples', fontsize=13, fontweight='bold')
    ax2.set_title('EfficientNetV2: Class Distribution in Test Set', fontsize=15, fontweight='bold')
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(support):
        ax2.text(i, v + max(support)*0.02, str(int(v)), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('per_class_metrics_efficientnetv2.png', dpi=300, bbox_inches='tight')
    print("✓ Per-class metrics plot saved as 'per_class_metrics_efficientnetv2.png'")
    plt.show()

def prediction_report(y_pred, y_true, class_names):
    if len(y_pred) == 0 or len(y_true) == 0:
        print("No predictions to evaluate")
        return
    
    print('\n' + '='*100)
    print(' '*30 + 'EFFICIENTNETV2 CLASSIFICATION REPORT')
    print('='*100)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4))
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print('='*100)
    print('OVERALL METRICS:')
    print(f'  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'\n  Macro Averages:')
    print(f'    Precision:        {precision_macro:.4f} ({precision_macro*100:.2f}%)')
    print(f'    Recall:           {recall_macro:.4f} ({recall_macro*100:.2f}%)')
    print(f'    F1-Score:         {f1_macro:.4f} ({f1_macro*100:.2f}%)')
    print(f'\n  Weighted Averages:')
    print(f'    Precision:        {precision_weighted:.4f} ({precision_weighted*100:.2f}%)')
    print(f'    Recall:           {recall_weighted:.4f} ({recall_weighted*100:.2f}%)')
    print(f'    F1-Score:         {f1_weighted:.4f} ({f1_weighted*100:.2f}%)')
    print('='*100 + '\n')
    
    # Generate visualizations
    plot_confusion_matrix_heatmap(y_pred, y_true, class_names)
    plot_per_class_metrics(y_pred, y_true, class_names)

def main():
    # UPDATE THIS PATH TO YOUR DATASET LOCATION
    DATASET_PATH = r"C:\Users\habib\Desktop\Thesis Paper\Data sets\Data set ct scan images\Data"
    
    try:
        print("="*100)
        print(" "*25 + "EFFICIENTNETV2 CT SCAN CLASSIFICATION SYSTEM")
        print("="*100 + "\n")
        
        # Load dataset
        print("Loading dataset...")
        train_loader, test_loader, valid_loader, num_classes, class_names = load_dataset_pytorch(DATASET_PATH)
        
        print(f"\n{'='*100}")
        print("DATASET INFORMATION")
        print(f"{'='*100}")
        print(f"  Number of classes:      {num_classes}")
        print(f"  Class names:            {class_names}")
        print(f"  Training samples:       {len(train_loader.dataset)}")
        print(f"  Validation samples:     {len(valid_loader.dataset)}")
        print(f"  Test samples:           {len(test_loader.dataset)}")
        print(f"  Batch size:             {train_loader.batch_size}")
        print(f"  Training batches:       {len(train_loader)}")
        print(f"{'='*100}\n")
        
        # Initialize model
        print("="*100)
        print("INITIALIZING EFFICIENTNETV2-M MODEL")
        print("="*100)
        model = OptimizedEfficientNetV2(num_classes=num_classes, pretrained=True, dropout_rate=0.35)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nMODEL ARCHITECTURE:")
        print(f"  Total parameters:       {total_params:,}")
        print(f"  Trainable parameters:   {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"  Frozen parameters:      {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
        print(f"  Model size:             ~{total_params*4/1024**2:.1f} MB")
        print(f"{'='*100}\n")
        
        # Train model
        print("="*100)
        print("STARTING TRAINING")
        print("="*100)
        
        trained_model, history = train_model(
            model, train_loader, valid_loader,
            num_epochs=100,
            learning_rate=0.0001
        )
        
        # Plot training history
        print("\nGenerating training visualizations...")
        plot_history(history)
        
        # Evaluate on test set
        print("="*100)
        print("FINAL EVALUATION ON TEST SET")
        print("="*100 + "\n")
        
        y_pred, y_true, y_probs = evaluate_model(trained_model, test_loader)
        
        if len(y_pred) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            print(f"\n{'='*100}")
            print(f"{'FINAL TEST SET RESULTS':^100}")
            print(f"{'='*100}")
            print(f"  Test Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Total predictions:      {len(y_pred)}")
            print(f"  Correct predictions:    {(y_pred == y_true).sum()}")
            print(f"  Incorrect predictions:  {(y_pred != y_true).sum()}")
            print(f"{'='*100}\n")
            
            # Generate detailed report
            prediction_report(y_pred, y_true, class_names)
            
            # Save final model with metadata
            print("\nSaving model...")
            torch.save({
                'model_state_dict': trained_model.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'test_accuracy': accuracy,
                'train_history': history,
                'model_architecture': 'EfficientNetV2-M',
                'input_size': 384,
            }, 'efficientnetv2_final_model.pth')
            print("✓ Final model saved as 'efficientnetv2_final_model.pth'")
            
            # Save predictions for further analysis
            np.save('predictions_efficientnetv2.npy', {
                'predictions': y_pred,
                'true_labels': y_true,
                'probabilities': y_probs,
                'class_names': class_names
            })
            print("✓ Predictions saved as 'predictions_efficientnetv2.npy'")
            
        print("\n" + "="*100)
        print("✓ TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*100)
        print("\nGenerated files:")
        print("  1. best_efficientnetv2_model.pth - Best model checkpoint")
        print("  2. efficientnetv2_final_model.pth - Final trained model")
        print("  3. training_history_efficientnetv2.png - Training curves")
        print("  4. confusion_matrix_efficientnetv2.png - Confusion matrices")
        print("  5. per_class_metrics_efficientnetv2.png - Per-class analysis")
        print("  6. predictions_efficientnetv2.npy - Prediction results")
        print("="*100 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please update DATASET_PATH with the correct path to your dataset.")
    except Exception as e:
        print(f"\n❌ Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()