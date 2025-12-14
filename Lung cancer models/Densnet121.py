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

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom Dataset Class with enhanced error handling
class DiseaseDataset(Dataset):
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
        
        # Enhanced augmentations for medical images
        if is_training:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.85, 1.15)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.is_training and hasattr(self, 'augment_transform'):
            if random.random() > 0.4:  # Apply augmentation 60% of the time
                image = self.augment_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_dataset_pytorch(dataset_path):
    # Optimized transforms with stronger augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.20))
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(dataset_path, 'train')
    test_dir = os.path.join(dataset_path, 'test')
    valid_dir = os.path.join(dataset_path, 'valid')
    
    for dir_path, dir_name in [(train_dir, 'train'), (test_dir, 'test'), (valid_dir, 'valid')]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    train_dataset = DiseaseDataset(train_dir, train_transform, is_training=True)
    test_dataset = DiseaseDataset(test_dir, val_test_transform, is_training=False)
    valid_dataset = DiseaseDataset(valid_dir, val_test_transform, is_training=False)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    # Optimized batch size and workers
    batch_size = 16  # Smaller batch for better generalization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, valid_loader, num_classes, class_names

# Enhanced DenseNet121 with improved architecture
class OptimizedDenseNet121(nn.Module):
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(OptimizedDenseNet121, self).__init__()
        
        # Load pretrained DenseNet121
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Fine-tune strategy: freeze early layers, unfreeze later dense blocks
        # DenseNet has: features.denseblock1-4 and features.transition1-3
        for name, param in self.densenet.features.named_parameters():
            # Freeze denseblock1 and transition1 (early layers)
            if 'denseblock1' in name or 'transition1' in name or 'conv0' in name or 'norm0' in name:
                param.requires_grad = False
            else:
                # Unfreeze denseblock2, 3, 4 and their transitions
                param.requires_grad = True
        
        # Get number of features from DenseNet121
        num_features = self.densenet.classifier.in_features  # 1024 for DenseNet121
        
        # Enhanced classifier with progressive feature reduction
        self.densenet.classifier = nn.Sequential(
            # First block: 1024 -> 512
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            
            # Second block: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.7),
            
            # Third block: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.5),
            
            # Output layer
            nn.Linear(128, num_classes)
        )
        
        # Xavier initialization for better convergence
        for m in self.densenet.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.densenet(x)

# Training with advanced techniques
def train_model(model, train_loader, valid_loader, num_epochs=100, learning_rate=0.0001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Calculate class weights
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    
    unique, counts = np.unique(all_labels, return_counts=True)
    class_weights = len(all_labels) / (len(unique) * counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0
    best_epoch = 0
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            try:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if len(train_loader) == 0:
            continue
            
        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train if total_train > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in valid_loader:
                try:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
                except Exception as e:
                    continue
        
        if len(valid_loader) > 0:
            val_loss /= len(valid_loader)
            val_acc = correct_val / total_val if total_val > 0 else 0
        else:
            val_loss = 0
            val_acc = 0
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_densenet121_model.pth')
            print(f"  >>> New best model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        print(f'Epoch [{epoch+1}/{num_epochs}], LR: {current_lr:.7f}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model was at epoch {best_epoch}")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}\n")
    
    if os.path.exists('best_densenet121_model.pth'):
        model.load_state_dict(torch.load('best_densenet121_model.pth'))
        print("Loaded best model weights for final evaluation")
    
    history = {
        'loss': train_losses,
        'val_loss': val_losses,
        'accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    }
    
    return model, history

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            try:
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
                
            except Exception as e:
                print(f"Error in evaluation: {e}")
                continue
    
    if len(all_predictions) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)

def plot_history(history):
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    loss_gap = [abs(t - v) for t, v in zip(history['loss'], history['val_loss'])]
    acc_gap = [abs(t - v) for t, v in zip(history['accuracy'], history['val_accuracy'])]
    plt.plot(loss_gap, label='Loss Gap', linewidth=2, color='red', alpha=0.7)
    plt.plot(acc_gap, label='Accuracy Gap', linewidth=2, color='orange', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training-Validation Gap', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_densenet121.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_heatmap(y_pred, y_true, class_names):
    if len(y_pred) == 0 or len(y_true) == 0:
        print("No predictions to plot confusion matrix")
        return
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names,
               cbar=True, square=True, annot_kws={'size': 12, 'weight': 'bold'}, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', 
               xticklabels=class_names, yticklabels=class_names,
               cbar=True, square=True, annot_kws={'size': 12, 'weight': 'bold'}, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix_heatmap_densenet121.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix heatmap saved as 'confusion_matrix_heatmap_densenet121.png'")
    plt.show()

def prediction_report(y_pred, y_true, class_names):
    if len(y_pred) == 0 or len(y_true) == 0:
        print("No predictions to evaluate")
        return
    
    print('\n' + '='*80)
    print('CLASSIFICATION REPORT')
    print('='*80)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4))
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print('='*80)
    print(f'OVERALL METRICS:')
    print(f'  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)')
    print(f'  Macro Precision: {precision:.4f} ({precision*100:.2f}%)')
    print(f'  Macro Recall:    {recall:.4f} ({recall*100:.2f}%)')
    print(f'  Macro F1 Score:  {f1:.4f} ({f1*100:.2f}%)')
    print('='*80 + '\n')
    
    plot_confusion_matrix_heatmap(y_pred, y_true, class_names)

def main():
    # UPDATE THIS PATH TO YOUR DATASET LOCATION
    DATASET_PATH = r"C:\Users\habib\Desktop\Thesis Paper\Data sets\Data set ct scan images\Data"
    
    try:
        print("="*80)
        print("OPTIMIZED DENSENET121 DISEASE CLASSIFICATION - 100 EPOCHS")
        print("="*80 + "\n")
        
        # Load dataset
        train_loader, test_loader, valid_loader, num_classes, class_names = load_dataset_pytorch(DATASET_PATH)
        
        print(f"\nDataset loaded successfully!")
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(valid_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")
        
        # Train optimized model
        print("\n" + "="*80)
        print("TRAINING OPTIMIZED DENSENET121 MODEL")
        print("="*80)
        model = OptimizedDenseNet121(num_classes=num_classes, pretrained=True, dropout_rate=0.5)
        
        trained_model, history = train_model(
            model, train_loader, valid_loader,
            num_epochs=100,
            learning_rate=0.0001
        )
        
        # Plot training history
        plot_history(history)
        
        # Evaluate on test set
        print("\n" + "="*80)
        print("FINAL EVALUATION ON TEST SET")
        print("="*80)
        y_pred, y_true, y_probs = evaluate_model(trained_model, test_loader)
        
        if len(y_pred) > 0:
            accuracy = accuracy_score(y_true, y_pred)
            print(f"\n{'='*80}")
            print(f"FINAL TEST ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"{'='*80}\n")
            
            prediction_report(y_pred, y_true, class_names)
            
            # Save final model
            torch.save(trained_model.state_dict(), 'optimized_densenet121_final.pth')
            print("\nFinal model saved as 'optimized_densenet121_final.pth'")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()