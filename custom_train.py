import os
import sys
import time
import argparse
from datetime import datetime, timedelta
from huggingface_hub import HfApi, create_repo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from pathlib import Path
import torch.nn.functional as F

# Add the project path to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class EnhancedCNN(nn.Module):
    """A real CNN model for image classification with improved architecture"""

    def __init__(self, num_classes=3):
        super(EnhancedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),  # Adjust based on your image size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class CustomImageDataset(Dataset):
    """Dataset class to handle your nested folder structure"""

    def __init__(self, data_dir, classes, transform=None):
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.transform = transform
        self.samples = []

        # Build dataset from your nested structure: class/subclass/img_folder/*
        for class_idx, class_name in enumerate(classes):
            class_path = self.data_dir / class_name
            if class_path.exists():
                # Iterate through all subclasses
                for subclass in class_path.iterdir():
                    if subclass.is_dir():
                        img_folder = subclass / 'img'
                        if img_folder.exists():
                            for img_file in img_folder.glob('*.*'):
                                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                    self.samples.append((img_file, class_idx))

        print(f"Found {len(self.samples)} images across {len(classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = datasets.folder.default_loader(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label


class AdvancedTrainingLogger:
    def __init__(self, log_file_path, team_number=9):
        self.log_file_path = log_file_path
        self.team_number = team_number
        self.start_time = time.time()
        self.epoch_start_time = None
        self.batch_start_time = None
        self.last_50_start_time = None
        self.epoch_times = []

        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def log(self, message, print_to_console=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"

        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

        if print_to_console:
            print(log_message)

    def format_time_detailed(self, seconds):
        """Format time in HH:MM:SS format for detailed reporting"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def start_epoch(self):
        self.epoch_start_time = time.time()
        self.batch_start_time = self.epoch_start_time
        self.last_50_start_time = self.epoch_start_time

    def log_sample_progress(self, epoch, current_sample, total_samples, loss=None, iou=None, accuracy=None):
        """Log progress every 50 samples with detailed timing information"""
        if current_sample % 50 == 0 and current_sample > 0:
            current_time = time.time()

            # Calculate timing information
            time_last_50 = current_time - self.last_50_start_time
            time_since_beginning = current_time - self.start_time
            time_since_epoch_start = current_time - self.epoch_start_time

            # Calculate remaining time for epoch
            if current_sample > 0:
                avg_time_per_sample = time_since_epoch_start / current_sample
                samples_remaining = total_samples - current_sample
                time_remaining_epoch = avg_time_per_sample * samples_remaining
            else:
                time_remaining_epoch = 0

            # Build the main progress message
            message = (f"Epoch {epoch} : {current_sample} / {total_samples} samples ,\n"
                       f"time for last 50 samples : {self.format_time_detailed(time_last_50)} hours ,\n"
                       f"time since beginning : {self.format_time_detailed(time_since_beginning)} hours ,\n"
                       f"time left to finish the epoch : {self.format_time_detailed(time_remaining_epoch)} hours")

            # Add metrics if provided
            if loss is not None:
                message += f"\nLoss: {loss:.4f}"
            if iou is not None:
                message += f" | IoU: {iou:.4f}"
            if accuracy is not None:
                message += f" | Accuracy: {accuracy:.4f}"

            self.log(message)
            self.last_50_start_time = current_time

    def log_batch_progress(self, epoch, current_batch, total_batches, batch_size, loss=None, iou=None, accuracy=None):
        """Log batch progress with timing and metrics"""
        current_time = time.time()
        batch_duration = current_time - self.batch_start_time
        epoch_duration = current_time - self.epoch_start_time

        if current_batch > 0:
            avg_time_per_batch = epoch_duration / current_batch
            batches_remaining = total_batches - current_batch
            time_remaining = avg_time_per_batch * batches_remaining

            samples_processed = current_batch * batch_size

            message = (f"Epoch {epoch}: Batch {current_batch}/{total_batches} "
                       f"({samples_processed} samples) | "
                       f"Batch Time: {batch_duration:.2f}s | "
                       f"Epoch Time: {self.format_time_detailed(epoch_duration)} | "
                       f"ETA: {self.format_time_detailed(time_remaining)}")

            # Add metrics if provided
            if loss is not None:
                message += f" | Loss: {loss:.4f}"
            if iou is not None:
                message += f" | IoU: {iou:.4f}"
            if accuracy is not None:
                message += f" | Accuracy: {accuracy:.4f}"

            self.log(message)

        self.batch_start_time = current_time

    def end_epoch(self, epoch, metrics, train_loss=None, val_loss=None, iou=None):
        """Log epoch completion with comprehensive metrics"""
        epoch_duration = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_duration)

        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        epochs_remaining = (getattr(self, 'total_epochs', 5) - epoch) if hasattr(self, 'total_epochs') else 0
        total_estimated = avg_epoch_time * epochs_remaining

        # Log epoch completion with timing
        self.log(f"=== Epoch {epoch} Completion ===")
        self.log(f"Epoch completed in {self.format_time_detailed(epoch_duration)}")
        self.log(f"Average epoch time: {self.format_time_detailed(avg_epoch_time)}")
        self.log(f"Estimated training completion: {self.format_time_detailed(total_estimated)}")

        # Log comprehensive metrics
        self.log(f"Epoch {epoch} Detailed Metrics:")
        for metric_name, metric_value in metrics.items():
            self.log(f"  {metric_name}: {metric_value:.4f}")

        # Log specific metrics if provided
        if train_loss is not None:
            self.log(f"  Training Loss: {train_loss:.4f}")
        if val_loss is not None:
            self.log(f"  Validation Loss: {val_loss:.4f}")
        if iou is not None:
            self.log(f"  IoU: {iou:.4f}")

    def set_total_epochs(self, total_epochs):
        """Set total epochs for better time estimation"""
        self.total_epochs = total_epochs


class CheckpointManager:
    def __init__(self, local_dir, hf_repo_id, team_number, enable_hf_upload=True):
        self.local_dir = local_dir
        self.hf_repo_id = hf_repo_id
        self.team_number = team_number
        self.enable_hf_upload = enable_hf_upload
        self.api = HfApi()
        self.repo_created = False
        os.makedirs(local_dir, exist_ok=True)

    def ensure_repo_exists(self):
        """Create the repository on Hugging Face Hub if it doesn't exist"""
        if not self.enable_hf_upload or self.repo_created:
            return True
        try:
            # Ensure we're logged in first
            try:
                login(token="your_hugging_face_token_here")  # Replace with your token
            except:
                self.log("Warning: Could not authenticate with Hugging Face Hub")
                self.enable_hf_upload = False
                return False

            create_repo(
                repo_id=self.hf_repo_id,
                repo_type="model",
                private=True,
                exist_ok=True
            )
            self.repo_created = True
            self.log(f"Successfully created/verified repository: {self.hf_repo_id}")
            return True
        except Exception as e:
            self.log(f"Warning: Could not create repository {self.hf_repo_id}: {str(e)}")
            self.log("Checkpoints will be saved locally only")
            self.enable_hf_upload = False
            return False

    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save checkpoint locally and optionally upload to Hugging Face Hub"""
        if model is None:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': {},
                'optimizer_state_dict': {},
                'team_number': self.team_number,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'simulation': True
            }
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else {},
                'team_number': self.team_number,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'simulation': False
            }

        # Save locally
        local_path = os.path.join(self.local_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, local_path)

        # Upload to Hugging Face with retry logic
        hf_success = False
        if self.enable_hf_upload:
            if self.ensure_repo_exists():
                try:
                    self.api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=f'checkpoint_epoch_{epoch}.pth',
                        repo_id=self.hf_repo_id,
                        commit_message=f"Team {self.team_number} - Epoch {epoch} checkpoint",
                        repo_type="model"
                    )
                    hf_success = True
                    self.log(f"Successfully uploaded checkpoint to Hugging Face Hub: {self.hf_repo_id}")
                except Exception as e:
                    self.log(f"Failed to upload to Hugging Face: {str(e)}")
                    # Try one more time
                    try:
                        time.sleep(2)
                        self.api.upload_file(
                            path_or_fileobj=local_path,
                            path_in_repo=f'checkpoint_epoch_{epoch}.pth',
                            repo_id=self.hf_repo_id,
                            commit_message=f"Team {self.team_number} - Epoch {epoch} checkpoint (retry)",
                            repo_type="model"
                        )
                        hf_success = True
                        self.log(f"Successfully uploaded checkpoint on retry")
                    except Exception as e2:
                        self.log(f"Failed again to upload to Hugging Face: {str(e2)}")
                        hf_success = False
        return hf_success, local_path

    def log(self, message):
        print(f"[CheckpointManager] {message}")


def set_seed(seed):
    """Set fixed seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_real_model_and_optimizer(classes, device):
    """Setup real model and optimizer for actual training"""
    model = EnhancedCNN(num_classes=len(classes))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    model.to(device)
    print(f"Using real EnhancedCNN model with {sum(p.numel() for p in model.parameters())} parameters")
    return model, optimizer, scheduler


def setup_data_loaders(classes, batch_size=32):
    """Setup data loaders with proper transformations"""

    # Enhanced transformations for better accuracy
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    data_dir = "./data/lasot"  # Adjust path as needed

    # For now, using a simple dataset structure - you'll need to adapt to your nested structure
    try:
        full_dataset = CustomImageDataset(data_dir, classes, transform=train_transform)

        # Split dataset (80% train, 20% validation)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Apply validation transform to val dataset
        val_dataset.dataset = CustomImageDataset(data_dir, classes, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, len(full_dataset)

    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        print("Falling back to dummy data for demonstration")
        # Fallback to dummy data - REMOVE THIS IN PRODUCTION
        return None, None, 1000


def calculate_iou(predictions, targets):
    """Calculate Intersection over Union (IoU) for classification tasks"""
    # For classification, we can use a simplified IoU calculation
    # This is a placeholder - you may need to adapt this for your specific task
    pred_classes = predictions.argmax(dim=1)
    target_classes = targets

    iou_per_class = []
    for class_idx in range(predictions.size(1)):
        pred_mask = (pred_classes == class_idx)
        target_mask = (target_classes == class_idx)

        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()

        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou.item())
        else:
            iou_per_class.append(0.0)

    return sum(iou_per_class) / len(iou_per_class) if iou_per_class else 0.0


def calculate_real_metrics(model, data_loader, device, criterion):
    """Calculate real validation metrics including loss, accuracy, and IoU"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_iou = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Calculate IoU for this batch
            batch_iou = calculate_iou(output, target)
            total_iou += batch_iou
            num_batches += 1

    accuracy = correct / total if total > 0 else 0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'iou': avg_iou,
        'samples_processed': total
    }


def main():
    parser = argparse.ArgumentParser(description='Enhanced SeqTrack Training Script')
    parser.add_argument('--team_number', type=int, default=9, help='Team number for seed')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--hf_repo', type=str, default="NancyAbdullah11/assignment_3",
                        help='Hugging Face repository ID')
    parser.add_argument('--classes', nargs='+', default=['airplane', 'deer', 'electricfan'], help='Classes to use')
    parser.add_argument('--simulation', action='store_true', default=False, help='Run in simulation mode')
    parser.add_argument('--real_training', action='store_true', default=True, help='Run with real SeqTrack training')
    parser.add_argument('--local_only', action='store_true', default=False,
                        help='Run training locally without Hugging Face upload')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--log_interval', type=int, default=50, help='Log every N samples')

    args = parser.parse_args()

    # Set fixed seed
    set_seed(args.team_number)

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize logger
    logger = AdvancedTrainingLogger('./training_log.txt', args.team_number)
    logger.set_total_epochs(args.epochs)

    # Initialize checkpoint manager
    enable_hf_upload = not args.local_only and args.hf_repo
    checkpoint_manager = CheckpointManager(
        local_dir='./checkpoints',
        hf_repo_id=args.hf_repo if enable_hf_upload else "",
        team_number=args.team_number,
        enable_hf_upload=enable_hf_upload
    )

    logger.log(f"Starting training with seed: {args.team_number}")
    logger.log(f"Selected classes: {args.classes}")
    logger.log(f"Training for {args.epochs} epochs")
    logger.log(f"Using device: {device}")
    logger.log(f"Real training mode: {args.real_training}")
    logger.log(f"Log interval: every {args.log_interval} samples")

    # Setup model and data
    if args.real_training and not args.simulation:
        model, optimizer, scheduler = setup_real_model_and_optimizer(args.classes, device)
        train_loader, val_loader, dataset_size = setup_data_loaders(args.classes, args.batch_size)
        criterion = nn.CrossEntropyLoss()

        # Calculate total samples per epoch
        total_samples_per_epoch = len(train_loader.dataset)
        logger.log(f"Training samples per epoch: {total_samples_per_epoch}")
        logger.log(f"Total batches per epoch: {len(train_loader)}")
    else:
        logger.log("Running in simulation mode with dummy model")
        # Your existing simulation code here
        return

    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.log(f"=== Starting Epoch {epoch}/{args.epochs} ===")
        logger.start_epoch()

        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_accuracy = 0.0
        total_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate batch metrics
            pred = output.argmax(dim=1, keepdim=True)
            batch_accuracy = pred.eq(target.view_as(pred)).float().mean().item()
            batch_iou = calculate_iou(output, target)

            running_accuracy += batch_accuracy
            running_iou += batch_iou

            # Calculate current sample position
            current_sample = (batch_idx + 1) * args.batch_size
            if current_sample > total_samples_per_epoch:
                current_sample = total_samples_per_epoch

            # Log progress every 50 samples
            if current_sample % args.log_interval == 0 or batch_idx == total_batches - 1:
                avg_loss = running_loss / (batch_idx + 1)
                avg_iou = running_iou / (batch_idx + 1)
                avg_accuracy = running_accuracy / (batch_idx + 1)

                logger.log_sample_progress(
                    epoch=epoch,
                    current_sample=current_sample,
                    total_samples=total_samples_per_epoch,
                    loss=avg_loss,
                    iou=avg_iou,
                    accuracy=avg_accuracy
                )

        # Step the scheduler
        scheduler.step()

        # Calculate validation metrics at end of epoch
        val_metrics = calculate_real_metrics(model, val_loader, device, criterion)

        # Calculate final training metrics for the epoch
        final_train_loss = running_loss / total_batches
        final_train_iou = running_iou / total_batches
        final_train_accuracy = running_accuracy / total_batches

        # Log epoch completion with comprehensive metrics
        logger.end_epoch(
            epoch=epoch,
            metrics=val_metrics,
            train_loss=final_train_loss,
            val_loss=val_metrics['loss'],
            iou=val_metrics['iou']
        )

        # Save checkpoint
        success, checkpoint_path = checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, {**val_metrics, 'train_loss': final_train_loss}
        )

        if success:
            logger.log(f"Checkpoint saved locally and uploaded to HF: {checkpoint_path}")
        else:
            logger.log(f"Checkpoint saved locally: {checkpoint_path}")

    # Final training summary
    total_duration = time.time() - logger.start_time
    logger.log("=== TRAINING COMPLETED ===")
    logger.log(f"Total training time: {logger.format_time_detailed(total_duration)}")
    logger.log(f"Average epoch time: {logger.format_time_detailed(sum(logger.epoch_times) / len(logger.epoch_times))}")
    logger.log(f"Total epochs completed: {args.epochs}")


if __name__ == "__main__":
    main()