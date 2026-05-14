import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.data_loader import KFoldDataLoader
from src.models.csl500_model import IntermediateCSL500Model
from config.config import Config
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.data_loader import KFoldDataLoader
from src.models.csl500_model import IntermediateCSL500Model
from config.config import Config
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import os
import argparse
import glob
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, patience=10,
          start_epoch=0, fold=0, save_dir='model_checkpoints'):
    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss, correct_train, total_train = 0.0, 0, 0

            train_pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', ncols=100)
            for batch_idx, batch in enumerate(train_loader):
                if len(batch) == 3:
                    poses, labels, lengths = batch
                elif len(batch) == 2:
                    poses, labels = batch
                    lengths = torch.full((poses.size(0),), poses.size(1), dtype=torch.long)
                else:
                    raise ValueError(f"Unexpected batch size: {len(batch)}")

                if batch_idx == 0 and epoch == start_epoch:
                    logger.info(f"Batch shape: {poses.shape}")
                    logger.info(f"Labels shape: {labels.shape}")
                    logger.info(f"Lengths shape: {lengths.shape}")
                    logger.info(f"Pose data type: {poses.dtype}")
                    logger.info(f"Labels data type: {labels.dtype}")

                poses, labels, lengths = poses.to(device), labels.to(device), lengths.to(device)

                optimizer.zero_grad()
                outputs = model(poses, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

                train_pbar.update(1)
                train_pbar.set_postfix(
                    {'loss': f'{train_loss / (batch_idx + 1):.4f}',
                     'acc': f'{100. * correct_train / total_train:.2f}%'})

            train_pbar.close()

            model.eval()
            val_loss, correct_val, total_val = 0.0, 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        poses, labels, lengths = batch
                    elif len(batch) == 2:
                        poses, labels = batch
                        lengths = torch.full((poses.size(0),), poses.size(1), dtype=torch.long)
                    else:
                        raise ValueError(f"Unexpected batch size: {len(batch)}")

                    poses, labels, lengths = poses.to(device), labels.to(device), lengths.to(device)
                    outputs = model(poses, lengths)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = 100. * correct_train / total_train
            val_loss /= len(val_loader)
            val_accuracy = 100. * correct_val / total_val

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], '
                        f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
                        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, '
                        f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
            }, os.path.join(save_dir, f'checkpoint_fold{fold}_epoch{epoch + 1}.pth'))

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f'best_model_fold{fold}.pth'))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
        }, os.path.join(save_dir, f'interrupted_checkpoint_fold{fold}_epoch{epoch + 1}.pth'))
        print(f"Interrupted state saved. You can resume from epoch {epoch + 1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, f'final_model_fold{fold}.pth'))

    return best_val_accuracy, train_losses, train_accuracies, val_losses, val_accuracies

def plot_results(fold_results, save_dir):
    num_folds = len(fold_results)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Results')

    for i, metric in enumerate(['losses', 'accuracies']):
        for j, phase in enumerate(['train', 'val']):
            ax = axs[i, j]
            for fold in fold_results:
                ax.plot(fold[f'{phase}_{metric}'], label=f"Fold {fold['fold']}")
            ax.set_title(f'{phase.capitalize()} {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_results.png'))
    logger.info(f"Results plot saved as {os.path.join(save_dir, 'training_results.png')}")

def main(args):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    save_dir = 'model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    kfold_loader = KFoldDataLoader(config.data_path, config.labels_path, config.batch_size, n_splits=config.n_splits)

    fold_results = []

    try:
        for fold, (train_loader, val_loader) in enumerate(kfold_loader.get_loaders()):
            logger.info(f"Starting fold {fold + 1}/{config.n_splits}")

            model = IntermediateCSL500Model(config).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

            if config.lr_scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.cosine_lr_min)
            elif config.lr_scheduler == 'reduce_on_plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.reduce_factor,
                                              patience=config.reduce_patience)
            else:
                scheduler = None

            start_epoch = 0
            if args.resume:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_fold{fold}_epoch*.pth')
                checkpoints = sorted(glob.glob(checkpoint_path), key=lambda x: int(x.split('epoch')[1].split('.')[0]))
                if checkpoints:
                    latest_checkpoint = checkpoints[-1]
                    checkpoint = torch.load(latest_checkpoint)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")

            start_time = time.time()
            best_val_accuracy, train_losses, train_accuracies, val_losses, val_accuracies = train(
                model, train_loader, val_loader, criterion, optimizer, scheduler, device,
                config.num_epochs, patience=config.early_stopping_patience, start_epoch=start_epoch, fold=fold,
                save_dir=save_dir
            )
            end_time = time.time()

            fold_results.append({
                'fold': fold + 1,
                'best_val_accuracy': best_val_accuracy,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'val_losses': val_losses,
                'val_accuracies': val_accuracies,
            })

            logger.info(f"Fold {fold + 1} training time: {end_time - start_time:.2f} seconds")

    except KeyboardInterrupt:
        print("\nTraining process interrupted by user. Cleaning up...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
    finally:
        print("Training process completed or interrupted. Final cleanup...")
        # Perform any final cleanup operations here if needed

    # Calculate average performance
    if fold_results:
        avg_best_val_accuracy = np.mean([fold['best_val_accuracy'] for fold in fold_results])
        std_best_val_accuracy = np.std([fold['best_val_accuracy'] for fold in fold_results])
        logger.info(
            f"Average best validation accuracy across all folds: {avg_best_val_accuracy:.2f}% ± {std_best_val_accuracy:.2f}%")

        # Plot results
        plot_results(fold_results, save_dir)

        # Save results
        torch.save({
            'fold_results': fold_results,
            'avg_best_val_accuracy': avg_best_val_accuracy,
            'std_best_val_accuracy': std_best_val_accuracy,
            'config': config.__dict__
        }, os.path.join(save_dir, 'k_fold_results.pth'))

        logger.info(f"Training completed. Results saved in {os.path.join(save_dir, 'k_fold_results.pth')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--resume', action='store_true', help='Resume training from the latest checkpoint')
    args = parser.parse_args()

    main(args)