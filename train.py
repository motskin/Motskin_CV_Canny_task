import argparse
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from termcolor import colored
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data_loading import BasicDataset
from edge_detector_model import EdgeDetectorModel
from utils.dice_score import dice_coeff, dice_loss, multiclass_dice_coeff
from utils.draw import draw_train_statistics
from utils.progress_bar import ProgressBar

# parse command-line arguments
parser = argparse.ArgumentParser(description='PyTorch edge detector like Canny Training')

parser.add_argument('--train-images-dir', type=str, metavar='PATH',
                    help='path to train images')
parser.add_argument('--train-masks-dir', type=str, metavar='PATH',
                    help='path to train masks')
parser.add_argument('--val-images-dir', type=str, metavar='PATH',
                    help='path to validation images')
parser.add_argument('--val-masks-dir', type=str, metavar='PATH',
                    help='path to validation masks')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) ')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.95, type=float,
                     help='Momentum for optimizer')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--save-checkpoint', action='store_true',
                    help='Should save checkpoints')
parser.add_argument('--checkpoint-folder-name', type=str, default="checkpoints",
                    help='Folder name where we save checkpoints if use argument "--save-checkpoint"')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

args = parser.parse_args()

# Draw table with argument values
print(colored("\nArguments:", 'yellow'))
print("-" * 145)
print(f"|          Name          | {' ' * 55} Value {' ' * 55}|")
print("-" * 145)
for arg, value in vars(args).items():
    print(f"| {arg:>22} | {value:<116} |")
print("-" * 145)

# setting the device on which we will teach
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {colored(device.type, "cyan")}')

# Create Dataset
train_dataset = BasicDataset(images_dir=args.train_images_dir, masks_dir=args.train_masks_dir, resolution=args.resolution)
val_dataset = BasicDataset(images_dir=args.val_images_dir, masks_dir=args.val_masks_dir, resolution=args.resolution)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

model = EdgeDetectorModel()
model = model.to(device=device)

# Initializing the optimizer
optimizer = optim.RMSprop(model.parameters(),
                            lr=args.lr, weight_decay=1e-8, momentum=args.momentum, foreach=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
criterion = nn.CrossEntropyLoss()


train_epoch_statistic = dict()    # словарь, который будет накапливать статистику обучения
matplotlib.use('Agg')   # устраняем проблему падения в многопотоке https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

gradient_clipping = 1.0

for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    
    print(colored(f"\nEPOCH {epoch + 1}", 'yellow'))
    progress = ProgressBar(len(train_dataloader), f"Training: {epoch+1} / {args.epochs}:")

    for idx, batch in enumerate(train_dataloader):
        input_images, target_edges = batch['image'], batch['mask']
        input_images = input_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        target_edges = target_edges.to(device=device, dtype=torch.long)

        # predict and loss calculation
        masks_pred = model(input_images)
        loss = criterion(masks_pred, target_edges)
        loss += dice_loss(
            F.softmax(masks_pred, dim=1).float(),
            F.one_hot(target_edges, 2).permute(0, 3, 1, 2).float()    # 2 number of classes
        )

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        progress.update(idx)

    print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

    # Validation:
    val_score = 0
    progress = ProgressBar(len(val_dataloader), f"Validating:")
    for idx, batch in enumerate(val_dataloader):
        val_images, mask_true = batch['image'], batch['mask']
        val_images = val_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        mask_pred = model(val_images)

        mask_true = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
        mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
        # compute the Dice score, ignoring background
        val_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

        progress.update(idx)
    
    val_dice_coef = val_score.item() / len(val_dataloader)
    scheduler.step(val_dice_coef)

    print(colored(f"Validation dice coef: {val_dice_coef:.6f}", 'cyan'))
    
    train_epoch_statistic[epoch+1] = {
            'train_loss': loss.item(),
            'val_dice_coef': val_dice_coef,
        }

    # optimizer.step()
        
    if args.save_checkpoint:
        Path(args.checkpoint_folder_name).mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        state_dict['train_statistics'] = train_epoch_statistic
        torch.save(state_dict, f"{args.checkpoint_folder_name}/checkpoint_epoch_{epoch+1}.pth")
        print(f'Checkpoint {epoch+1} saved!')
    
    fig_last = draw_train_statistics(train_epoch_statistic)
    fig_last.savefig('last_training_chart.jpg')                             # сохраняем график в файл, где будут сохраняться последний график любого обучения
    plt.close()     


torch.save(model, 'last_model.pth')

plt.close('all')
matplotlib.use('TkAgg')                 # активируем интерактивный вывод графика
fig_last = draw_train_statistics(train_epoch_statistic)
plt.show()  


