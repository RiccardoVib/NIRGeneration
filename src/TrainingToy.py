import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from CheckpointManager import CheckpointManager
from DatasetClassH5 import H5Dataset
from pathlib import Path
from UtilsForTrainings import predict_waves
from utils import save_losses, plot_losses
from ConvModel import IR_Conv
from FFTModel import FFTConv, FFTConv_real
from IIRModel import IIR, IIR_real
from Evaluation import plot_tf_evaluation
from STFTloss import STFTLoss

script_path = Path(__file__).resolve()
script_dir = script_path.parent


def train(**kwargs):
    seq_len = kwargs.get('seq_len', 1)
    batch_size = kwargs.get('batch_size', 1)
    lr = kwargs.get('lr', 1e-1)
    filter_length = kwargs.get('filter_length', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    dataset = kwargs.get('dataset', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)
    fs = kwargs.get('fs', 48000)
    cond_dim = kwargs.get('cond_dim', 1)
    model_t = kwargs.get('model_t', "ir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('cuda available :', torch.cuda.is_available())

    model_path = script_dir.parent.parent / "TrainedModels" / save_folder

    print(f"model_name: {save_folder}")
    print(f"model_path: {model_path}")

    # Set seeds for reproducibility
    # np.random.seed(42)
    # random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)

    # start the timer for all the training script
    global_start = time.time()

    # Load test dataset
    test_dataset = H5Dataset(Path(data_dir) / f"{dataset}_test.h5", seg_len=seq_len, overlap=seq_len//4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_dataset = H5Dataset(Path(data_dir)/ f"{dataset}_train.h5", seg_len=seq_len, overlap=seq_len//4)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model depending on type
    if model_t == "ir":
        model = IR_Conv(filter_length=filter_length, cond_dim=cond_dim)
    elif model_t == "fft":
        model = FFTConv(filter_length=filter_length, cond_dim=cond_dim)
    elif model_t == "iir":
        model = IIR(filter_length=filter_length, cond_dim=cond_dim)
    elif model_t == "fft_real":
        model = FFTConv_real(filter_length=filter_length, cond_dim=cond_dim)
    elif model_t == "iir_real":
        model = IIR_real(filter_length=filter_length, cond_dim=cond_dim)
    else:
        model = None

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    print(f"Dataset length: {len(test_loader)}")
    print('\n train batch_size', batch_size)
    print('\n epochs ', epochs)
    print('\n lr ', lr)
    print('\n seq_len ', seq_len)
    print('\n')
    print(all(p.is_cuda for p in model.parameters()))  # True if all params on GPU

    lr_count = 0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        eps=1e-07,
        betas=(0.9, 0.999),
        weight_decay=0
    )
    # # Define the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # Explicitly set (loss should decrease)
        patience=1,           # Wait 2 epochs before reducing LR
        factor=0.75,           # Multiply LR by 0.5 (default is 0.1)
        threshold=1e-6,       # Increase threshold for more sensitivity
        )

    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(model_path / "my_checkpoints")

    # Define loss function
    criterion = torch.nn.MSELoss()
    criterion = STFTLoss().to(device)

    if epochs > 0:
        # Load last checkpoint
        checkpoint = ckpt_manager.load_last_checkpoint(model, optimizer, scheduler, device='cpu')

        if checkpoint:
            # print(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_val_loss']
            print(f"Resuming from epoch {start_epoch}, best metric: {best_loss}")
        else:
            print("Starting training from scratch")
            best_loss = float('inf')

        loss_training = []
        loss_val = []

        for epoch in range(epochs):
            model.train()
            start = time.time()
            print('Epoch:', epoch)

            train_loss_accum = 0.0
            for (inputs, targets) in train_loader:
                (X, C) = inputs
                X, C, targets = X.to(device), C.to(device), targets.to(device)

                loss = model.train_step(X, C, targets, optimizer, criterion)

                train_loss_accum += loss

            avg_train_loss = train_loss_accum / len(train_loader)
            loss_training.append(avg_train_loss)

            model.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for (inputs, targets) in test_loader:
                    (X, C) = inputs
                    X, C, targets = X.to(device), C.to(device), targets.to(device)

                    loss = model.val_step(X, C, targets, criterion)

                    val_loss_accum += loss

            avg_val_loss = val_loss_accum / len(test_loader)
            loss_val.append(avg_val_loss)

            print(f"Validation Loss: {avg_val_loss}")

            # Early stopping and learning rate scheduling
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # Save best weights
                state_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_loss
                }
                ckpt_manager.save_checkpoint(state_dict, is_best=True)
                print(f"Epoch {epoch + 1}, Validation loss improved: ", best_loss)
                print(f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
                lr_count = 0
            else:
                lr_count += 1
                print(f"Epoch {epoch + 1}, Validation loss did not improved. Best val loss: ", best_loss)
                if lr_count == 50:
                    print("Early stopping triggered.")
                    break

            avg_time_epoch = time.time() - start
            print(f"Average time/epoch: {avg_time_epoch / 60:.3f} min")

            # Save latest checkpoint
            state_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_loss
            }

            # Save last checkpoint
            ckpt_manager.save_last_checkpoint(state_dict)

            # Update learning rate scheduler with validation loss
            scheduler.step(avg_val_loss)

        filename_ = model_path / ('losses.json')
        save_losses(train_losses=loss_training, val_losses=loss_val, filename=filename_)
        filename_ = model_path / ('loss_plot.png')
        plot_losses(train_losses=loss_training, val_losses=loss_val, filename=filename_)

        # Final reporting
        print("Training done")

    avg_time_training = time.time() - global_start
    print(f"Average time training: {avg_time_training / 60:.3f} min")

    # Load best checkpoint
    best_checkpoint = ckpt_manager.load_best_checkpoint(model, device='cpu')
    if best_checkpoint:
        best_loss = best_checkpoint.get('best_val_loss', 0)
        print(f"Loaded best model with metric: {best_loss}")
    else:
        print(f"Problem!!!!")

    model.eval()

    x_list, y_list, z_list = [], [], []
    with torch.no_grad():
        for (X, C), Y in test_loader:
            x_list.append(X)  # (B, seq_len, 1)
            y_list.append(Y)  # (B, seq_len, 1)
            z_list.append(C)  # (B, 1, cond_dim)

    x_test = torch.cat(x_list, dim=0).to(device)  # (N, seq_len, 1)
    y_test = torch.cat(y_list, dim=0).to(device)  # (N, seq_len, 1)
    z_test = torch.cat(z_list, dim=0).to(device)  # (N, 1, cond_dim)

    predictions = []
    for x, z in zip(x_test, z_test):
        prediction = model(x.reshape(1, seq_len, 1), z.reshape(1, 1, -1))
        prediction = prediction.reshape(1, seq_len, 1)
        predictions.append(prediction)

    predictions = torch.stack(predictions, dim=0).squeeze(1)
    predict_waves(predictions.detach().cpu().numpy(), x_test.cpu().numpy(), y_test.cpu().numpy(), z_test, None, model_save_dir,
                  save_folder, fs, save_folder)

    idxs = [0, 50, 100]
    for i in idxs:
        # plot TF magnitude and phase for all test samples
        plot_tf_evaluation(
            predictions=predictions.detach().cpu().numpy(),  # (N, seq_len)
            targets=y_test.cpu().numpy(),  # (N, seq_len)
            fs=fs,
            save_dir=model_save_dir,
            save_folder=save_folder,
            sample_idx=i,  # change or loop over multiple samples
            nfft=seq_len,
        )


    test_loss = criterion(predictions, y_test)
    save_path_txt = os.path.join(model_save_dir, save_folder, f'results.txt')
    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)

    with open(save_path_txt, 'w') as f:
        f.write(f'test_loss : {test_loss}\n')

    return 42
