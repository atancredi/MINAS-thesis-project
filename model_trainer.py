import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shutil import copy2

from utils.config import TrainingConfig


class ModelTrainer:

    def __init__(
        self,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        scheduler,
        augmenter = None,
        loss_scheduler = None
    ):    
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loss_history = []
        self.val_loss_history = []
        self.lr_history = []
        self.best_val_loss = float('inf')
        self.best_val_epoch = 0

        self.augmenter = augmenter
        self.loss_scheduler = loss_scheduler


    def training_loop(self, model, validator, config: TrainingConfig, inverse: bool):
        
        tq = tqdm(range(config.epochs))
        for epoch in tq:
            model.train() # model in training mode
            running_train_loss = 0.0

            if self.loss_scheduler:
                current_weights = self.loss_scheduler.step(epoch)
        
                if epoch % 10 == 0:
                    print(f"Epoch {epoch} Weights: {current_weights}")


            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.view(inputs.size(0), -1)
                targets = targets.view(targets.size(0), -1)

                if self.augmenter != None:
                    targets = self.augmenter(targets)
                
                if inverse:
                    inputs = targets

                self.optimizer.zero_grad()
                outputs = model(inputs)
                prediction = outputs
                
                if isinstance(outputs, tuple):
                    prediction = outputs[1].squeeze()
                    targets = targets.squeeze()

                loss = self.loss_function(prediction, targets)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            # average training loss
            avg_train_loss = running_train_loss / len(self.train_loader)
            self.train_loss_history.append(avg_train_loss)

            # validation loss
            avg_val_loss = validator(model, self.val_loader, self.loss_function)
            self.val_loss_history.append(avg_val_loss)

            # Step the lr scheduler
            self.scheduler.step(avg_val_loss)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            tq.set_description_str(f'Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {current_lr:.6f} | Best Val: {self.best_val_loss:.5f} ({self.best_val_epoch})')
            self.lr_history.append(current_lr)

            # early stopping: save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                if "w_amp" in self.loss_function.__dict__:
                    checkpoint['loss_config'] = {
                        'w_amp': self.loss_function.w_amp,
                        'w_grad': self.loss_function.w_grad,
                        'w_wass': self.loss_function.w_wass,
                        'w_sam': self.loss_function.w_sam
                    }
                torch.save(checkpoint, config.model_path)
                self.best_val_epoch = epoch


    def training_stats(self, config: TrainingConfig):
        # save parameters file to model folder
        model_params_path = f"{config.model_dir}/{config.model_name}_params.env"
        copy2(config.config_path, model_params_path)

        # plot train and val loss
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        # lr drops
        lr = np.array(self.lr_history)
        drop_epochs = np.where(lr[1:] < lr[:-1])[0] + 1  # +1 to align epoch index

        axes[0].plot(self.train_loss_history, label='Training Loss')
        axes[0].plot(self.val_loss_history, label='Validation Loss', linestyle='--')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for e in drop_epochs:
            axes[0].axvline(e, color='red', linestyle='--', alpha=0.6)
            axes[0].text(
                e, axes[0].get_ylim()[1],
                f"LR → {lr[e]:.1e}",
                color='red',
                fontsize=9,
                rotation=90,
                verticalalignment='top',
                horizontalalignment='right'
            )

        axes[1].plot(self.train_loss_history, label='Training Loss', alpha=0.8)
        axes[1].plot(self.val_loss_history, label='Validation Loss', linestyle='--', alpha=0.8)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss (Log Scale)')
        axes[1].grid(True, which="both", alpha=0.3)

        for e in drop_epochs:
            axes[1].axvline(e, color='red', linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(f"{config.model_dir}/{config.model_name}_loss_curve.png", dpi=300)
        plt.close()
        