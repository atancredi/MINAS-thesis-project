from random import sample

import torch
import matplotlib.pyplot as plt

def test_model(model, criterion, x_test, y_test, n: int, output_path: str):
    # TODO test on all the test set and plot some spectra with different loss values (high, medium, low)

    
    total_samples = len(x_test)
    indices = sample(range(total_samples), k=n)
    print(f"indices", indices)

    fig, ax = plt.subplots(ncols=n,nrows=1,figsize=(int(5*n), 3))
    if n==1:
        ax = [ax]
    
    for i, idx in enumerate(indices):

        x_sample = x_test[idx] 
        y_sample = y_test[idx]

        x_tensor = torch.as_tensor(x_sample, dtype=torch.float32).view(1, -1)
        with torch.no_grad():
            prediction = model(x_tensor)
        y_pred = prediction.cpu()

        y_true = torch.as_tensor(y_sample, dtype=torch.float32).view(1, -1)

        total = criterion(y_pred, y_true)

        print(f"loss {total.item():.5f}")

        y_true_np = y_true.detach().numpy().flatten()
        y_pred_np = y_pred.detach().numpy().flatten()

        ax[i].plot(y_true_np, label='Ground Truth', color='black', linewidth=2, linestyle='--')
        ax[i].plot(y_pred_np, label='MLP Prediction', color='#d62728', linewidth=2)
        ax[i].fill_between(range(len(y_true_np)), y_true_np, y_pred_np, color='gray', alpha=0.2, label='Error')
        ax[i].set_title(f"Loss Analysis (Idx {idx}): Total={total.item():.4f}")
        ax[i].grid(True, alpha=0.3)
    
    plt.legend()
    plt.savefig(output_path)