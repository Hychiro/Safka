import numpy as np
import torch


def get_minibatches(X, y, batchsize):
    batch = np.array([i for i in range(len(X))])
    np.random.shuffle(batch)
    return X[batch[:batchsize]], y[batch[:batchsize]]


def setup_tensors(X, y=None, device='cuda'):
    if X.ndim != 3:
        X = X.reshape(X.shape[0], X.shape[2], X.shape[3])
    X_temp_tensor = torch.tensor(np.array(X).astype(np.float32)).to(dtype=torch.float32, device=device)

    if y is not None:
        y_temp_tensor = torch.tensor(np.array(y).astype(np.long)).to(dtype=torch.long, device=device)
    else:
        y_temp_tensor = None

    return X_temp_tensor, y_temp_tensor


def setup_training_tensors(validation, X, y=None, device='cuda'):
    X_train, y_train = X, y

    if isinstance(validation, tuple):
        X_val_tensor, y_val_tensor = validation
        X_val_tensor, y_val_tensor = setup_tensors(X_val_tensor, y_val_tensor, device)

    elif validation > 0:
        cut_val = int(len(y) * validation)
        X_val_tensor, y_val_tensor = X[:cut_val], y[:cut_val]
        X_val_tensor, y_val_tensor = setup_tensors(X_val_tensor, y_val_tensor, device)
        X_train, y_train = X[cut_val:], y[cut_val:]
    else:
        X_val_tensor, y_val_tensor = None, None

    X_tensor, y_tensor = setup_tensors(X_train, y_train, device)

    return X_tensor, y_tensor, X_val_tensor, y_val_tensor