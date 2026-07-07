import copy
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from modules.utils import (
    setup_tensors,
    get_minibatches,
    setup_training_tensors
)


class SklearnStructure:

    def predict_proba(self, X, device='cpu'):
        self.model.eval()

        with torch.no_grad():
            X_tensor, _ = setup_tensors(X, device=device)
            pred, _ = self.model(X_tensor, None)

        self.model.train()

        return F.softmax(pred, dim=-1).cpu().numpy()

    def predict(self, X, device='cpu'):
        return np.argmax(
            self.predict_proba(X, device=device),
            axis=-1
        )

    def get_model_size(self):
        print("Missing Function")
        return 0

    def train_model(
        self,
        X,
        y,
        batchsize=32,
        lr='auto',
        iterations=2000,
        device='cpu',
        validation=0,
        track=None,
        patience=500,
        verbose=False
    ):

        self.model.to(device)

        if lr == 'auto':
            lr = 1e-3

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        X_tensor, y_tensor, X_val_tensor, y_val_tensor = \
            setup_training_tensors(validation, X, y, device)

        X_track_tensor = None
        y_track_tensor = None

        if track is not None:
            X_track, y_track = track
            X_track_tensor, y_track_tensor = setup_tensors(
                X_track,
                y_track,
                device
            )

        # Variáveis para early stopping
        if validation != 0:
            best_loss = float('inf')
            best_fit = copy.deepcopy(
                self.model.state_dict()
            )
            stopped_iterations = 0

        for it in range(iterations):

            X_batch, y_batch = get_minibatches(
                X_tensor,
                y_tensor,
                batchsize
            )

            pred, loss = self.model(
                X_batch,
                y_batch
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # -------------------------
            # Validação
            # -------------------------
            if validation != 0:

                self.model.eval()

                with torch.no_grad():
                    _, loss_val = self.model(
                        X_val_tensor,
                        y_val_tensor
                    )

                self.model.train()

                loss_val_value = loss_val.item()

                if loss_val_value < best_loss:

                    best_loss = loss_val_value

                    best_fit = copy.deepcopy(
                        self.model.state_dict()
                    )

                    stopped_iterations = 0

                else:
                    stopped_iterations += 1

                if stopped_iterations >= patience:

                    if verbose:
                        print(
                            f"\nEarly stopping "
                            f"na iteração {it}"
                        )

                    break

            # -------------------------
            # Tracking
            # -------------------------
            if track is not None and verbose:

                if it % 10 == 0:

                    self.model.eval()

                    with torch.no_grad():

                        pred_track, loss_track = self.model(
                            X_track_tensor,
                            y_track_tensor
                        )

                        acc_track = accuracy_score(
                            y_track_tensor.cpu().numpy(),
                            pred_track.argmax(1).cpu().numpy()
                        )

                    self.model.train()

                    print(
                        f"Iter {it:5d} | "
                        f"Train loss: {loss.item():.4f} | "
                        f"Track loss: {loss_track.item():.4f} | "
                        f"Track acc: {acc_track:.4f}"
                    )

            elif verbose and it % 10 == 0:

                print(
                    f"Iter {it:5d} | "
                    f"Train loss: {loss.item():.4f}"
                )

        # -------------------------
        # Recupera melhor modelo
        # -------------------------
        if validation != 0:
            self.model.load_state_dict(best_fit)

        # -------------------------
        # Avaliação final
        # -------------------------
        final_acc = None

        if track is not None:

            self.model.eval()

            with torch.no_grad():

                pred_track, loss_track = self.model(
                    X_track_tensor,
                    y_track_tensor
                )

                final_acc = accuracy_score(
                    y_track_tensor.cpu().numpy(),
                    pred_track.argmax(1).cpu().numpy()
                )

            self.model.train()

            print("\nFinal result:")
            print(
                f"Loss: {loss_track.item():.4f} | "
                f"Acc: {final_acc:.4f}"
            )

        return final_acc