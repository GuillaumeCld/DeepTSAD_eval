import numpy as np
import torch

import tools as utils
import inference as inference
from evaluation.metrics import get_metrics, get_metrics_restr
import math

class Evaluator:

    def __init__(self, batch_size=32, device='cpu', metrics='restr', strategy='overlapping'):

        self.batch_size = batch_size
        self.device = device

        if metrics == 'all':
            self.metrics = ["AUC-PR", "AUC-ROC",
                            "VUS-PR", "VUS-ROC", "Standard-F1"]
            self.metrics_fnc = get_metrics
        elif metrics == 'restr':
            self.metrics = ["AUC-PR", "AUC-ROC", "Standard-F1"]
            self.metrics_fnc = get_metrics_restr
        else:
            raise ValueError(f"Unknown metrics setting: {metrics}")
        
        self.strategy = strategy
        if strategy == "overlapping":
            self.inference_strategy = inference.combined_pointwise_profile_median
        elif strategy == "disjoint":
            self.inference_strategy = inference.disjoint_pointwise_profile
        elif strategy == "MSE":
            self.inference_strategy = inference.rowwise_mse
        else:
            raise ValueError(
                f"Unknown inference strategy: {strategy}")


    

    def evaluate(self, data, label, model, win_size):

        reconstruction_error = self.reconstruction_error(data, model, win_size)

        rank = utils.find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        result = self.metrics_fnc(
            reconstruction_error, label, slidingWindow=rank)

        return result


    def reconstruction_error(self, data, model, win_size):

        model = model.to(self.device)
        model.eval()

        data = data.astype(np.float32)
        n = data.shape[0]


        ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=1, normalize=False
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        outs = []
        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                outs.append(out.cpu())

        output = torch.cat(outs, dim=0).numpy()[:, :, 0]

        data_seq = inference.make_sequences_1d(
            data.ravel(), win_size)  # (n - w + 1, w)
        pw_error = inference.squared_pointwise_error_numpy(data_seq, output)
        reconstruction_error = self.inference_strategy(
            pw_error, n, win_size)

        # Pad to match original length when using "MSE" strategy
        if reconstruction_error.shape[0] < len(data):
            reconstruction_error = np.array([reconstruction_error[0]]*math.ceil((win_size-1)/2) + 
                        list(reconstruction_error) + [reconstruction_error[-1]]*((win_size-1)//2))

        return reconstruction_error
    

    def reconstruct(self, data, model, win_size):

        model = model.to(self.device)
        model.eval()

        data = data.astype(np.float32)
        n = data.shape[0]


        ds = utils.ReconstructDataset(
            data, window_size=win_size, stride=1, normalize=False
        )
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        outs = []
        with torch.inference_mode():
            for xb in dl:
                xb = xb.to(self.device).float()
                out = model(xb)

                outs.append(out.cpu())

        output = torch.cat(outs, dim=0).numpy()[:, :, 0]

        
        reconstruction = self.inference_strategy(
            output, n, win_size)

        return reconstruction