from eval import Evaluator
from models import Linear
from tools import ReconstructDataset
from types import SimpleNamespace
import torch, math
import numpy as np

def all_errors(data, model, win_size, device, batch_size):
    model = model.to(device)
    model.eval()
    ds = ReconstructDataset(
        data, window_size=win_size, stride=1, normalize=False)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, drop_last=False, )
    outs = []
    with torch.inference_mode():
        for xb in dl:
            xb = xb.to(device).float()
            out = model(xb)

            error = torch.mean((out - xb) ** 2, dim=-1)  # (B,W)
            outs.append(error.cpu())

    output = torch.cat(outs, dim=0)
    return output

def disjoint_error(all_error, n, win_size):
    rec_err = torch.zeros(n)
    for i, err in enumerate(all_error[::win_size]):
        start = i * win_size
        end = min(start + win_size, n)
        rec_err[start:end] = err.squeeze()
    
    # handle last partial window
    if n % win_size != 0:
        rec_err[n - (n % win_size): n] = all_error[-1].squeeze()[-(n % win_size):]
    return rec_err.cpu()

def overlapping_error(all_error, n, win_size, stride):
    rec_err = torch.zeros(n)
    count = torch.zeros(n)

    for i in range(0, all_error.shape[0], stride):
        start = i
        end = start + win_size
        if end > n:
            end = n
        rec_err[start:end] += all_error[i].squeeze()
        count[start:end] += 1

    rec_err /= count.clamp_min(1)
    return rec_err.cpu()

def mse_error(all_error, n, win_size):
    
    rec_err = all_error.mean(dim=-1)
    print(rec_err.shape)
    # rec_err = torch.ones(n-win_size+1) * mses.item()

        
    rec_err = np.array([rec_err[0]]*math.ceil((win_size-1)/2) +
                        list(rec_err) + [rec_err[-1]]*((win_size-1)//2))
    
    return torch.tensor(rec_err).cpu()



# ------------------
# Simple test
# ------------------
def test_reconstruction():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator = Evaluator(batch_size=20, device=device)
    data = torch.randn(1000, dtype=torch.float32).reshape(-1,1).to(device)
    win_size = 32


    config = SimpleNamespace(
        task_name='anomaly_detection',
        seq_len=win_size,
    )
    model = Linear.Model(config).to(device)

    errors = all_errors(data, model, win_size, device, 20)

    err_overlap = evaluator._overlapping_reconstruction(data, model, win_size, stride=1).cpu()
    err_disjoint = evaluator._disjoint_reconstruction(data, model, win_size).cpu()
    err_mse = evaluator._mse_reconstruction(data, model, win_size).cpu()


    my_err_disjoint = disjoint_error(errors, len(data), win_size)
    my_err_overlapping = overlapping_error(errors, len(data), win_size, stride=1)
    my_err_mse = mse_error(errors, len(data), win_size)
    

    print(f"Disjoint average: {torch.mean(err_disjoint).item()}, My Disjoint MSE: {torch.mean(my_err_disjoint).item()}")
    assert torch.allclose(err_disjoint, my_err_disjoint), "Disjoint errors do not match!"

    print(f"Overlapping average: {torch.mean(err_overlap).item()}, My Overlapping MSE: {torch.mean(my_err_overlapping).item()}")
    assert torch.allclose(err_overlap, my_err_overlapping), "Overlapping errors do not match!"

    print(f"MSE average: {torch.mean(err_mse).item()}, My MSE: {torch.mean(my_err_mse).item()}")
    assert torch.allclose(err_mse, my_err_mse), "MSE errors do not match!"



    my_err_overlapping_2 = overlapping_error(errors, len(data), win_size, stride=2)
    err_overlap_2 = evaluator._overlapping_reconstruction(data, model, win_size, stride=2).cpu()

    print(f"Overlapping (stride=2) average: {torch.mean(err_overlap_2).item()}, My Overlapping (stride=2) MSE: {torch.mean(my_err_overlapping_2).item()}")
    assert torch.allclose(err_overlap_2, my_err_overlapping_2), "Overlapping (stride=2) errors do not match!"

    my_err_overlapping_4 = overlapping_error(errors, len(data), win_size, stride=4)
    err_overlap_4 = evaluator._overlapping_reconstruction(data, model, win_size, stride=4).cpu()

    print(f"Overlapping (stride=4) average: {torch.mean(err_overlap_4).item()}, My Overlapping (stride=4) MSE: {torch.mean(my_err_overlapping_4).item()}")
    assert torch.allclose(err_overlap_4, my_err_overlapping_4), "Overlapping (stride=4) errors do not match!"

    my_err_overlapping_8 = overlapping_error(errors, len(data), win_size, stride=8)
    err_overlap_8 = evaluator._overlapping_reconstruction(data, model, win_size, stride=8).cpu()

    print(f"Overlapping (stride=8) average: {torch.mean(err_overlap_8).item()}, My Overlapping (stride=8) MSE: {torch.mean(my_err_overlapping_8).item()}")
    assert torch.allclose(err_overlap_8, my_err_overlapping_8), "Overlapping (stride=8) errors do not match!"
    
# ------------------
# Run
# ------------------
if __name__ == "__main__":
    # assumes your class is already instantiated as `tester`
    # e.g. tester = MyReconstructionClass(device="cpu", batch_size=2)

    
    test_reconstruction()
