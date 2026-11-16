"""Physical-space metrics; keep ensemble-mean and member-RMS errors distinct."""

import numpy as np


def evaluate_ensemble(ensemble, truth):
    if truth.shape[0] != 1 or ensemble.shape[1:] != truth.shape[1:]:
        raise ValueError("Expected ensemble (members,time,points,channels) and one matching truth.")
    mean = np.mean(ensemble, axis=0, keepdims=True)
    denominator = np.sum(truth ** 2, axis=(0, 2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_rmse = np.sqrt(np.sum((mean - truth) ** 2, axis=(0, 2, 3)) / denominator)
        error = np.sqrt(np.mean((mean[0] - truth[0]) ** 2, axis=(1, 2)))
        spread = np.sqrt(np.mean(np.std(ensemble, axis=0) ** 2, axis=(1, 2)))
        ser = spread / (error + 1e-12)
        member_rms = np.sqrt(np.mean((ensemble - truth) ** 2, axis=(0, 2, 3)) /
                             np.mean(truth ** 2, axis=(0, 2, 3)))
    return {"relative_rmse": relative_rmse, "ser": ser,
            "member_rms_relative_rmse": member_rms}
