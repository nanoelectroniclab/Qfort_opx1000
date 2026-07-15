import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode

NUM_STATES = 32  # 2^5 for 5 qubits


@dataclass
class FitParameters:
    """Analysis results for a single quintet."""

    ghz_fidelity: float
    corrected_probs: List[float]
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for name, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(f"Quintet {name}: {status} — GHZ fidelity = {res['ghz_fidelity']:.3f}")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    n_shots = node.parameters.num_shots
    quintet_names = node.results["quintet_names"]

    raw_probs_list = []
    for qname in quintet_names:
        state_shots = ds["state"].sel(quintet=qname).values  # (shot,)
        counts = np.array([(state_shots == s).sum() for s in range(NUM_STATES)], dtype=float)
        raw_probs_list.append(counts / n_shots)

    raw_probs = np.array(raw_probs_list)  # (num_quintets, 32)
    ds = ds.assign(
        raw_probs=xr.DataArray(
            raw_probs,
            dims=["quintet", "state_label"],
            coords={"quintet": quintet_names, "state_label": np.arange(NUM_STATES)},
            attrs={"long_name": "Raw state probability"},
        )
    )
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    quintet_names = node.results["quintet_names"]
    conf_matrices = node.results["confusion_matrices"]

    corrected_list = []
    fit_results = {}
    for qname in quintet_names:
        raw_p = ds["raw_probs"].sel(quintet=qname).values  # (32,)
        conf_mats = conf_matrices[qname]  # list of 5 (2,2) arrays

        conf_mat_5q = np.array([[1.0]])
        for cm in conf_mats:
            conf_mat_5q = np.kron(conf_mat_5q, np.array(cm))  # builds 32×32 matrix

        corrected = np.linalg.inv(conf_mat_5q) @ raw_p
        corrected = np.clip(corrected, 0, None)
        corrected /= corrected.sum()

        # GHZ fidelity in Z basis: P(|00000⟩) + P(|11111⟩)
        ghz_fidelity = float(corrected[0] + corrected[31])
        corrected_list.append(corrected)
        fit_results[qname] = FitParameters(
            ghz_fidelity=ghz_fidelity,
            corrected_probs=corrected.tolist(),
            success=ghz_fidelity > 0.5,
        )

    corrected_arr = np.array(corrected_list)
    ds = ds.assign(
        corrected_probs=xr.DataArray(
            corrected_arr,
            dims=["quintet", "state_label"],
            coords={"quintet": quintet_names, "state_label": np.arange(NUM_STATES)},
            attrs={"long_name": "Confusion-matrix-corrected probability"},
        )
    )
    return ds, fit_results
