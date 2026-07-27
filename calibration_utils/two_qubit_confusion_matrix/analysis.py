import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

NUM_STATES = 4  # 2^2 for a qubit pair
STATE_LABELS = ["00", "01", "10", "11"]


# ---------------------------------------------------------------------------
## Data class


@dataclass
class FitParameters:
    """Readout confusion results for a single qubit pair."""

    confusion_matrix: List[List[float]]
    assignment_fidelity: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qp_name, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Qubit pair {qp_name}: {status} - assignment fidelity = {res['assignment_fidelity']:.4f}"
        )


# ---------------------------------------------------------------------------
## Analysis


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Build the 4x4 readout confusion matrix of every qubit pair.

    The matrix is indexed as conf[measured, prepared] = P(measured | prepared), so every column
    sums to 1 and p_measured = conf @ p_true. Readout error mitigation is therefore inv(conf),
    with no transpose, which is what 21b_Bell_state_tomography expects.

    Note this is the transpose of the single-qubit qubit.resonator.confusion_matrix written by
    07_iq_blobs, which stores [prepared][measured] and does need a transpose when inverted.
    """
    qubit_pairs = node.namespace["qubit_pairs"]

    confusion_all = []
    fit_results = {}
    for qp in qubit_pairs:
        conf = np.zeros((NUM_STATES, NUM_STATES))
        for control in (0, 1):
            for target in (0, 1):
                prepared = control * 2 + target
                shots = ds["state"].sel(
                    qubit_pair=qp.name,
                    init_state_control=control,
                    init_state_target=target,
                ).values
                for measured in range(NUM_STATES):
                    conf[measured, prepared] = np.count_nonzero(shots == measured) / shots.size

        assignment_fidelity = float(np.mean(np.diag(conf)))
        # A usable confusion matrix needs the correct outcome to be the most likely one
        success = bool(np.all(np.argmax(conf, axis=0) == np.arange(NUM_STATES)))

        confusion_all.append(conf)
        fit_results[qp.name] = FitParameters(
            confusion_matrix=conf.tolist(),
            assignment_fidelity=assignment_fidelity,
            success=success,
        )

    ds_fit = ds.assign(
        confusion=xr.DataArray(
            # reshape keeps the array 3D even when no qubit pair was selected
            np.array(confusion_all).reshape(len(confusion_all), NUM_STATES, NUM_STATES),
            dims=["qubit_pair", "measured", "prepared"],
            coords={
                "qubit_pair": [qp.name for qp in qubit_pairs],
                "measured": np.arange(NUM_STATES),
                "prepared": np.arange(NUM_STATES),
            },
            attrs={"long_name": "P(measured | prepared)"},
        )
    )
    return ds_fit, fit_results
