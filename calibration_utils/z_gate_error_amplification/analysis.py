import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V


@dataclass
class FitParameters:
    """Fit results for a single qubit."""

    pi_amplitude: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q, res in fit_results.items():
        status = "SUCCESS" if res["success"] else "FAIL"
        log_callable(
            f"Qubit {q}: {status} — pi_amplitude = {res['pi_amplitude'] * 1e3:.3f} mV\n"
        )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    if not node.parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, node.namespace["qubits"])

    qubits = node.namespace["qubits"]
    operation = node.parameters.operation
    # Use amplitude captured at program creation time to stay accurate on data reload
    base_amps = (
        node.namespace.get("sweep_base_amp")
        or node.results.get("sweep_base_amp")
        or {q.name: q.z.operations[operation].amplitude for q in qubits}
    )
    amp_factors = ds.amp.values
    abs_amp = np.array([base_amps[q.name] * amp_factors for q in qubits])
    ds = ds.assign_coords({"abs_amp": (["qubit", "amp"], abs_amp)})
    ds["abs_amp"].attrs = {"long_name": "Z amplitude", "units": "V"}
    ds.attrs["operation"] = operation
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    y_name = "state" if node.parameters.use_state_discrimination else "I"
    # Average over the N-pulse axis to find the amplitude with minimum signal
    I_n = ds[y_name].mean(dim="N")
    best_idx = I_n.argmin(dim="amp")

    qubits = node.namespace["qubits"]
    n_amps = ds.sizes["amp"]
    optimal_amps, successes = [], []

    for q in qubits:
        idx = int(best_idx.sel(qubit=q.name).values)
        pi_amp = float(ds.abs_amp.sel(qubit=q.name).isel(amp=idx).values)
        # Fail if the minimum is at the sweep edge (sweep range too narrow)
        at_edge = idx == 0 or idx == n_amps - 1
        optimal_amps.append(pi_amp)
        successes.append(not at_edge and pi_amp > 0)

    optimal_amp_da = xr.DataArray(
        optimal_amps,
        dims=["qubit"],
        coords={"qubit": [q.name for q in qubits]},
        attrs={"long_name": "Optimal Z amplitude", "units": "V"},
    )
    ds_fit = ds.assign({"I_n_avg": I_n, "optimal_amp": optimal_amp_da})

    fit_results = {
        q.name: FitParameters(pi_amplitude=opt_amp, success=suc)
        for q, opt_amp, suc in zip(qubits, optimal_amps, successes)
    }
    return ds_fit, fit_results
