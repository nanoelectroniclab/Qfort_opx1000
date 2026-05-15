import logging
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V
from qualibration_libs.analysis import peaks_dips


@dataclass
class FitParameters:
    """Stores the relevant resonator spectroscopy experiment fit parameters for a single qubit"""
    frequency: float
    fwhm: float
    success: bool
    best_twpa_power: float | None = None
    best_twpa_freq: float | None = None
    best_snr: float | None = None

def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
        
    for q in fit_results.keys():
        # 1. 第一行：印出 Qubit 名稱與成功/失敗狀態
        s_qubit = f"Results for qubit {q}: "
        if fit_results[q]["success"]:
            s_qubit += " SUCCESS!\n"
        else:
            s_qubit += " FAIL!\n"
            
        # 2. 第二行前半段：印出共振腔擬合參數
        s_freq = f"\tResonator frequency: {1e-9 * fit_results[q]['frequency']:.3f} GHz | "
        s_fwhm = f"FWHM: {1e-3 * fit_results[q]['fwhm']:.1f} kHz | \n"
        
        # 3. 第三行：提取 TWPA 新參數，並加入防呆機制 (避免 None 報錯)
        power = fit_results[q].get("best_twpa_power")
        freq = fit_results[q].get("best_twpa_freq")
        snr = fit_results[q].get("best_snr")
        
        # 判斷如果有算出來，就漂亮地印出 Power 和 Freq
        if power is not None and freq is not None:
            s_twpa = f"\tBest TWPA: {power:.2f} dBm, {freq / 1e3:.3f} GHz | "
        else:
            s_twpa = "\tBest TWPA: None | "
            
        # 判斷 SNR
        if snr is not None:
            s_snr = f"Best SNR: {snr:.2f}\n"
        else:
            s_snr = "Best SNR: None\n"
            
        # 4. 把所有字串組合起來，交給 logger 印出
        log_callable(s_qubit + s_freq + s_fwhm + s_twpa + s_snr)

def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = convert_IQ_to_V(ds, node.namespace["qubits"])
    ds = add_amplitude_and_phase(ds, "detuning", subtract_slope_flag=True)
    full_freq = np.array([ds.detuning + q.resonator.RF_frequency for q in node.namespace["qubits"]])
    ds = ds.assign_coords(full_freq=(["qubit", "detuning"], full_freq))
    ds.full_freq.attrs = {"long_name": "RF frequency", "units": "Hz"}
    return ds
    


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the T1 relaxation time for each qubit according to ``a * np.exp(t * decay) + offset``.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The QUAlibrate node.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # Fit the resonator line
    fit_results = peaks_dips(ds.IQ_abs, "detuning")
    # Compute SNR vs TWPA_amp (mean/std over detuning)
    snr, best_twpa_power, best_twpa_freq, best_snr = _compute_snr_vs_twpa_amp(ds)
    
    # Extract the relevant fitted parameters (and optionally best TWPA info)
    fit_data, fit_results = _extract_relevant_fit_parameters(
        fit_results, node, best_twpa_power=best_twpa_power, best_twpa_freq=best_twpa_freq, best_snr=best_snr
    )
    # Attach SNR info to dataset and results
    if snr is not None:
        fit_data = fit_data.assign(snr=snr)
        fit_data = fit_data.assign_coords(
            best_twpa_power=("qubit", best_twpa_power.data),
            best_twpa_freq=("qubit", best_twpa_freq.data),
            best_snr=("qubit", best_snr.data),
        )

        fit_data.best_twpa_power.attrs = {"long_name": "best TWPA power", "units": "dBm"}
        fit_data.best_twpa_freq.attrs = {"long_name": "best TWPA frequency", "units": "Hz"}
        fit_data.best_snr.attrs = {"long_name": "best SNR", "units": "mean/std"}

        node.results["snr_vs_twpa"] = snr.to_dataset(name="snr_value") 
        node.results["best_twpa_power"] = {q: best_twpa_power.sel(qubit=q).item() for q in best_twpa_power.qubit.values}
        node.results["best_twpa_freq"] = {q: best_twpa_freq.sel(qubit=q).item() for q in best_twpa_freq.qubit.values}
        node.results["best_snr"] = {q: best_snr.sel(qubit=q).item() for q in best_snr.qubit.values}

    return fit_data, fit_results


def _extract_relevant_fit_parameters(
    fit: xr.Dataset,
    node: QualibrationNode,
    best_twpa_power: xr.DataArray | None = None,
    best_twpa_freq: xr.DataArray | None = None,
    best_snr: xr.DataArray | None = None,
):
    """Add metadata to the dataset and fit results."""
    # Add metadata to fit results
    fit.attrs = {"long_name": "frequency", "units": "Hz"}

    # Process resonator frequency for specific qubit
    full_freq = np.array([q.resonator.RF_frequency for q in node.namespace["qubits"]])
    full_freq_da = xr.DataArray(full_freq, dims=["qubit"])
    res_freq = fit.position + full_freq_da  # broadcast qubit baseline over TWPA_amp if present
    # Align/broadcast resonator frequency across TWPA_amp while tied to each qubit
    res_freq_coord = res_freq

    for sweep_dim in ["TWPA_power", "TWPA_freq"]:
        if sweep_dim in fit.dims and sweep_dim not in res_freq_coord.dims:
            res_freq_coord = res_freq_coord.expand_dims({sweep_dim: fit.coords[sweep_dim]})

    fit = fit.assign_coords(res_freq=res_freq_coord)
    fit.res_freq.attrs = {"long_name": "resonator frequency", "units": "Hz"}
    # Get the fitted FWHM
    fwhm = np.abs(fit.width)
    fit = fit.assign_coords(fwhm=fwhm)
    fit.fwhm.attrs = {"long_name": "resonator fwhm", "units": "Hz"}
    # Assess whether the fit was successful or not
    freq_success = np.abs(res_freq_coord) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq_da
    fwhm_success = np.abs(fwhm) < node.parameters.frequency_span_in_mhz * 1e6 + full_freq_da
    success_criteria = freq_success & fwhm_success
    fit = fit.assign_coords(success=success_criteria)

    fit_results = {}
    for q in fit.qubit.values:
        fit_results[q] = FitParameters(
            frequency=_scalar_from_qubit_coord(fit.sel(qubit=q).res_freq),
            fwhm=_scalar_from_qubit_coord(fit.sel(qubit=q).fwhm),
            success=bool(_scalar_from_qubit_coord(fit.sel(qubit=q).success)),
            best_twpa_power=float(best_twpa_power.sel(qubit=q)) if best_twpa_power is not None else None,
            best_twpa_freq=float(best_twpa_freq.sel(qubit=q)) if best_twpa_freq is not None else None,
            best_snr=float(best_snr.sel(qubit=q)) if best_snr is not None else None,
        )
    return fit, fit_results


def _scalar_from_qubit_coord(da: xr.DataArray):
    """
    Extract a single scalar from a qubit-indexed DataArray.
    Automatically selects the 0-th index for any extra dimensions (like TWPA_power, TWPA_freq).
    """
    # 無論剩下什麼維度（例如 TWPA_power, TWPA_freq），
    # 迴圈都會把它們一個個切下 index 0，強迫降維到只剩單一純量
    for dim in da.dims:
        da = da.isel({dim: 0})
        
    return da.values.item()


def _compute_snr_vs_twpa_amp(ds: xr.Dataset) -> Tuple[xr.DataArray | None, xr.DataArray | None, xr.DataArray | None, xr.DataArray | None]:
    """
    Estimate SNR per TWPA_power and TWPA_freq for each qubit using IQ_abs along detuning.
    """
    # 確認兩個維度都存在
    if "TWPA_power" not in ds.dims or "TWPA_freq" not in ds.dims:
        return None, None, None, None
        
    iq = ds.IQ_abs
    signal = iq.mean(dim="detuning")
    noise = iq.std(dim="detuning")
    snr = (signal / noise).rename("snr")
    
    # 1. 找最大的 SNR 數值 (直接傳入兩個維度即可)
    best_snr = snr.max(dim=["TWPA_power", "TWPA_freq"])
    
    # 2. 找對應的座標 (將 power 和 freq 疊加成一個名為 "twpa_params" 的 1D 軸)
    snr_stacked = snr.stack(twpa_params=["TWPA_power", "TWPA_freq"])
    
    # idxmax 現在可以運作了！它會回傳一個 DataArray，裡面的值是 (power, freq) 的 Tuple
    best_twpa_coords = snr_stacked.idxmax(dim="twpa_params")
    
    # 3. 為了後續方便儲存，我們把 Tuple 拆開成兩個獨立的 DataArray
    best_twpa_power = xr.DataArray(
        [val[0] for val in best_twpa_coords.values], 
        coords=[snr.qubit], dims=["qubit"]
    )
    best_twpa_freq = xr.DataArray(
        [val[1] for val in best_twpa_coords.values], 
        coords=[snr.qubit], dims=["qubit"]
    )
    
    # 注意：回傳值從 3 個變成了 4 個！
    return snr, best_twpa_power, best_twpa_freq, best_snr

def construct_ds_raw(
    node: QualibrationNode,
    I: list,
    Q: list
) -> xr.Dataset:

    num_qubits = len(node.namespace["qubits"])
    # --- 2. 建立 Qubit 軸 (堆疊陣列) ---
    # np.array() 會自動把你 List 裡的 N 個矩陣，疊成一個新的維度 (放在第 0 軸)
    # 假設原本 I1 是 (freq, power)，現在 I_stacked 就會變成 (qubit, freq, power)
    I_stacked = np.array(I)
    Q_stacked = np.array(Q)

    # 建立 qubit 的名稱標籤 (例如: ['q0', 'q1', 'q2'])
    qubit_labels = [f"q{i}" for i in range(num_qubits)]

    # --- 3. 封裝成 xarray.Dataset ---
    # 注意：這裡的維度名稱 (dims) 必須跟你 QUA 裡面 buffer() 的順序對齊！
    # 假設你之前的順序是： buffer(df) -> buffer(twpa_freq) -> buffer(twpa_power)
    # node.namespace["sweep_axes"] = {
    #     "qubit": xr.DataArray(qubits.get_names()),
    #     "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    #     "TWPA_freq": xr.DataArray(twpa_freq_sweep, attrs={"long_name": "TWPA pump frequency", "units": "Hz"}),
    #     "TWPA_power": xr.DataArray(twpa_power_sweep, attrs={"long_name": "TWPA pump power", "units": "dBm"}),
    # }
    sweep_axes = node.namespace["sweep_axes"]
    ds = xr.Dataset(
        data_vars={
            # ⚠️ 注意：維度名稱必須跟下面 coords 的 key 完全一致！
            # 這裡補上了 n_avg 這個維度
            # "I": (["qubit", "TWPA_power", "TWPA_freq", "n_avg", "detuning"], I_stacked),
            # "Q": (["qubit", "TWPA_power", "TWPA_freq", "n_avg", "detuning"], Q_stacked),
            "I": (["qubit", "TWPA_power", "TWPA_freq", "detuning"], I_stacked),
            "Q": (["qubit", "TWPA_power", "TWPA_freq", "detuning"], Q_stacked),
        },
        coords={
            # 直接拿你存好的完美座標軸來用
            "qubit": sweep_axes["qubit"],
            "TWPA_power": sweep_axes["TWPA_power"],
            "TWPA_freq": sweep_axes["TWPA_freq"],
            "detuning": sweep_axes["detuning"],
            
            # 補上 n_avg 的座標軸 (例如 0 到 n_avg-1)
            # "n_avg": np.arange(I_stacked.shape[3]) # 假設 n_avg 在第 3 軸
        }
    )

    # --- 建立完 Dataset 後，你就可以用 xarray 輕鬆平均了 ---
    # 直接把 n_avg 維度平均掉，留下的就是 4D 資料！
    # ds = ds.mean(dim="n_avg")

    return ds
