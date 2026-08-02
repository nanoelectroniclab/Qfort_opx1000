from typing import List

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure


STATE_LABELS = [format(i, "05b") for i in range(32)]
GHZ_STATES = [0, 31]  # |00000⟩ and |11111⟩


def plot_raw_data_with_fit(ds: xr.Dataset, quintet_names: List[str], ds_fit: xr.Dataset) -> Figure:
    num_quintets = len(quintet_names)
    fig, axes = plt.subplots(num_quintets, 1, figsize=(18, 3 * num_quintets), squeeze=False)

    for row, qname in enumerate(quintet_names):
        ax = axes[row, 0]
        corrected = ds_fit["corrected_probs"].sel(quintet=qname).values  # (32,)
        colors = ["steelblue" if i not in GHZ_STATES else "tomato" for i in range(32)]

        ax.bar(STATE_LABELS, corrected, color=colors, edgecolor="navy", linewidth=0.5)
        for i, v in enumerate(corrected):
            if v > 0.02:
                ax.text(i, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        ghz_fidelity = float(corrected[0] + corrected[31])
        ax.set_ylim(0, max(corrected.max() * 1.15, 0.55))
        ax.set_ylabel("Probability")
        ax.set_xlabel("5-qubit state")
        ax.set_title(f"{qname}  |  GHZ fidelity = {ghz_fidelity:.3f}")
        ax.tick_params(axis="x", labelsize=7, rotation=45)

    fig.suptitle("GHZ State Z-basis Measurement (5Q)", fontsize=12)
    fig.tight_layout()
    return fig
