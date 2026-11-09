import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .config import SimulationConfig
from .radar_processing import RangeDopplerResult


def convert_power_to_relative_db(power: np.ndarray, floor_db: float = -60.0) -> np.ndarray:
    peak = max(float(np.max(power)), np.finfo(float).tiny)
    return np.maximum(10 * np.log10(np.maximum(power / peak, np.finfo(float).tiny)), floor_db)


def plot_range_doppler_map(
    result: RangeDopplerResult,
    config: SimulationConfig,
) -> plt.Figure:
    display_ranges = result.ranges_m <= config.output.maximum_display_range_m
    display_velocities = (
        (result.velocities_m_s >= config.output.minimum_display_velocity_m_s)
        & (result.velocities_m_s <= config.output.maximum_display_velocity_m_s)
    )
    power_db = convert_power_to_relative_db(result.power)

    figure, axis = plt.subplots(figsize=(11, 7), layout="constrained")
    image = axis.pcolormesh(
        result.ranges_m[display_ranges],
        result.velocities_m_s[display_velocities],
        power_db[np.ix_(display_velocities, display_ranges)],
        shading="auto",
        cmap="magma",
        vmin=-60,
        vmax=0,
        rasterized=True,
    )
    for target in config.targets:
        axis.scatter(
            target.range_m,
            target.velocity_m_s,
            marker="o",
            s=95,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.5,
        )
        axis.annotate(target.name, (target.range_m, target.velocity_m_s), xytext=(6, 6),
                      textcoords="offset points", color="white", fontsize=9)
    axis.set(
        xlim=(0, min(config.output.maximum_display_range_m, result.ranges_m[-1])),
        ylim=(config.output.minimum_display_velocity_m_s,
              config.output.maximum_display_velocity_m_s),
        xlabel="Range (m)",
        ylabel="Radial velocity (m/s; positive = approaching)",
        title="OFDM radar range–Doppler map",
    )
    figure.colorbar(image, ax=axis, label="Power relative to peak (dB)")
    return figure


def plot_signal_space_diagram(
    transmitted_symbols: np.ndarray,
    received_symbols: np.ndarray,
    subcarrier_frequencies_hz: np.ndarray,
    symbol_index: int | None = None,
) -> plt.Figure:
    """Overlay all transmitted and received symbols, coloring received points by frequency."""
    if received_symbols.shape != transmitted_symbols.shape:
        raise ValueError("Transmitted and received symbol grids must have matching shapes")
    if transmitted_symbols.shape[1] != subcarrier_frequencies_hz.size:
        raise ValueError("There must be one frequency value per subcarrier")
    if symbol_index is not None and not 0 <= symbol_index < transmitted_symbols.shape[0]:
        raise ValueError("symbol_index is outside the available OFDM symbols")

    subcarrier_indices = np.arange(subcarrier_frequencies_hz.size)
    color_scale = Normalize(
        vmin=float(subcarrier_indices.min()),
        vmax=float(subcarrier_indices.max()),
    )
    figure, axis = plt.subplots(figsize=(7, 7), layout="constrained")
    if symbol_index is None:
        transmitted = transmitted_symbols.reshape(-1)
        received = received_symbols.reshape(-1)
        received_subcarrier_indices = np.tile(
            subcarrier_indices,
            received_symbols.shape[0],
        )
        title = "OFDM equalized signal space · all symbols"
    else:
        transmitted = transmitted_symbols[symbol_index]
        received = received_symbols[symbol_index]
        received_subcarrier_indices = subcarrier_indices
        title = f"OFDM equalized signal space · symbol {symbol_index}"
    axis.scatter(
        transmitted.real,
        transmitted.imag,
        marker="x",
        s=55,
        color="red",
        alpha=1.0,
        linewidths=1.5,
        label="Transmitted",
        rasterized=True,
        zorder=3,
    )
    received_scatter = axis.scatter(
        received.real,
        received.imag,
        c=received_subcarrier_indices,
        cmap="viridis",
        norm=color_scale,
        s=22,
        alpha=0.85,
        linewidths=0,
        label="Downlink equalized",
        rasterized=True,
    )
    axis.set(
        xlabel="In-phase (I)",
        ylabel="Quadrature (Q)",
        title=title,
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(True, alpha=0.25)
    axis.legend()
    colorbar = figure.colorbar(received_scatter, ax=axis, pad=0.02)
    colorbar.set_label("Subcarrier index")
    return figure
