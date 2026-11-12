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


def plot_signal_space_comparison(
    transmitted_symbols: np.ndarray,
    received_before_equalisation: np.ndarray,
    received_after_equalisation: np.ndarray,
    channel_response: np.ndarray,
    symbol_index: int = 0,
) -> plt.Figure:
    """Compare pre- and post-equalisation signal spaces for one OFDM symbol."""
    if (
        received_before_equalisation.shape != transmitted_symbols.shape
        or received_after_equalisation.shape != transmitted_symbols.shape
        or channel_response.shape != transmitted_symbols.shape
    ):
        raise ValueError("Symbol grids and channel response must have matching shapes")
    if transmitted_symbols.ndim != 2 or transmitted_symbols.shape[1] == 0:
        raise ValueError("Symbol grids must be nonempty OFDM-symbol by subcarrier arrays")
    if not 0 <= symbol_index < transmitted_symbols.shape[0]:
        raise ValueError("symbol_index is outside the available OFDM symbols")

    channel_magnitude = np.abs(channel_response[symbol_index])
    color_scale = Normalize(
        vmin=float(channel_magnitude.min()),
        vmax=float(channel_magnitude.max()),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    stages = (
        (received_before_equalisation, "Before equalisation"),
        (received_after_equalisation, "After equalisation"),
    )
    received_scatter = None
    for axis, (received_grid, stage) in zip(axes, stages):
        transmitted = transmitted_symbols[symbol_index]
        received = received_grid[symbol_index]
        axis.scatter(
            transmitted.real,
            transmitted.imag,
            marker="x",
            s=55,
            color="red",
            linewidths=1.5,
            label="Transmitted",
            zorder=3,
        )
        received_scatter = axis.scatter(
            received.real,
            received.imag,
            c=channel_magnitude,
            cmap="viridis",
            norm=color_scale,
            s=28,
            alpha=0.85,
            linewidths=0,
            label="Received",
        )
        axis.set(
            xlabel="In-phase (I)",
            ylabel="Quadrature (Q)",
            title=f"{stage} · OFDM symbol {symbol_index}",
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
        axis.legend()

    colorbar = figure.colorbar(received_scatter, ax=axes, pad=0.02)
    colorbar.set_label("Channel magnitude |H[k]|")
    return figure
