"""Create the OFDM data grid that the transmitter sends over both links."""
from dataclasses import dataclass

import numpy as np

from .config import DataConfig, RadarConfig


@dataclass(frozen=True)
class OFDMTransmission:
    transmitted_symbols: np.ndarray
    subcarrier_frequencies_hz: np.ndarray
    symbol_times_s: np.ndarray


def create_ofdm_transmission(
    data: DataConfig,
    radar: RadarConfig,
    constellation: np.ndarray,
) -> OFDMTransmission:
    """Create the modulated OFDM resource grid and its coordinates.

    The supplied constellation is expected to be scaled to the desired transmit
    power before it is passed here.
    """
    if constellation.ndim != 1 or constellation.size == 0:
        raise ValueError("constellation must be a nonempty one-dimensional array")
    rng = np.random.default_rng(radar.random_seed)
    symbol_indices = rng.integers(
        0,
        constellation.size,
        size=(data.ofdm_symbol_count, data.subcarrier_count),
    )
    transmitted_symbols = constellation[symbol_indices].copy()
    subcarrier_frequencies_hz = (
        np.arange(data.subcarrier_count) - data.subcarrier_count // 2
    ) * data.subcarrier_spacing_hz
    symbol_times_s = np.arange(data.ofdm_symbol_count) * data.symbol_duration_s
    return OFDMTransmission(
        transmitted_symbols,
        subcarrier_frequencies_hz,
        symbol_times_s,
    )
