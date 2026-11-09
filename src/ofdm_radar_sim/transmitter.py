"""Create the OFDM data grid that the transmitter sends over both links."""
from dataclasses import dataclass

import numpy as np

from .config import DataConfig, RadarConfig


@dataclass(frozen=True)
class OFDMTransmission:
    transmitted_symbols: np.ndarray
    subcarrier_frequencies_hz: np.ndarray
    symbol_times_s: np.ndarray


def create_ofdm_transmission(data: DataConfig, radar: RadarConfig) -> OFDMTransmission:
    """Create the QPSK resource grid and its time and frequency coordinates."""
    rng = np.random.default_rng(radar.random_seed)
    qpsk_indices = rng.integers(
        0,
        4,
        size=(data.ofdm_symbol_count, data.subcarrier_count),
    )
    qpsk_constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
    transmitted_symbols = qpsk_constellation[qpsk_indices]
    transmitted_symbols *= np.sqrt(radar.transmit_power_w / data.subcarrier_count)
    subcarrier_frequencies_hz = (
        np.arange(data.subcarrier_count) - data.subcarrier_count // 2
    ) * data.subcarrier_spacing_hz
    symbol_times_s = np.arange(data.ofdm_symbol_count) * data.symbol_duration_s
    return OFDMTransmission(
        transmitted_symbols,
        subcarrier_frequencies_hz,
        symbol_times_s,
    )
