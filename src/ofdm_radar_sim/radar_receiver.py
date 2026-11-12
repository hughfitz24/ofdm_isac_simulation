"""Receive target echoes and form a radar range–Doppler estimate."""
from dataclasses import dataclass

from .config import (
    SPEED_OF_LIGHT_M_S,
    DataConfig,
    ProcessingConfig,
    RadarConfig,
    Target,
    WirelessChannelConfig,
)
from .radar_processing import RangeDopplerResult, estimate_range_doppler
from .transmitter import OFDMTransmission
from .wireless_channel import create_reflected_signals, create_reflections_from_target

import numpy as np


@dataclass(frozen=True)
class RadarReception:
    received_symbols: np.ndarray
    channel_estimates: np.ndarray
    range_doppler: RangeDopplerResult


def receive_target_echoes(
    transmission: OFDMTransmission,
    radar: RadarConfig,
    data: DataConfig,
    channel: WirelessChannelConfig,
    targets: tuple[Target, ...],
    processing: ProcessingConfig,
) -> RadarReception:
    """Propagate target reflections back to the radar and estimate their ranges and speeds."""
    reflections_by_target = tuple(
        create_reflections_from_target(
            transmission.transmitted_symbols,
            transmission.subcarrier_frequencies_hz,
            transmission.symbol_times_s,
            target,
            radar.carrier_frequency_hz,
        )
        for target in targets
    )
    received_symbols = create_reflected_signals(
        reflections_by_target,
        targets,
        channel,
        SPEED_OF_LIGHT_M_S / radar.carrier_frequency_hz,
        radar.random_seed + 1,
    )
    # Dividing the received symbol by the transmitted symbol gives the affect of the channel + any noise contribution
    channel_estimates = received_symbols / transmission.transmitted_symbols
    range_doppler = estimate_range_doppler(
        channel_estimates,
        radar,
        data,
        processing,
    )
    return RadarReception(received_symbols, channel_estimates, range_doppler)
