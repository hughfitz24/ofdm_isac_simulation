from dataclasses import dataclass

import numpy as np
from scipy import fft, signal

from .config import DataConfig, ProcessingConfig, RadarConfig


@dataclass(frozen=True)
class RangeDopplerResult:
    ranges_m: np.ndarray
    velocities_m_s: np.ndarray
    power: np.ndarray


def estimate_range_doppler(
    channel_estimates: np.ndarray,
    radar: RadarConfig,
    data: DataConfig,
    processing: ProcessingConfig,
) -> RangeDopplerResult:
    subcarrier_window = signal.windows.hann(data.subcarrier_count, sym=False)
    slow_time_window = signal.windows.hann(data.ofdm_symbol_count, sym=False)
    windowed_channel = channel_estimates * slow_time_window[:, None]
    windowed_channel = windowed_channel * subcarrier_window[None, :]

    n_subcarriers = data.subcarrier_count
    n_fft = processing.range_fft_size
    if n_fft < n_subcarriers:
        raise ValueError("Range FFT must be at least as large as subcarrier count")

    padding = n_fft - n_subcarriers
    pad_left = padding // 2
    pad_right = padding - pad_left
    padded_channel = np.pad(
        windowed_channel,
        ((0, 0), (pad_left, pad_right)),
        mode="constant",
    )
    range_response = fft.ifft(fft.ifftshift(padded_channel, axes=1), axis=1)
    range_doppler_response = fft.fftshift(
        fft.fft(range_response, n=processing.doppler_fft_size, axis=0), axes=0
    )
    power = np.abs(range_doppler_response) ** 2

    round_trip_delays_s = fft.fftfreq(
        processing.range_fft_size, d=data.subcarrier_spacing_hz
    )
    ranges_m = round_trip_delays_s * 299_792_458.0 / 2
    doppler_frequencies_hz = fft.fftshift(
        fft.fftfreq(processing.doppler_fft_size, d=data.symbol_duration_s)
    )
    wavelength_m = 299_792_458.0 / radar.carrier_frequency_hz
    velocities_m_s = doppler_frequencies_hz * wavelength_m / 2

    positive_ranges = ranges_m >= 0
    return RangeDopplerResult(
        ranges_m[positive_ranges],
        velocities_m_s,
        power[:, positive_ranges],
    )
