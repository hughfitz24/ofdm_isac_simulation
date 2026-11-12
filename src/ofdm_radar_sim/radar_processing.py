# ===========================================
# radar_processing.py
# Handles the estimation of both range and doppler shift from a set of transmit signals.
# Uses information passed from transmitter.py to perform delay-doppler estimation.
# ===========================================

from dataclasses import dataclass

import numpy as np
from scipy import fft, signal

from .config import DataConfig, ProcessingConfig, RadarConfig, SPEED_OF_LIGHT_M_S


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
    """
    Estimates the range and velocity of the targets in a wireless channel using the periodogram.

    :param channel_estimates: The estimation of the channel. Done by dividing the reflected signals by the transmitted, to isolate the channel effect.
    :param radar: The radar configuration, i.e. carrier freq
    :param data: The OFDM configuration, i.e. symbol count, CP length, etc
    :param processing: Configuration of the size of the FFT bins for estimating range and Doppler.
    """

    # Applies a Hann window to taper the data for each individual subcarrier at both sides
    subcarrier_window = signal.windows.hann(data.subcarrier_count, sym=False)

    # Applies a Hann window to taper the data across the seperate symbols, to smooth the Doppler estimation across different transmissions
    doppler_window = signal.windows.hann(data.ofdm_symbol_count, sym=False)

    # Apply the window across the vertical axis i.e. the different symbols
    windowed_channel = channel_estimates * doppler_window[:, None]

    # Apply the window to the horizontal axis i.e. the different subcarriers
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

    # Perform an IFFT along the subcarrier (frequency) axis to transform
    # the channel estimates into the delay domain.
    # Each row represents the delay response for one OFDM symbol as a complex value. 
    # The magnitude of this complex value is how strong the channel response is, the phase is the phase of the channel response
    # This phase will evolve over time, which is the key measurement in this processing
    range_response = fft.ifft(fft.ifftshift(padded_channel, axes=1), axis=1)

    # Performs FFT on the slow time axis i.e. the vertical
    # Estimates the Doppler frequencies, based on how phase evolves over slow time
    range_doppler_response = fft.fftshift(
        fft.fft(range_response, n=processing.doppler_fft_size, axis=0), axes=0
    )

    # Finds the power associated with each range-doppler bin
    power = np.abs(range_doppler_response) ** 2

    # Finds the delay value represented by each fft bin, in seconds
    round_trip_delays_s = fft.fftfreq(
        processing.range_fft_size, d=data.subcarrier_spacing_hz
    )

    # Converts that delay value into a distance, assuming waves propagate at c
    # 1/2 factor accounts for the forward and back paths
    ranges_m = round_trip_delays_s * SPEED_OF_LIGHT_M_S / 2

    # Finds the doppler frequency measured in each FFT bin
    doppler_frequencies_hz = fft.fftshift(
        fft.fftfreq(processing.doppler_fft_size, d=data.symbol_duration_s)
    )
    wavelength_m = 299_792_458.0 / radar.carrier_frequency_hz

    # Converts doppler frequencies to radial velocity estimates 
    velocities_m_s = doppler_frequencies_hz * wavelength_m / 2

    positive_ranges = ranges_m >= 0
    return RangeDopplerResult(
        ranges_m[positive_ranges],
        velocities_m_s,
        power[:, positive_ranges],
    )
