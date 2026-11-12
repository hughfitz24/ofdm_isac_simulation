from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class Target:
    name: str
    range_m: float
    velocity_m_s: float
    rcs_m2: float = 1.0


@dataclass(frozen=True)
class RadarConfig:
    carrier_frequency_hz: float
    transmit_power_w: float
    random_seed: int


@dataclass(frozen=True)
class DataConfig:
    modulation: str
    subcarrier_spacing_hz: float
    subcarrier_count: int
    ofdm_symbol_count: int
    cyclic_prefix_samples: int

    @property
    def sample_rate_hz(self) -> float:
        return self.subcarrier_spacing_hz * self.subcarrier_count

    @property
    def symbol_duration_s(self) -> float:
        return (self.subcarrier_count + self.cyclic_prefix_samples) / self.sample_rate_hz

    @property
    def bandwidth_hz(self) -> float:
        return self.subcarrier_spacing_hz * self.subcarrier_count

    @property
    def range_resolution_m(self) -> float:
        return SPEED_OF_LIGHT_M_S / (2 * self.bandwidth_hz)

    @property
    def maximum_cyclic_prefix_range_m(self) -> float:
        return SPEED_OF_LIGHT_M_S * self.cyclic_prefix_samples / (2 * self.sample_rate_hz)


@dataclass(frozen=True)
class ProcessingConfig:
    range_fft_size: int
    doppler_fft_size: int


@dataclass(frozen=True)
class OutputConfig:
    show_plots: bool
    save_plots: bool
    directory: Path
    maximum_display_range_m: float
    minimum_display_velocity_m_s: float
    maximum_display_velocity_m_s: float


@dataclass(frozen=True)
class WirelessChannelConfig:
    tx_antenna_gain_linear: float
    rx_antenna_gain_linear: float
    snr_db: float
    noise_enabled: bool


@dataclass(frozen=True)
class DownlinkChannelConfig:
    path_delays_s: tuple[float, ...]
    average_path_gains_db: tuple[float, ...]
    rician_k_factor_db: float
    maximum_doppler_hz: float
    direct_path_doppler_hz: float
    snr_db: float
    noise_enabled: bool


@dataclass(frozen=True)
class SimulationConfig:
    radar: RadarConfig
    data: DataConfig
    channel: WirelessChannelConfig
    downlink: DownlinkChannelConfig
    processing: ProcessingConfig
    output: OutputConfig
    targets: tuple[Target, ...]

    @property
    def maximum_unambiguous_velocity_m_s(self) -> float:
        wavelength_m = SPEED_OF_LIGHT_M_S / self.radar.carrier_frequency_hz
        return wavelength_m / (4 * self.data.symbol_duration_s)


def _positive_number(table: dict, name: str) -> float:
    value = float(table[name])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def load_simulation_config(config_path: str | Path) -> SimulationConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("rb") as config_file:
        values = tomllib.load(config_file)

    data_values = values["ofdm"]
    modulation = str(data_values["modulation"])
    if modulation not in {"BPSK", "QPSK", "16QAM", "32QAM"}:
        raise ValueError("modulation must be one of BPSK, QPSK, 16QAM, or 32QAM")
    data = DataConfig(
        modulation=modulation,
        subcarrier_spacing_hz=_positive_number(data_values, "subcarrier_spacing_hz"),
        subcarrier_count=int(data_values["num_subcarriers"]),
        ofdm_symbol_count=int(data_values["num_symbols"]),
        cyclic_prefix_samples=int(data_values["cyclic_prefix_samples"]),
    )
    if data.subcarrier_count < 2 or data.ofdm_symbol_count < 1:
        raise ValueError("subcarrier_count must be at least 2 and ofdm_symbol_count at least 1")
    if not 0 <= data.cyclic_prefix_samples < data.subcarrier_count:
        raise ValueError("cyclic_prefix_samples must be in [0, subcarrier_count)")

    radar_values = values["radar"]
    random_seed = int(values["random_seed"])
    radar = RadarConfig(
        carrier_frequency_hz=_positive_number(radar_values, "carrier_frequency_hz"),
        transmit_power_w=_positive_number(radar_values, "transmit_power_w"),
        random_seed=random_seed,
    )
    if radar.random_seed < 0:
        raise ValueError("random_seed must be nonnegative")
    channel_values = values.get("radar_channel", {})
    radar_noise_enabled = "reference_snr_db" in channel_values
    channel = WirelessChannelConfig(
        tx_antenna_gain_linear=_positive_number(radar_values, "tx_antenna_gain_linear"),
        rx_antenna_gain_linear=_positive_number(radar_values, "rx_antenna_gain_linear"),
        snr_db=float(channel_values.get("reference_snr_db", 0.0)),
        noise_enabled=radar_noise_enabled,
    )
    if channel.noise_enabled and not np.isfinite(channel.snr_db):
        raise ValueError("reference_snr_db must be finite")

    downlink_values = values["downlink_channel"]
    downlink_paths = downlink_values.get("paths", [])
    downlink_noise_enabled = "snr_db" in downlink_values
    downlink = DownlinkChannelConfig(
        path_delays_s=tuple(float(path_value["delay_s"]) for path_value in downlink_paths),
        average_path_gains_db=tuple(float(path_value["gain_db"]) for path_value in downlink_paths),
        rician_k_factor_db=float(downlink_values["rician_k_factor_db"]),
        maximum_doppler_hz=float(downlink_values["max_doppler_hz"]),
        direct_path_doppler_hz=float(downlink_values["direct_path_doppler_hz"]),
        snr_db=float(downlink_values.get("snr_db", 0.0)),
        noise_enabled=downlink_noise_enabled,
    )
    if not downlink.path_delays_s:
        raise ValueError("downlink path_delays_s must contain at least the direct path")
    if len(downlink.path_delays_s) != len(downlink.average_path_gains_db):
        raise ValueError("downlink path delays and average path gains must have the same length")
    if downlink.path_delays_s[0] != 0:
        raise ValueError("the first downlink path must be the direct path with zero excess delay")
    if any(not np.isfinite(value) or value < 0 for value in downlink.path_delays_s):
        raise ValueError("downlink path delays must be finite and nonnegative")
    if any(not np.isfinite(value) for value in downlink.average_path_gains_db):
        raise ValueError("downlink average path gains must be finite")
    if not np.isfinite(downlink.rician_k_factor_db):
        raise ValueError("rician_k_factor_db must be finite")
    if not np.isfinite(downlink.maximum_doppler_hz) or downlink.maximum_doppler_hz < 0:
        raise ValueError("maximum_doppler_hz must be finite and nonnegative")
    if not np.isfinite(downlink.direct_path_doppler_hz):
        raise ValueError("direct_path_doppler_hz must be finite")
    if downlink.noise_enabled and not np.isfinite(downlink.snr_db):
        raise ValueError("downlink snr_db must be finite")
    cyclic_prefix_duration_s = data.cyclic_prefix_samples / data.sample_rate_hz
    if max(downlink.path_delays_s) > cyclic_prefix_duration_s:
        raise ValueError("CP is too short, the delays modelled for multipath will cause ISI at this CP length")
    if downlink.maximum_doppler_hz > data.subcarrier_spacing_hz / 10:
        raise ValueError("maximum_doppler_hz must be at most one tenth of subcarrier spacing")

    processing_values = values["processing"]
    processing = ProcessingConfig(
        range_fft_size=int(processing_values["range_fft_size"]),
        doppler_fft_size=int(processing_values["doppler_fft_size"]),
    )
    if processing.range_fft_size < data.subcarrier_count:
        raise ValueError("range_fft_size must be at least subcarrier_count")
    if processing.doppler_fft_size < data.ofdm_symbol_count:
        raise ValueError("doppler_fft_size must be at least ofdm_symbol_count")
    output_values = values["output"]
    display_velocity_range = tuple(
        float(value) for value in output_values["display_velocity_range_m_s"]
    )
    if len(display_velocity_range) != 2:
        raise ValueError("display_velocity_range_m_s must contain a minimum and maximum")
    if not all(np.isfinite(value) for value in display_velocity_range):
        raise ValueError("display_velocity_range_m_s values must be finite")
    if display_velocity_range[0] >= display_velocity_range[1]:
        raise ValueError("display velocity minimum must be less than its maximum")
    output = OutputConfig(
        show_plots=bool(output_values["show_plots"]),
        save_plots=bool(output_values["save_plots"]),
        directory=(path.parent / output_values["figure_directory"]).resolve(),
        maximum_display_range_m=_positive_number(output_values, "max_display_range_m"),
        minimum_display_velocity_m_s=display_velocity_range[0],
        maximum_display_velocity_m_s=display_velocity_range[1],
    )
    targets = tuple(
        Target(
            name=str(target.get("name", f"Target {index + 1}")),
            range_m=float(target["range_m"]),
            velocity_m_s=float(target["velocity_m_s"]),
            rcs_m2=float(target.get("rcs_m2", 1.0)),
        )
        for index, target in enumerate(values.get("targets", []))
    )
    for target in targets:
        if not all(np.isfinite(value) for value in (
            target.range_m,
            target.velocity_m_s,
            target.rcs_m2,
        )):
            raise ValueError(f"{target.name} values must be finite")
        if target.range_m <= 0:
            raise ValueError(f"{target.name} range_m must be positive")
        if target.rcs_m2 < 0:
            raise ValueError(f"{target.name} rcs_m2 must be nonnegative")
        if target.range_m > data.maximum_cyclic_prefix_range_m:
            raise ValueError(
                f"{target.name} exceeds the cyclic-prefix range of "
                f"{data.maximum_cyclic_prefix_range_m:.2f} m"
            )
        maximum_unambiguous_velocity_m_s = (
            SPEED_OF_LIGHT_M_S / radar.carrier_frequency_hz
            / (4 * data.symbol_duration_s)
        )
        if abs(target.velocity_m_s) > maximum_unambiguous_velocity_m_s:
            raise ValueError(f"{target.name} exceeds the unambiguous Doppler velocity")

    if not targets:
        raise ValueError("Add at least one [[targets]] entry to the TOML configuration")
    return SimulationConfig(radar, data, channel, downlink, processing, output, targets)
