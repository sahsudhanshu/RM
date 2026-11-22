"""Physical and mission parameters (values from the base paper, arXiv:2504.21111)."""

from __future__ import annotations

from dataclasses import dataclass


def uav_power_w(speed_ms: float) -> float:
    """UAV electrical power draw at forward speed `speed_ms` (m/s), base paper Sec. V-A."""
    v = speed_ms
    return 0.0461 * v**3 - 0.5834 * v**2 - 1.8761 * v + 229.6


@dataclass(frozen=True)
class Params:
    width_km: float = 20.0
    height_km: float = 20.0
    aerial_radius_km: float = 7.0
    uav_speed_ms: float = 10.0
    ugv_speed_ms: float = 4.5
    fuel_kj: float = 287.7          # F_a, per-sortie battery
    recharge_min: float = 15.0      # T_R -- ASSUMPTION, not stated in the paper text
    penalty: float = 800.0          # failure penalty (minutes)
    wait_mode: str = "hover"        # "hover": waiting UAV burns P(0); "landed": waiting is free
    wait_tol_min: float = 30.0      # road-aware mask: reject rendezvous with larger projected wait
    max_rounds: int = 80
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.wait_mode not in {"hover", "landed"}:
            raise ValueError("wait_mode must be 'hover' or 'landed'")
        if min(self.uav_speed_ms, self.ugv_speed_ms, self.fuel_kj, self.recharge_min) <= 0:
            raise ValueError("speeds, battery and recharge time must be positive")

    @property
    def uav_kmmin(self) -> float:
        return self.uav_speed_ms * 60.0 / 1000.0

    @property
    def ugv_kmmin(self) -> float:
        return self.ugv_speed_ms * 60.0 / 1000.0

    @property
    def fly_power_w(self) -> float:
        return uav_power_w(self.uav_speed_ms)

    @property
    def hover_power_w(self) -> float:
        return uav_power_w(0.0) if self.wait_mode == "hover" else 0.0

    @property
    def fly_kj_per_km(self) -> float:
        """Energy (kJ) per km of UAV flight: P[W] * (min/km) * 60 s/min / 1000."""
        return self.fly_power_w * (1.0 / self.uav_kmmin) * 60.0 / 1000.0

    @property
    def hover_kj_per_min(self) -> float:
        return self.hover_power_w * 60.0 / 1000.0

    @property
    def max_range_km(self) -> float:
        return self.fuel_kj / self.fly_kj_per_km
