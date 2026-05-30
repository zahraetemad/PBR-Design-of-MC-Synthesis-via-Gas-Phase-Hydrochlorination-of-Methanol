"""Packed-bed reactor model for methyl chloride synthesis.

The model solves a coupled, non-isothermal PFR with pressure drop and coolant
heat removal for:

    R1: CH3OH + HCl  <-> CH3Cl + H2O
    R2: 2 CH3OH     <-> DME + H2O

Run:
    python reactor_model.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


# =============================================================================
# Thermodynamic data
# =============================================================================

R_GAS = 8.314  # J/mol/K
T_REF = 298.15  # K
SMALL = 1.0e-14

SHOMATE = {
    "HCl": np.array(
        [32.12392, -13.45805, 19.86852, -6.853936, -0.049672, -101.6206, 228.6866, -92.31201]
    ),
    "MeCl": np.array(
        [3.524690, 136.9277, -82.14196, 20.22797, 0.278032, -89.19995, 202.8391, -83.68000]
    ),
    "H2O": np.array(
        [30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264]
    ),
    "MeOH": np.array(
        [1.02174, 145.69696, -71.23345, 13.60109, 0.49387, -4.52149, 203.52073, 0.0]
    ),
    "DME": np.array(
        [5.42132, 216.58801, -105.26906, 19.60664, 0.39139, -9.03898, 215.69206, 0.0]
    ),
}

FORMATION_ENTHALPY_298 = {
    "MeOH": -205.000,  # kJ/mol
    "HCl": -92.307,
    "MeCl": -83.680,
    "H2O": -241.826,
    "DME": -184.100,
}

MOLECULAR_WEIGHT = {
    "MeOH": 32.04e-3,  # kg/mol
    "HCl": 36.4606e-3,
    "MeCl": 50.4875e-3,
    "H2O": 18.0153e-3,
    "DME": 46.069e-3,
}

SPECIES = ("MeOH", "HCl", "MeCl", "H2O", "DME")


@dataclass(frozen=True)
class ReactorConfig:
    """Operating conditions, geometry, and transport assumptions."""

    reactor_volume: float = 10.0  # m3 total packed-bed volume
    tube_length: float = 8.0  # m
    void_fraction: float = 0.52
    bulk_density: float = 570.0  # kg catalyst / m3 packed bed
    particle_diameter: float = 0.0035  # m
    tube_to_particle_diameter: float = 8.0

    inlet_temperature: float = 280.0 + 273.15  # K
    coolant_inlet_temperature: float = 260.0 + 273.15  # K
    pressure_bar: float = 5.0  # bar

    meoh_feed_kmol_h: float = 152.813
    hcl_to_meoh_ratio: float = 1.7

    mixture_viscosity: float = 2.26851e-5  # Pa s
    overall_heat_transfer: float = 600.0  # W/m2/K
    coolant_mass_flow: float = 13.0  # kg/s, total plant coolant
    coolant_heat_capacity: float = 1507.0  # J/kg/K

    points: int = 1000
    rtol: float = 1.0e-6
    atol: float = 1.0e-8


@dataclass(frozen=True)
class Geometry:
    tube_diameter: float
    tube_area: float
    tube_volume: float
    tube_count: int
    catalyst_mass_total: float
    catalyst_mass_per_tube: float
    mass_flux: float
    pressure_drop_alpha: float
    total_cross_section: float
    heat_transfer_area_per_volume: float


@dataclass(frozen=True)
class LocalRates:
    r1: float  # mol/m3/s
    r2: float  # mol/m3/s
    dH1: float  # kJ/mol
    dH2: float  # kJ/mol
    heat_generation: float  # J/m3/s


def shomate_h_sensible(coefficients: np.ndarray, temperature: float) -> float:
    """Return H(T) - H(298.15 K), kJ/mol."""

    t = temperature / 1000.0
    return (
        coefficients[0] * t
        + coefficients[1] * t**2 / 2.0
        + coefficients[2] * t**3 / 3.0
        + coefficients[3] * t**4 / 4.0
        - coefficients[4] / t
        + coefficients[5]
        - coefficients[7]
    )


def shomate_entropy(coefficients: np.ndarray, temperature: float) -> float:
    """Return absolute Shomate entropy, J/mol/K."""

    t = temperature / 1000.0
    return (
        coefficients[0] * np.log(t)
        + coefficients[1] * t
        + coefficients[2] * t**2 / 2.0
        + coefficients[3] * t**3 / 3.0
        - coefficients[4] / (2.0 * t**2)
        + coefficients[6]
    )


def shomate_cp(coefficients: np.ndarray, temperature: float) -> float:
    """Return gas heat capacity, J/mol/K."""

    t = temperature / 1000.0
    return coefficients[0] + coefficients[1] * t + coefficients[2] * t**2 + coefficients[3] * t**3 + coefficients[4] / t**2


def reaction_delta_h(temperature: float, products: dict[str, float], reactants: dict[str, float]) -> float:
    """Reaction enthalpy at temperature, kJ/mol of reaction."""

    def species_enthalpy(species: str) -> float:
        return FORMATION_ENTHALPY_298[species] + shomate_h_sensible(SHOMATE[species], temperature)

    return sum(nu * species_enthalpy(sp) for sp, nu in products.items()) - sum(
        nu * species_enthalpy(sp) for sp, nu in reactants.items()
    )


def reaction_delta_s(temperature: float, products: dict[str, float], reactants: dict[str, float]) -> float:
    """Reaction entropy at temperature, J/mol/K."""

    return sum(nu * shomate_entropy(SHOMATE[sp], temperature) for sp, nu in products.items()) - sum(
        nu * shomate_entropy(SHOMATE[sp], temperature) for sp, nu in reactants.items()
    )


def reaction_thermodynamics(temperature: float) -> dict[str, float]:
    """Return dH, dS, dG, and equilibrium constants for both reactions."""

    r1_products = {"MeCl": 1.0, "H2O": 1.0}
    r1_reactants = {"MeOH": 1.0, "HCl": 1.0}
    r2_products = {"DME": 1.0, "H2O": 1.0}
    r2_reactants = {"MeOH": 2.0}

    dH1 = reaction_delta_h(temperature, r1_products, r1_reactants)
    dH2 = reaction_delta_h(temperature, r2_products, r2_reactants)
    dS1 = reaction_delta_s(temperature, r1_products, r1_reactants)
    dS2 = reaction_delta_s(temperature, r2_products, r2_reactants)
    dG1 = dH1 - temperature * dS1 / 1000.0
    dG2 = dH2 - temperature * dS2 / 1000.0

    return {
        "dH1": dH1,
        "dH2": dH2,
        "dS1": dS1,
        "dS2": dS2,
        "dG1": dG1,
        "dG2": dG2,
        "Keq1": np.exp(-dG1 * 1000.0 / (R_GAS * temperature)),
        "Keq2": np.exp(-dG2 * 1000.0 / (R_GAS * temperature)),
    }


def feed_flows(config: ReactorConfig) -> dict[str, float]:
    """Return total plant inlet molar flows, mol/s."""

    meoh = config.meoh_feed_kmol_h * 1000.0 / 3600.0
    hcl = meoh * config.hcl_to_meoh_ratio
    return {"MeOH": meoh, "HCl": hcl, "MeCl": 0.0, "H2O": 0.0, "DME": 0.0}


def mixture_molecular_weight(flows: np.ndarray) -> float:
    """Mixture molecular weight, kg/mol."""

    total_flow = float(np.sum(flows))
    if total_flow <= SMALL:
        inlet = feed_flows(ReactorConfig())
        inlet_total = inlet["MeOH"] + inlet["HCl"]
        return (inlet["MeOH"] * MOLECULAR_WEIGHT["MeOH"] + inlet["HCl"] * MOLECULAR_WEIGHT["HCl"]) / inlet_total

    return sum(flow * MOLECULAR_WEIGHT[species] for flow, species in zip(flows, SPECIES)) / total_flow


def build_geometry(config: ReactorConfig) -> Geometry:
    """Calculate tube bundle and packed-bed hydrodynamic quantities."""

    tube_diameter = config.tube_to_particle_diameter * config.particle_diameter
    tube_area = np.pi * tube_diameter**2 / 4.0
    tube_volume = tube_area * config.tube_length
    tube_count = int(np.ceil(config.reactor_volume / tube_volume))
    catalyst_mass_total = config.bulk_density * config.reactor_volume
    catalyst_mass_per_tube = config.bulk_density * tube_volume

    inlet = feed_flows(config)
    total_mass_flow = inlet["MeOH"] * MOLECULAR_WEIGHT["MeOH"] + inlet["HCl"] * MOLECULAR_WEIGHT["HCl"]
    mass_flux = (total_mass_flow / tube_count) / tube_area

    eps = config.void_fraction
    pressure_drop_alpha = (mass_flux / config.particle_diameter) * ((1.0 - eps) / eps**3) * (
        150.0 * (1.0 - eps) * config.mixture_viscosity / config.particle_diameter + 1.75 * mass_flux
    )

    return Geometry(
        tube_diameter=tube_diameter,
        tube_area=tube_area,
        tube_volume=tube_volume,
        tube_count=tube_count,
        catalyst_mass_total=catalyst_mass_total,
        catalyst_mass_per_tube=catalyst_mass_per_tube,
        mass_flux=mass_flux,
        pressure_drop_alpha=pressure_drop_alpha,
        total_cross_section=tube_area * tube_count,
        heat_transfer_area_per_volume=4.0 / tube_diameter,
    )


def partial_pressures_bar(flows: np.ndarray, pressure_pa: float) -> dict[str, float]:
    """Return gas partial pressures in bar."""

    clipped_flows = np.maximum(flows, 0.0)
    total_flow = float(np.sum(clipped_flows))
    if total_flow <= SMALL:
        return {species: 0.0 for species in SPECIES}

    pressure_bar = max(pressure_pa, 1.0) / 1.0e5
    return {species: flow / total_flow * pressure_bar for species, flow in zip(SPECIES, clipped_flows)}


def local_rates(flows: np.ndarray, pressure_pa: float, temperature: float, config: ReactorConfig) -> LocalRates:
    """Calculate reaction rates and heat generation at the local state."""

    temperature = max(temperature, 10.0)
    partials = partial_pressures_bar(flows, pressure_pa)
    thermo = reaction_thermodynamics(temperature)

    p_meoh = partials["MeOH"]
    p_hcl = partials["HCl"]
    p_mecl = partials["MeCl"]
    p_h2o = partials["H2O"]
    p_dme = partials["DME"]

    k_a2 = (1.64e5 / 1.01325) * np.exp(-54.4e3 / (R_GAS * temperature))
    k_hcl = (3.76e-5 / 1.01325) * np.exp(90.6e3 / (R_GAS * temperature))

    k_s = 1.45088e3 * np.exp(-62338.37 / (R_GAS * temperature))
    k_meoh = 9.75e-1 * np.exp(64150.82 / (R_GAS * temperature))
    k_h2o = 6.53e-1 * np.exp(5868.85 / (R_GAS * temperature))

    ratio1 = p_mecl**2 / max(p_meoh * thermo["Keq1"], SMALL)
    num1 = k_a2 * (p_hcl - ratio1)
    denom1 = max(1.0 + k_hcl * ratio1, SMALL)
    r1 = (num1 / denom1) * 1000.0 / 3600.0 * config.bulk_density

    sqrt_inner = np.sqrt(max(k_meoh * p_meoh, 0.0))
    num2 = k_s * k_meoh**2 * (p_meoh**2 - p_dme * p_h2o / max(thermo["Keq2"], SMALL))
    denom2 = max((1.0 + 2.0 * sqrt_inner + k_h2o * p_h2o) ** 4, SMALL)
    r2 = (num2 / denom2) * 1000.0 / 3600.0 * config.bulk_density

    heat_generation = (r1 * (-thermo["dH1"]) + r2 * (-thermo["dH2"])) * 1000.0
    return LocalRates(r1=r1, r2=r2, dH1=thermo["dH1"], dH2=thermo["dH2"], heat_generation=heat_generation)


def reactor_odes(volume: float, state: np.ndarray, config: ReactorConfig, geometry: Geometry) -> list[float]:
    """Coupled species, pressure, reactor temperature, and coolant ODEs."""

    del volume

    flows = np.asarray(state[:5], dtype=float)
    pressure = max(float(state[5]), 1.0)
    temperature = max(float(state[6]), 10.0)
    coolant_temperature = max(float(state[7]), 10.0)

    if np.sum(np.maximum(flows, 0.0)) <= SMALL:
        return [0.0] * 8

    rates = local_rates(flows, pressure, temperature, config)

    dflows_dv = np.array(
        [
            -rates.r1 - 2.0 * rates.r2,
            -rates.r1,
            rates.r1,
            rates.r1 + rates.r2,
            rates.r2,
        ]
    )

    rho = pressure * mixture_molecular_weight(np.maximum(flows, 0.0)) / (R_GAS * temperature)
    dpressure_dz = -geometry.pressure_drop_alpha / max(rho, 1.0e-12)
    dpressure_dv = dpressure_dz / geometry.total_cross_section

    heat_capacity_flow = sum(
        max(flow, 0.0) * shomate_cp(SHOMATE[species], temperature) for flow, species in zip(flows, SPECIES)
    )
    heat_removed = config.overall_heat_transfer * geometry.heat_transfer_area_per_volume * (
        temperature - coolant_temperature
    )

    dtemperature_dv = (rates.heat_generation - heat_removed) / max(heat_capacity_flow, 1.0e-12)
    dcoolant_temperature_dv = heat_removed / max(config.coolant_mass_flow * config.coolant_heat_capacity, 1.0e-12)

    return [*dflows_dv, dpressure_dv, dtemperature_dv, dcoolant_temperature_dv]


def solve_reactor(config: ReactorConfig) -> dict[str, np.ndarray | ReactorConfig | Geometry]:
    """Solve the reactor model and return profiles plus derived summaries."""

    geometry = build_geometry(config)
    inlet = feed_flows(config)
    pressure0 = config.pressure_bar * 1.0e5

    initial_state = [
        inlet["MeOH"],
        inlet["HCl"],
        inlet["MeCl"],
        inlet["H2O"],
        inlet["DME"],
        pressure0,
        config.inlet_temperature,
        config.coolant_inlet_temperature,
    ]
    volume_eval = np.linspace(0.0, config.reactor_volume, config.points)

    solution = solve_ivp(
        lambda volume, state: reactor_odes(volume, state, config, geometry),
        (0.0, config.reactor_volume),
        initial_state,
        t_eval=volume_eval,
        method="RK45",
        rtol=config.rtol,
        atol=config.atol,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    flows = solution.y[:5]
    pressure = solution.y[5]
    temperature = solution.y[6]
    coolant_temperature = solution.y[7]

    r1 = np.zeros_like(solution.t)
    r2 = np.zeros_like(solution.t)
    heat_generation = np.zeros_like(solution.t)
    for index, values in enumerate(zip(flows.T, pressure, temperature)):
        local_flow, local_pressure, local_temperature = values
        rates = local_rates(local_flow, local_pressure, local_temperature, config)
        r1[index] = rates.r1
        r2[index] = rates.r2
        heat_generation[index] = rates.heat_generation / 1000.0

    heat_removed = (
        config.overall_heat_transfer
        * geometry.heat_transfer_area_per_volume
        * (temperature - coolant_temperature)
        / 1000.0
    )

    return {
        "config": config,
        "geometry": geometry,
        "volume": solution.t,
        "flows": flows,
        "pressure": pressure,
        "temperature": temperature,
        "coolant_temperature": coolant_temperature,
        "r1": r1,
        "r2": r2,
        "heat_generation": heat_generation,
        "heat_removed": heat_removed,
    }


def kmol_per_hour(mol_per_second: float | np.ndarray) -> float | np.ndarray:
    return mol_per_second * 3600.0 / 1000.0


def summary_text(result: dict[str, np.ndarray | ReactorConfig | Geometry]) -> str:
    """Format a concise model summary for console output."""

    config = result["config"]
    geometry = result["geometry"]
    volume = result["volume"]
    flows = result["flows"]
    pressure = result["pressure"]
    temperature = result["temperature"]
    coolant_temperature = result["coolant_temperature"]
    heat_removed = result["heat_removed"]

    assert isinstance(config, ReactorConfig)
    assert isinstance(geometry, Geometry)

    inlet = feed_flows(config)
    thermo0 = reaction_thermodynamics(config.inlet_temperature)

    meoh_conversion = (inlet["MeOH"] - flows[0]) / inlet["MeOH"]
    hcl_conversion = (inlet["HCl"] - flows[1]) / inlet["HCl"]
    pressure_drop_bar = (config.pressure_bar * 1.0e5 - pressure[-1]) / 1.0e5
    total_heat_removed_kjs = np.trapezoid(heat_removed, volume)
    total_heat_removed_mw = total_heat_removed_kjs / 1000.0

    lines = [
        "=" * 68,
        f"Reaction thermodynamics at T = {config.inlet_temperature:.2f} K",
        "=" * 68,
        "R1: CH3OH + HCl -> CH3Cl + H2O",
        f"  dH_rxn = {thermo0['dH1']:+.4f} kJ/mol",
        f"  dS_rxn = {thermo0['dS1']:+.4f} J/mol/K",
        f"  dG_rxn = {thermo0['dG1']:+.4f} kJ/mol",
        f"  Keq    = {thermo0['Keq1']:.4f}",
        "",
        "R2: 2 CH3OH -> DME + H2O",
        f"  dH_rxn = {thermo0['dH2']:+.4f} kJ/mol",
        f"  dS_rxn = {thermo0['dS2']:+.4f} J/mol/K",
        f"  dG_rxn = {thermo0['dG2']:+.4f} kJ/mol",
        f"  Keq    = {thermo0['Keq2']:.4f}",
        "",
        "=" * 68,
        "Reactor geometry and performance",
        "=" * 68,
        f"Tube diameter        = {geometry.tube_diameter:.6f} m",
        f"Tube area            = {geometry.tube_area:.8f} m2",
        f"Volume per tube      = {geometry.tube_volume:.6f} m3",
        f"Number of tubes      = {geometry.tube_count}",
        f"Total catalyst mass  = {geometry.catalyst_mass_total:.2f} kg",
        f"Mass flux            = {geometry.mass_flux:.6f} kg/m2/s",
        "",
        f"Final MeOH conversion = {meoh_conversion[-1]:.4f}",
        f"Final HCl conversion  = {hcl_conversion[-1]:.4f}",
        f"Outlet pressure       = {pressure[-1] / 1.0e5:.5f} bar",
        f"Total pressure drop   = {pressure_drop_bar:.5f} bar",
        f"Inlet reactor T       = {temperature[0] - 273.15:.2f} deg C",
        f"Outlet reactor T      = {temperature[-1] - 273.15:.2f} deg C",
        f"Inlet coolant T       = {coolant_temperature[0] - 273.15:.2f} deg C",
        f"Outlet coolant T      = {coolant_temperature[-1] - 273.15:.2f} deg C",
        "",
        "=" * 68,
        "Outlet flowrates, total plant",
        "=" * 68,
    ]

    for species, flow in zip(SPECIES, flows[:, -1]):
        lines.append(f"{species:<8} {kmol_per_hour(flow):>12.4f} kmol/h")
    lines.append(f"{'Total':<8} {np.sum(kmol_per_hour(flows[:, -1])):>12.4f} kmol/h")

    lines.extend(
        [
            "",
            "=" * 68,
            "Heat duty",
            "=" * 68,
            f"Total heat removed = {total_heat_removed_kjs:.2f} kJ/s",
            f"Total heat removed = {total_heat_removed_mw:.4f} MW",
        ]
    )

    return "\n".join(lines)


def plot_profiles(result: dict[str, np.ndarray | ReactorConfig | Geometry], output_path: Path) -> None:
    """Save a multi-panel profile plot."""

    config = result["config"]
    volume = result["volume"]
    flows = result["flows"]
    pressure = result["pressure"]
    temperature = result["temperature"]
    coolant_temperature = result["coolant_temperature"]
    r1 = result["r1"]
    r2 = result["r2"]
    heat_generation = result["heat_generation"]
    heat_removed = result["heat_removed"]

    assert isinstance(config, ReactorConfig)

    inlet = feed_flows(config)
    meoh_conversion = (inlet["MeOH"] - flows[0]) / inlet["MeOH"]
    hcl_conversion = (inlet["HCl"] - flows[1]) / inlet["HCl"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("Packed-bed methyl chloride reactor profiles", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(volume, meoh_conversion, label="MeOH")
    ax.plot(volume, hcl_conversion, label="HCl")
    ax.set_xlabel("Reactor volume, m3")
    ax.set_ylabel("Conversion")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(volume, temperature - 273.15, label="Reactor")
    ax.plot(volume, coolant_temperature - 273.15, label="Coolant")
    ax.set_xlabel("Reactor volume, m3")
    ax.set_ylabel("Temperature, deg C")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(volume, pressure / 1.0e5)
    ax.set_xlabel("Reactor volume, m3")
    ax.set_ylabel("Pressure, bar")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(volume, r1, label="R1")
    ax.plot(volume, r2, label="R2")
    ax.set_xlabel("Reactor volume, m3")
    ax.set_ylabel("Rate, mol/m3/s")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2, 0]
    ax.plot(volume, heat_generation, label="Generated")
    ax.plot(volume, heat_removed, label="Removed")
    ax.set_xlabel("Reactor volume, m3")
    ax.set_ylabel("Heat rate, kJ/m3/s")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2, 1]
    outlet_flows = kmol_per_hour(flows[:, -1])
    colors = ["#3b82f6", "#ef4444", "#10b981", "#06b6d4", "#f59e0b"]
    ax.bar(SPECIES, outlet_flows, color=colors)
    ax.set_ylabel("Outlet flow, kmol/h")
    ax.grid(True, axis="y", alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve and plot a methyl chloride packed-bed reactor model.")
    parser.add_argument("--volume", type=float, default=ReactorConfig.reactor_volume, help="Total reactor volume, m3.")
    parser.add_argument("--hcl-ratio", type=float, default=ReactorConfig.hcl_to_meoh_ratio, help="HCl/MeOH feed ratio.")
    parser.add_argument("--coolant-flow", type=float, default=ReactorConfig.coolant_mass_flow, help="Coolant mass flow, kg/s.")
    parser.add_argument("--points", type=int, default=ReactorConfig.points, help="Number of solution points.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated files.")
    parser.add_argument("--no-plot", action="store_true", help="Print results without saving a plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        ReactorConfig(),
        reactor_volume=args.volume,
        hcl_to_meoh_ratio=args.hcl_ratio,
        coolant_mass_flow=args.coolant_flow,
        points=args.points,
    )

    result = solve_reactor(config)
    print(summary_text(result))

    if not args.no_plot:
        output_path = args.output_dir / "reactor_profiles.png"
        plot_profiles(result, output_path)
        print(f"\nSaved profile plot to: {output_path}")


if __name__ == "__main__":
    main()
