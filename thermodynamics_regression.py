"""Fit and report Shomate-style thermodynamic properties.

The reactor model uses NIST Shomate coefficients where available. For methanol
and dimethyl ether, this script fits the Shomate heat-capacity form to tabulated
NIST Cp data and derives F/G so that H(298.15 K) and S(298.15 K) match the
chosen reference values.

Run:
    python thermodynamics_regression.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))

import numpy as np


R_GAS = 8.314  # J/mol/K


def cp_shomate(temperature: np.ndarray | float, coefficients: np.ndarray) -> np.ndarray | float:
    """Cp(T), J/mol/K, using the NIST Shomate convention."""

    t = np.asarray(temperature) / 1000.0
    return coefficients[0] + coefficients[1] * t + coefficients[2] * t**2 + coefficients[3] * t**3 + coefficients[4] / t**2


def h_sensible(temperature: np.ndarray | float, coefficients: np.ndarray) -> np.ndarray | float:
    """H(T) - H(298.15 K), kJ/mol."""

    t = np.asarray(temperature) / 1000.0
    return (
        coefficients[0] * t
        + coefficients[1] * t**2 / 2.0
        + coefficients[2] * t**3 / 3.0
        + coefficients[3] * t**4 / 4.0
        - coefficients[4] / t
        + coefficients[5]
        - coefficients[7]
    )


def h_absolute(temperature: np.ndarray | float, coefficients: np.ndarray, h_formation_298: float) -> np.ndarray | float:
    """Absolute enthalpy, kJ/mol, relative to the formation enthalpy at 298.15 K."""

    return h_formation_298 + h_sensible(temperature, coefficients)


def entropy(temperature: np.ndarray | float, coefficients: np.ndarray) -> np.ndarray | float:
    """S(T), J/mol/K."""

    t = np.asarray(temperature) / 1000.0
    return (
        coefficients[0] * np.log(t)
        + coefficients[1] * t
        + coefficients[2] * t**2 / 2.0
        + coefficients[3] * t**3 / 3.0
        - coefficients[4] / (2.0 * t**2)
        + coefficients[6]
    )


NIST_SHOMATE = {
    "HCl": np.array(
        [32.12392, -13.45805, 19.86852, -6.853936, -0.049672, -101.6206, 228.6866, -92.31201]
    ),
    "CH3Cl": np.array(
        [3.524690, 136.9277, -82.14196, 20.22797, 0.278032, -89.19995, 202.8391, -83.68000]
    ),
    "H2O": np.array(
        [30.09200, 6.832514, 6.793435, -2.534480, 0.082139, -250.8810, 223.3967, -241.8264]
    ),
}

H_FORMATION_298 = {
    "HCl": -92.307,
    "CH3Cl": -83.680,
    "H2O": -241.826,
    "CH3OH": -205.000,
    "DME": -184.100,
}

CP_DATA = {
    "CH3OH": {
        "temperature": np.array(
            [200, 273.15, 298.15, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
        ),
        "cp": np.array(
            [39.71, 42.59, 44.06, 44.17, 51.63, 59.70, 67.19, 73.86, 79.76, 84.95, 89.54, 93.57, 97.12, 100.24, 102.98, 105.40]
        ),
        "s298": 239.9,
    },
    "DME": {
        "temperature": np.array(
            [
                100,
                150,
                200,
                272.20,
                273.15,
                298.15,
                300,
                300.76,
                333.25,
                370.42,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
            ]
        ),
        "cp": np.array(
            [
                42.27,
                48.99,
                54.47,
                62.01,
                62.56,
                65.57,
                65.80,
                65.90,
                70.33,
                75.14,
                78.68,
                91.36,
                102.86,
                113.03,
                121.99,
                129.84,
                136.70,
                142.69,
                147.89,
                152.41,
                156.35,
                159.77,
            ]
        ),
        "s298": 267.0,
    },
}


def fit_shomate(
    temperature_data: np.ndarray,
    cp_data: np.ndarray,
    s298_reference: float,
    temperature_min: float,
    temperature_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit A-E to Cp data and derive F/G from 298.15 K reference constraints."""

    mask = (temperature_data >= temperature_min) & (temperature_data <= temperature_max)
    t_fit = temperature_data[mask] / 1000.0
    cp_fit = cp_data[mask]

    design = np.column_stack([np.ones_like(t_fit), t_fit, t_fit**2, t_fit**3, 1.0 / t_fit**2])
    coefficients, residual_sum, rank, _ = np.linalg.lstsq(design, cp_fit, rcond=None)
    if rank < design.shape[1]:
        raise RuntimeError("Shomate Cp regression design matrix is rank deficient.")
    a, b, c, d, e = coefficients

    t0 = 298.15 / 1000.0
    poly298 = a * t0 + b * t0**2 / 2.0 + c * t0**3 / 3.0 + d * t0**4 / 4.0 - e / t0
    f = -poly298
    h = 0.0

    entropy_poly298 = a * np.log(t0) + b * t0 + c * t0**2 / 2.0 + d * t0**3 / 3.0 - e / (2.0 * t0**2)
    g = s298_reference - entropy_poly298

    fitted = np.array([a, b, c, d, e, f, g, h])
    residuals = cp_fit - design @ coefficients
    rmse = float(np.sqrt(np.mean(residuals**2)))
    r2 = float(1.0 - np.sum(residuals**2) / np.sum((cp_fit - np.mean(cp_fit)) ** 2))
    degrees_of_freedom = max(len(cp_fit) - design.shape[1], 1)
    variance = (float(residual_sum[0]) if len(residual_sum) else float(np.sum(residuals**2))) / degrees_of_freedom
    covariance = variance * np.linalg.inv(design.T @ design)
    stderr = np.sqrt(np.diag(covariance))

    return fitted, stderr, rmse, r2


def build_species() -> dict[str, dict[str, object]]:
    """Return all species metadata, including fitted CH3OH and DME coefficients."""

    methanol_coefficients, methanol_stderr, methanol_rmse, methanol_r2 = fit_shomate(
        CP_DATA["CH3OH"]["temperature"], CP_DATA["CH3OH"]["cp"], CP_DATA["CH3OH"]["s298"], 200.0, 1500.0
    )
    dme_coefficients, dme_stderr, dme_rmse, dme_r2 = fit_shomate(
        CP_DATA["DME"]["temperature"], CP_DATA["DME"]["cp"], CP_DATA["DME"]["s298"], 200.0, 1500.0
    )

    return {
        "HCl": {
            "coefficients": NIST_SHOMATE["HCl"],
            "h_formation_298": H_FORMATION_298["HCl"],
            "temperature_min": 298.0,
            "temperature_max": 1200.0,
            "source": "NIST Shomate, Chase 1998",
        },
        "CH3Cl": {
            "coefficients": NIST_SHOMATE["CH3Cl"],
            "h_formation_298": H_FORMATION_298["CH3Cl"],
            "temperature_min": 298.0,
            "temperature_max": 1200.0,
            "source": "NIST Shomate, Chase 1998",
        },
        "H2O": {
            "coefficients": NIST_SHOMATE["H2O"],
            "h_formation_298": H_FORMATION_298["H2O"],
            "temperature_min": 500.0,
            "temperature_max": 1200.0,
            "source": "NIST Shomate, Chase 1998",
        },
        "CH3OH": {
            "coefficients": methanol_coefficients,
            "stderr": methanol_stderr,
            "h_formation_298": H_FORMATION_298["CH3OH"],
            "temperature_min": 200.0,
            "temperature_max": 1500.0,
            "source": f"NIST Cp regression, RMSE={methanol_rmse:.4f}, R2={methanol_r2:.6f}",
        },
        "DME": {
            "coefficients": dme_coefficients,
            "stderr": dme_stderr,
            "h_formation_298": H_FORMATION_298["DME"],
            "temperature_min": 200.0,
            "temperature_max": 1500.0,
            "source": f"NIST Cp regression, RMSE={dme_rmse:.4f}, R2={dme_r2:.6f}",
        },
    }


def print_fit_report(species: dict[str, dict[str, object]]) -> None:
    print("=" * 78)
    print("Shomate Cp regression for species without full NIST Shomate coefficients")
    print("Cp = A + B*t + C*t^2 + D*t^3 + E/t^2, where t = T/1000")
    print("=" * 78)

    for name in ("CH3OH", "DME"):
        data = species[name]
        coefficients = data["coefficients"]
        stderr = data["stderr"]
        print(f"\n{name}: {data['source']}")
        for label, value, error in zip("ABCDE", coefficients[:5], stderr):
            print(f"  {label} = {value:+12.5f} +/- {error:.5f}")
        print(f"  F = {coefficients[5]:+12.5f}  [H_sensible(298.15 K)=0 enforced]")
        print(f"  G = {coefficients[6]:+12.5f}  [S(298.15 K) reference enforced]")
        print(f"  H = {coefficients[7]:+12.5f}")


def print_species_table(species: dict[str, dict[str, object]]) -> None:
    temperatures = [298.15, 350.0, 400.0, 450.0, 500.0, 550.0, 573.15, 600.0, 650.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]

    for name, data in species.items():
        coefficients = data["coefficients"]
        h_formation_298 = float(data["h_formation_298"])
        temperature_min = float(data["temperature_min"])
        print("\n" + "=" * 88)
        print(
            f"{name}: {data['source']} | Hf(298)={h_formation_298:.3f} kJ/mol | "
            f"range {temperature_min:.0f}-{float(data['temperature_max']):.0f} K"
        )
        print("=" * 88)
        print(f"{'T (K)':>10} {'Cp (J/mol/K)':>16} {'H-H298 (kJ/mol)':>18} {'H_abs (kJ/mol)':>18} {'S (J/mol/K)':>16}")
        for temperature in temperatures:
            if temperature < temperature_min - 0.5:
                print(f"{temperature:10.2f} {'[below range]':>16}")
                continue
            print(
                f"{temperature:10.2f} "
                f"{cp_shomate(temperature, coefficients):16.4f} "
                f"{h_sensible(temperature, coefficients):18.4f} "
                f"{h_absolute(temperature, coefficients, h_formation_298):18.4f} "
                f"{entropy(temperature, coefficients):16.4f}"
            )


def reaction_properties(
    temperature: np.ndarray | float,
    products: list[tuple[np.ndarray, float]],
    reactants: list[tuple[np.ndarray, float]],
    delta_h_formation_298: float,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    dcp = sum(nu * cp_shomate(temperature, coefficients) for coefficients, nu in products) - sum(
        nu * cp_shomate(temperature, coefficients) for coefficients, nu in reactants
    )
    sensible_delta = sum(nu * h_sensible(temperature, coefficients) for coefficients, nu in products) - sum(
        nu * h_sensible(temperature, coefficients) for coefficients, nu in reactants
    )
    ds = sum(nu * entropy(temperature, coefficients) for coefficients, nu in products) - sum(
        nu * entropy(temperature, coefficients) for coefficients, nu in reactants
    )
    dh = delta_h_formation_298 + sensible_delta
    dg = dh * 1000.0 - np.asarray(temperature) * ds
    keq = np.exp(-dg / (R_GAS * np.asarray(temperature)))
    return dcp, dh, ds, dg / 1000.0, keq


def print_reaction_table(species: dict[str, dict[str, object]]) -> None:
    temperatures = [500.0, 550.0, 573.15, 600.0, 650.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0]

    c_ch3oh = species["CH3OH"]["coefficients"]
    c_hcl = species["HCl"]["coefficients"]
    c_ch3cl = species["CH3Cl"]["coefficients"]
    c_h2o = species["H2O"]["coefficients"]
    c_dme = species["DME"]["coefficients"]

    delta_hf_r1 = H_FORMATION_298["CH3Cl"] + H_FORMATION_298["H2O"] - H_FORMATION_298["CH3OH"] - H_FORMATION_298["HCl"]
    delta_hf_r2 = H_FORMATION_298["DME"] + H_FORMATION_298["H2O"] - 2.0 * H_FORMATION_298["CH3OH"]

    reactions = [
        (
            "R1: CH3OH + HCl -> CH3Cl + H2O",
            [(c_ch3cl, 1.0), (c_h2o, 1.0)],
            [(c_ch3oh, 1.0), (c_hcl, 1.0)],
            delta_hf_r1,
        ),
        (
            "R2: 2 CH3OH -> DME + H2O",
            [(c_dme, 1.0), (c_h2o, 1.0)],
            [(c_ch3oh, 2.0)],
            delta_hf_r2,
        ),
    ]

    print("\n" + "=" * 92)
    print("Reaction thermodynamics, limited to T >= 500 K by the H2O Shomate range")
    print("=" * 92)

    for label, products, reactants, delta_hf_298 in reactions:
        print(f"\n{label} | dHf(298) = {delta_hf_298:.3f} kJ/mol")
        print(f"{'T (K)':>10} {'dCp':>12} {'dH_rxn':>12} {'dS_rxn':>12} {'dG_rxn':>12} {'Keq':>14}")
        for temperature in temperatures:
            dcp, dh, ds, dg, keq = reaction_properties(temperature, products, reactants, delta_hf_298)
            print(f"{temperature:10.2f} {dcp:12.4f} {dh:12.4f} {ds:12.4f} {dg:12.4f} {keq:14.4f}")


def plot_results(species: dict[str, dict[str, object]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"CH3OH": "#2563eb", "HCl": "#dc2626", "CH3Cl": "#16a34a", "H2O": "#f97316", "DME": "#7c3aed"}

    fig1, axes1 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig1.suptitle("Cp(T): NIST Shomate equations and regression fits", fontsize=13, fontweight="bold")
    for index, name in enumerate(["HCl", "CH3Cl", "H2O", "CH3OH", "DME"]):
        data = species[name]
        ax = axes1.flatten()[index]
        temperature = np.linspace(float(data["temperature_min"]), float(data["temperature_max"]), 400)
        ax.plot(temperature, cp_shomate(temperature, data["coefficients"]), color=colors[name], linewidth=2.5, label="Shomate/fit")
        if name in CP_DATA:
            ax.scatter(CP_DATA[name]["temperature"], CP_DATA[name]["cp"], color="black", s=22, zorder=5, label="NIST Cp data")
        ax.axvspan(500.0, 620.0, alpha=0.08, color="green", label="reactor range")
        ax.axvline(573.15, color="gray", linestyle=":", linewidth=1.0)
        ax.set_xlabel("Temperature, K")
        ax.set_ylabel("Cp, J/mol/K")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes1.flatten()[-1].set_visible(False)
    fig1.savefig(output_dir / "cp_fits.png", dpi=180)
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig2.suptitle("Species thermodynamic properties", fontsize=13, fontweight="bold")
    for name, data in species.items():
        temperature = np.linspace(float(data["temperature_min"]), float(data["temperature_max"]), 300)
        coefficients = data["coefficients"]
        h_formation_298 = float(data["h_formation_298"])
        axes2[0].plot(temperature, h_sensible(temperature, coefficients), color=colors[name], linewidth=2, label=name)
        axes2[1].plot(temperature, h_absolute(temperature, coefficients, h_formation_298), color=colors[name], linewidth=2, label=name)
        axes2[2].plot(temperature, entropy(temperature, coefficients), color=colors[name], linewidth=2, label=name)
    for ax, title, ylabel in zip(
        axes2,
        ["Sensible heat", "Absolute enthalpy", "Standard entropy"],
        ["H-H298, kJ/mol", "H, kJ/mol", "S, J/mol/K"],
    ):
        ax.axvline(573.15, color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Temperature, K")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig2.savefig(output_dir / "species_thermodynamics.png", dpi=180)
    plt.close(fig2)

    c_ch3oh = species["CH3OH"]["coefficients"]
    c_hcl = species["HCl"]["coefficients"]
    c_ch3cl = species["CH3Cl"]["coefficients"]
    c_h2o = species["H2O"]["coefficients"]
    c_dme = species["DME"]["coefficients"]
    delta_hf_r1 = H_FORMATION_298["CH3Cl"] + H_FORMATION_298["H2O"] - H_FORMATION_298["CH3OH"] - H_FORMATION_298["HCl"]
    delta_hf_r2 = H_FORMATION_298["DME"] + H_FORMATION_298["H2O"] - 2.0 * H_FORMATION_298["CH3OH"]
    temperature = np.linspace(500.0, 1200.0, 400)
    r1 = reaction_properties(temperature, [(c_ch3cl, 1.0), (c_h2o, 1.0)], [(c_ch3oh, 1.0), (c_hcl, 1.0)], delta_hf_r1)
    r2 = reaction_properties(temperature, [(c_dme, 1.0), (c_h2o, 1.0)], [(c_ch3oh, 2.0)], delta_hf_r2)

    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig3.suptitle("Reaction thermodynamics", fontsize=13, fontweight="bold")
    labels = [
        ("Delta Cp, J/mol/K", "Reaction Delta Cp"),
        ("Delta H, kJ/mol", "Reaction enthalpy"),
        ("Delta S, J/mol/K", "Reaction entropy"),
        ("Delta G, kJ/mol", "Gibbs energy"),
    ]
    for index, (ylabel, title) in enumerate(labels):
        ax = axes3.flatten()[index]
        ax.plot(temperature, r1[index], color="#2563eb", linewidth=2.5, label="R1")
        ax.plot(temperature, r2[index], color="#dc2626", linewidth=2.5, linestyle="--", label="R2")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axvline(573.15, color="gray", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Temperature, K")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes3.flatten()[4]
    ax.semilogy(temperature, r1[4], color="#2563eb", linewidth=2.5, label="Keq R1")
    ax.semilogy(temperature, r2[4], color="#dc2626", linewidth=2.5, linestyle="--", label="Keq R2")
    ax.axvline(573.15, color="gray", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Temperature, K")
    ax.set_ylabel("Keq")
    ax.set_title("Equilibrium constants")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    axes3.flatten()[-1].set_visible(False)
    fig3.savefig(output_dir / "reaction_thermodynamics.png", dpi=180)
    plt.close(fig3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Shomate-style thermodynamics and generate diagnostic plots.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "thermodynamics", help="Directory for generated plots.")
    parser.add_argument("--no-plots", action="store_true", help="Print tables without generating plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    species = build_species()
    print_fit_report(species)
    print_species_table(species)
    print_reaction_table(species)
    if not args.no_plots:
        plot_results(species, args.output_dir)
        print(f"\nSaved thermodynamic plots to: {args.output_dir}")


if __name__ == "__main__":
    main()
