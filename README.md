# Methyl Chloride Packed-Bed Reactor Model

This repository contains a non-isothermal packed-bed reactor model for methyl chloride synthesis from methanol and hydrogen chloride, with dimethyl ether as a side product.

The work was developed for a chemical engineering design project on methyl chloride production via gas-phase hydrochlorination of methanol. It couples first-principles species, energy, and momentum balances with temperature-dependent thermodynamics and literature-style LHHW kinetics.

The model includes:

- Coupled molar balances for MeOH, HCl, CH3Cl, H2O, and DME
- Temperature-dependent Shomate thermodynamics
- Reversible kinetic expressions for the main and side reactions
- Ergun-style pressure drop through a multitube packed bed
- Reactor and coolant energy balances
- Console summaries and a generated profile plot

## Reactions

```text
R1: CH3OH + HCl  <-> CH3Cl + H2O
R2: 2 CH3OH     <-> DME + H2O
```

## Quick Start

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the base case:

```bash
python reactor_model.py
```

The script prints a performance summary and saves:

```text
outputs/reactor_profiles.png
```

Run the thermodynamic regression and diagnostic plots:

```bash
python thermodynamics_regression.py
```

This saves:

```text
outputs/thermodynamics/cp_fits.png
outputs/thermodynamics/species_thermodynamics.png
outputs/thermodynamics/reaction_thermodynamics.png
```

## Common Options

```bash
python reactor_model.py --volume 10
python reactor_model.py --hcl-ratio 1.7
python reactor_model.py --coolant-flow 13
python reactor_model.py --no-plot
```

## Notes

- Feed, geometry, coolant, and transport assumptions live in `ReactorConfig` inside `reactor_model.py`.
- HCl, CH3Cl, and H2O use full NIST Shomate coefficients.
- CH3OH and DME use a Shomate-form regression against NIST Cp tables over 200-1500 K; `thermodynamics_regression.py` reproduces the fitted coefficients and plots the diagnostics.
- Main-reaction kinetics follow Becerra, Castro Luna, Ardissone, and Ponzi, "Kinetics of the catalytic hydrochlorination of methanol to methyl chloride", Industrial & Engineering Chemistry Research, 1992, 31, 1040-1045.
- Side-reaction kinetics follow Bercic and Levec, "Catalytic dehydration of methanol to dimethyl ether. Kinetic investigation and reactor simulation", Industrial & Engineering Chemistry Research, 1993, 32, 2478-2484.
- The calculations are intended for engineering exploration and should be checked against validated kinetics, catalyst data, and plant design constraints before design use.
- The coolant flow is treated as total plant coolant flow over the modeled reactor volume.

## Base Case Snapshot

Using the default inputs, the model predicts:

- Final MeOH conversion: `0.9785`
- Final HCl conversion: `0.5737`
- Outlet pressure: `4.56337 bar`
- Total pressure drop: `0.43663 bar`
- Outlet reactor temperature: `311.15 deg C`
- Total heat removed: `0.9990 MW`
