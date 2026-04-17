"""
@author: zahraetemad
(A)COMPLETE THERMODYNAMIC PROPERTIES — ALL SPECIES
  CH3OH, HCl, CH3Cl, H2O, DME

  Shomate Equation (NIST convention, t = T/1000):

    Cp°(T) = A + B·t + C·t² + D·t³ + E/t²                        [J/mol·K]
    H°(T)  = A·t + B·t²/2 + C·t³/3 + D·t⁴/4 − E/t + F − H       [kJ/mol]
             (this is H°(T) − H°₂₉₈, i.e. sensible heat)
    S°(T)  = A·ln(t) + B·t + C·t²/2 + D·t³/3 − E/(2t²) + G       [J/mol·K]

  Species with FULL NIST Shomate coefficients (A–H given):
    HCl     Chase 1998  298–1200 K
    CH3Cl   Chase 1998  298–1200 K
    H2O     Chase 1998  500–1200 K

  Species with tabulated Cp data only (regression fitting):
    CH3OH   Fit Shomate form to NIST Cp table (200–1500 K)
    DME     Fit Shomate form to NIST Cp table (200–1500 K)
            H and S derived from fitted coefficients + Hf/S(298) references

  Standard enthalpies of formation Hf(298 K) [kJ/mol]:
    HCl:-92.307  CH3Cl:-83.680  H2O:-241.826  CH3OH:-205.000  DME:-184.100
    
(B) Reactions
Main reaction:
   CH3OH + HCl -> CH3Cl + H2O

Side reaction:
   2 CH3OH <-> DME + H2O

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 0) SHOMATE / THERMODYNAMIC CONSTANTS
# =============================================================================
c_HCl  = np.array([32.12392, -13.45805,  19.86852, -6.853936, -0.049672,
                   -101.6206, 228.6866, -92.31201])
c_MeCl = np.array([3.524690, 136.9277, -82.14196, 20.22797, 0.278032,
                   -89.19995, 202.8391, -83.68000])
c_H2O  = np.array([30.09200, 6.832514, 6.793435, -2.534480, 0.082139,
                   -250.8810, 223.3967, -241.8264])
c_MeOH = np.array([1.02174, 145.69696, -71.23345, 13.60109, 0.49387,
                   -4.52149, 203.52073, 0.0])
c_DME  = np.array([5.42132, 216.58801, -105.26906, 19.60664, 0.39139,
                   -9.03898, 215.69206, 0.0])

Hf298 = {
    "MeOH": -205.000,
    "HCl":  -92.307,
    "MeCl": -83.680,
    "H2O":  -241.826,
    "DME":  -184.100
}

dHf_R1 = Hf298["MeCl"] + Hf298["H2O"] - Hf298["MeOH"] - Hf298["HCl"]
dHf_R2 = Hf298["DME"]  + Hf298["H2O"] - 2 * Hf298["MeOH"]


# 1) CONSTANTS AND OPERATING CONDITIONS
# =============================================================================
R_gas = 8.314
T0    = 280 + 273.15
Ta0   = 260 + 273.15
P_bar = 5.0
P0    = P_bar * 1e5

# 2) BED / TUBE PROPERTIES
# =============================================================================
eps   = 0.52
rho_b = 570.0
#rho_b=(1-eps)*rho_b
#rho_b=rho_b/(1-eps)
Dp    = 0.0035
Dc    = 8 * Dp
Ac    = np.pi / 4.0 * Dc**2

V_total =10.0
L_tube  = 8.0
V_tube  = Ac * L_tube
n_tubes = int(np.ceil(V_total / V_tube))

W_tube  = rho_b * V_tube
W_total = rho_b * V_total

# 3) FEED
# =============================================================================
FA0 = 152.813 * 1000.0 / 3600.0
FB0 = FA0*1.7

MW_MeOH = 32.04e-3
MW_HCl  = 36.4606e-3
MW_MeCl = 50.4875e-3
MW_H2O  = 18.0153e-3
MW_DME  = 46.069e-3

m_dot      = FA0 * MW_MeOH + FB0 * MW_HCl
m_dot_tube = m_dot / n_tubes

mu_mix    = 2.26851e-5
MW_mix_in = (FA0 * MW_MeOH + FB0 * MW_HCl) / (FA0 + FB0)
rho0_mix  = P0 * MW_mix_in / (R_gas * T0)

G = m_dot_tube / Ac

# 4) COOLANT PROPERTIES
# =============================================================================
m_dot_cool = 13.0      # kg/s coolant
Cp_cool    = 1507.0    # J/kg/K coolant Cp

# 5) MIXTURE MOLECULAR WEIGHT
# =============================================================================
def mixture_mw(FA, FB, FP, FW, FE):
    FT = FA + FB + FP + FW + FE
    if FT <= 0.0:
        return MW_mix_in
    return (FA*MW_MeOH + FB*MW_HCl + FP*MW_MeCl + FW*MW_H2O + FE*MW_DME) / FT


# 6) COUPLED PFR + ERGUN ODEs
# =============================================================================
alpha_term = (G / Dp) * ((1.0 - eps) / eps**3) * (
    150.0 * (1.0 - eps) * mu_mix / Dp + 1.75 * G
)
dV_dz = Ac * n_tubes


def coupled_odes(V_vol, y):
    FA, FB, FP, FW, FE, P, T_loc, Ta_loc = y

    P      = max(P, 1.0)
    T_loc  = max(T_loc, 10.0)
    Ta_loc = max(Ta_loc, 10.0)

    # Local total flow
    # -------------------------------------------------------------------------
    FT = FA + FB + FP + FW + FE
    if FT <= 1e-14:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    P_local_bar = P / 1e5

    pA = FA / FT * P_local_bar   # MeOH
    pB = FB / FT * P_local_bar   # HCl
    pP = FP / FT * P_local_bar   # MeCl
    pW = FW / FT * P_local_bar   # H2O
    pE = FE / FT * P_local_bar   # DME

    # Local thermodynamics at T_loc
    # -------------------------------------------------------------------------
    t = T_loc / 1000.0

    def H_sensible_local(c):
        return c[0]*t + c[1]*t**2/2 + c[2]*t**3/3 + c[3]*t**4/4 - c[4]/t + c[5] - c[7]

    def S_shomate_local(c):
        return c[0]*np.log(t) + c[1]*t + c[2]*t**2/2 + c[3]*t**3/3 - c[4]/(2*t**2) + c[6]

    def H_sensible_ref(Tref, c):
        tref = Tref / 1000.0
        return c[0]*tref + c[1]*tref**2/2 + c[2]*tref**3/3 + c[3]*tref**4/4 - c[4]/tref + c[5] - c[7]

    T_ref = 298.15

    def Cp_integral_local(c):
        return H_sensible_local(c) - H_sensible_ref(T_ref, c)

    dHrxn_1 = dHf_R1 + (
        Cp_integral_local(c_MeCl) + Cp_integral_local(c_H2O)
        - Cp_integral_local(c_MeOH) - Cp_integral_local(c_HCl)
    )
    dHrxn_2 = dHf_R2 + (
        Cp_integral_local(c_DME) + Cp_integral_local(c_H2O)
        - 2.0 * Cp_integral_local(c_MeOH)
    )

    dSrxn_1 = (
        S_shomate_local(c_MeCl) + S_shomate_local(c_H2O)
        - S_shomate_local(c_MeOH) - S_shomate_local(c_HCl)
    )
    dSrxn_2 = (
        S_shomate_local(c_DME) + S_shomate_local(c_H2O)
        - 2.0 * S_shomate_local(c_MeOH)
    )

    dGrxn_1 = dHrxn_1 - T_loc * dSrxn_1 / 1000.0
    dGrxn_2 = dHrxn_2 - T_loc * dSrxn_2 / 1000.0

    Ka_eq = np.exp(-dGrxn_1 * 1000.0 / (R_gas * T_loc))
    Ks_eq = np.exp(-dGrxn_2 * 1000.0 / (R_gas * T_loc))

    # Local kinetics at T_loc
    # -------------------------------------------------------------------------
    kA2   = (1.64e5 / 1.01325) * np.exp(-54.4e3    / (R_gas * T_loc))
    KHCl  = (3.76e-5 / 1.01325) * np.exp( 90.6e3   / (R_gas * T_loc))

    ks    = 1.45088e3 * np.exp(-62338.37 / (R_gas * T_loc))
    KMeOH = 9.75e-1   * np.exp( 64150.82 / (R_gas * T_loc))
    KH2O  = 6.53e-1   * np.exp( 5868.85  / (R_gas * T_loc))

    # Local reaction rates at T_loc
    # -------------------------------------------------------------------------
    
    num1   = kA2 * (pB - pP**2 / (pA * Ka_eq))
    denom1 = max(1.0 + KHCl * pP**2 / (pA * Ka_eq), 1e-14)
    r1_kgh = num1 / denom1
    r1     = r1_kgh * 1000.0 / 3600.0 * rho_b   # mol/m3/s
    #eta1=
    #r1=r*eta1
    
    num2       = ks * KMeOH**2 * (pA**2 - pE * pW / Ks_eq)
    denom2     = max((1.0 + 2.0*np.sqrt(max(KMeOH * pA, 0.0)) + KH2O*pW)**4, 1e-14)
    r2_kgh     = num2 / denom2
    r2         = r2_kgh * 1000.0 / 3600.0 * rho_b   # mol/m3/s
    

    # Molar balances
    # -------------------------------------------------------------------------
    dFA_dV = -r1 - 2.0*r2
    dFB_dV = -r1
    dFP_dV =  r1
    dFW_dV =  r1 + r2
    dFE_dV =  r2

    # Pressure drop
    # -------------------------------------------------------------------------
    MW_mix = mixture_mw(FA / n_tubes, FB / n_tubes, FP / n_tubes,
                        FW / n_tubes, FE / n_tubes)
    rho = P * MW_mix / (R_gas * T_loc)

    dPdz  = -alpha_term / max(rho, 1e-12)
    dP_dV = dPdz / dV_dz

    # Cp at T_loc
    # -------------------------------------------------------------------------
    def Cp_shomate_local(c):
        return c[0] + c[1]*t + c[2]*t**2 + c[3]*t**3 + c[4]/t**2

    Cp_A = Cp_shomate_local(c_MeOH)
    Cp_B = Cp_shomate_local(c_HCl)
    Cp_C = Cp_shomate_local(c_MeCl)
    Cp_D = Cp_shomate_local(c_DME)
    Cp_E = Cp_shomate_local(c_H2O)

    # Energy balances
    # -------------------------------------------------------------------------
    U = 600.0 #w/m2/k
    a = 4.0 / Dc #1/m

    FCp = FA*Cp_A + FB*Cp_B + FP*Cp_C + FE*Cp_D + FW*Cp_E
    
    heat_gen = (r1 * (-dHrxn_1) + r2 * (-dHrxn_2)) * 1000.0   # J/m3/s
    heat_rem = U * a * (T_loc - Ta_loc)                       # J/m3/s

    dT_dV  = (heat_gen - heat_rem) / max(FCp, 1e-12)
    dTa_dV = heat_rem / max(m_dot_cool * Cp_cool, 1e-12)
    return [dFA_dV, dFB_dV, dFP_dV, dFW_dV, dFE_dV, dP_dV, dT_dV, dTa_dV]

# 7) SOLVE COUPLED SYSTEM
# =============================================================================
V_eval = np.linspace(0.0, V_total, 1000)

sol = solve_ivp(
    coupled_odes,
    (0.0, V_total),
    [FA0, FB0, 0.0, 0.0, 0.0, P0, T0, Ta0],
    t_eval=V_eval,
    method='RK45',     # Runge–Kutta method
    rtol=1e-6,
    atol=1e-8
)

if not sol.success:
    raise RuntimeError(sol.message)

V_sol = sol.t
FA, FB, FP, FW, FE, P_sol, T_sol, Ta_sol = sol.y

# 8) POST-PROCESS LOCAL RATES / HEAT PROFILE
# =============================================================================
r1_sol = np.zeros_like(V_sol)
r2_sol = np.zeros_like(V_sol)
Qg_profile = np.zeros_like(V_sol)

for i in range(len(V_sol)):
    FAi, FBi, FPi, FWi, FEi = FA[i], FB[i], FP[i], FW[i], FE[i]
    Pi = max(P_sol[i], 1.0)
    Ti = max(T_sol[i], 10.0)

    FT = FAi + FBi + FPi + FWi + FEi
    if FT <= 1e-14:
        continue

    P_local_bar = Pi / 1e5
    pA = FAi / FT * P_local_bar
    pB = FBi / FT * P_local_bar
    pP = FPi / FT * P_local_bar
    pW = FWi / FT * P_local_bar
    pE = FEi / FT * P_local_bar

    t = Ti / 1000.0

    def H_sensible_local(c):
        return c[0]*t + c[1]*t**2/2 + c[2]*t**3/3 + c[3]*t**4/4 - c[4]/t + c[5] - c[7]

    def S_shomate_local(c):
        return c[0]*np.log(t) + c[1]*t + c[2]*t**2/2 + c[3]*t**3/3 - c[4]/(2*t**2) + c[6]

    def H_sensible_ref(Tref, c):
        tref = Tref / 1000.0
        return c[0]*tref + c[1]*tref**2/2 + c[2]*tref**3/3 + c[3]*tref**4/4 - c[4]/tref + c[5] - c[7]

    T_ref = 298.15

    def Cp_integral_local(c):
        return H_sensible_local(c) - H_sensible_ref(T_ref, c)

    dHrxn_1 = dHf_R1 + (
        Cp_integral_local(c_MeCl) + Cp_integral_local(c_H2O)
        - Cp_integral_local(c_MeOH) - Cp_integral_local(c_HCl)
    )
    dHrxn_2 = dHf_R2 + (
        Cp_integral_local(c_DME) + Cp_integral_local(c_H2O)
        - 2.0 * Cp_integral_local(c_MeOH)
    )

    dSrxn_1 = (
        S_shomate_local(c_MeCl) + S_shomate_local(c_H2O)
        - S_shomate_local(c_MeOH) - S_shomate_local(c_HCl)
    )
    dSrxn_2 = (
        S_shomate_local(c_DME) + S_shomate_local(c_H2O)
        - 2.0 * S_shomate_local(c_MeOH)
    )

    dGrxn_1 = dHrxn_1 - Ti * dSrxn_1 / 1000.0
    dGrxn_2 = dHrxn_2 - Ti * dSrxn_2 / 1000.0

    Ka_eq = np.exp(-dGrxn_1 * 1000.0 / (R_gas * Ti))
    Ks_eq = np.exp(-dGrxn_2 * 1000.0 / (R_gas * Ti))

    kA2   = (1.64e5 / 1.01325) * np.exp(-54.4e3    / (R_gas * Ti))
    KHCl  = (3.76e-5 / 1.01325) * np.exp( 90.6e3   / (R_gas * Ti))
    ks    = 1.45088e3 * np.exp(-62338.37 / (R_gas * Ti))
    KMeOH = 9.75e-1   * np.exp( 64150.82 / (R_gas * Ti))
    KH2O  = 6.53e-1   * np.exp( 5868.85  / (R_gas * Ti))

    ratio1 = pP**2 / (pA * Ka_eq) if pA > 1e-14 else 0.0
    num1   = kA2 * (pB - ratio1)
    denom1 = max(1.0 + KHCl * ratio1, 1e-14)
    r1_sol[i] = (num1 / denom1) * 1000.0 / 3600.0 * rho_b

    inner      = KMeOH * pA
    sqrt_inner = np.sqrt(max(inner, 0.0))
    num2       = ks * KMeOH**2 * (pA**2 - pE * pW / Ks_eq)
    denom2     = max((1.0 + 2.0*sqrt_inner + KH2O*pW)**4, 1e-14)
    r2_sol[i]  = (num2 / denom2) * 1000.0 / 3600.0 * rho_b

    Qg_profile[i] = r1_sol[i] * (-dHrxn_1) + r2_sol[i] * (-dHrxn_2)

W_sol = rho_b * V_sol
XA = (FA0 - FA) / FA0
XB = (FB0 - FB) / FB0

P_bar_sol  = P_sol / 1e5
P_ratio    = P_sol / P0
deltaP_bar = (P0 - P_sol[-1]) / 1e5

def to_kmh(F_mol_s):
    return F_mol_s * 3600.0 / 1000.0

# 9) HEAT DUTY
# =============================================================================
U = 600.0
a = 4.0 / Dc

Qr_profile = U * a * (T_sol - Ta_sol) / 1000.0
Q_total_kJs = np.trapz(Qr_profile, V_sol)
Q_total_MW = Q_total_kJs / 1000.0


# 10) PRINT SUMMARY
# =============================================================================
print("="*60)
print(f"  Reaction Thermodynamics at T = {T0:.2f} K ({T0-273.15:.0f} °C)")
print("="*60)

# inlet thermodynamics only for reporting
t0 = T0 / 1000.0

def H_sensible0(c):
    return c[0]*t0 + c[1]*t0**2/2 + c[2]*t0**3/3 + c[3]*t0**4/4 - c[4]/t0 + c[5] - c[7]

def S_shomate0(c):
    return c[0]*np.log(t0) + c[1]*t0 + c[2]*t0**2/2 + c[3]*t0**3/3 - c[4]/(2*t0**2) + c[6]

def H_sensible_ref0(Tref, c):
    tref = Tref / 1000.0
    return c[0]*tref + c[1]*tref**2/2 + c[2]*tref**3/3 + c[3]*tref**4/4 - c[4]/tref + c[5] - c[7]

def Cp_integral0(c):
    return H_sensible0(c) - H_sensible_ref0(298.15, c)

dHrxn1_0 = dHf_R1 + (Cp_integral0(c_MeCl) + Cp_integral0(c_H2O) - Cp_integral0(c_MeOH) - Cp_integral0(c_HCl))
dHrxn2_0 = dHf_R2 + (Cp_integral0(c_DME) + Cp_integral0(c_H2O) - 2.0*Cp_integral0(c_MeOH))

dSrxn1_0 = S_shomate0(c_MeCl) + S_shomate0(c_H2O) - S_shomate0(c_MeOH) - S_shomate0(c_HCl)
dSrxn2_0 = S_shomate0(c_DME) + S_shomate0(c_H2O) - 2.0*S_shomate0(c_MeOH)

dGrxn1_0 = dHrxn1_0 - T0 * dSrxn1_0 / 1000.0
dGrxn2_0 = dHrxn2_0 - T0 * dSrxn2_0 / 1000.0

Keq1_0 = np.exp(-dGrxn1_0 * 1000.0 / (R_gas * T0))
Keq2_0 = np.exp(-dGrxn2_0 * 1000.0 / (R_gas * T0))

for rxn in [1, 2]:
    label = "R1: CH3OH+HCl→CH3Cl+H2O" if rxn == 1 else "R2: 2CH3OH→DME+H2O"
    print(f"\n  {label}")
    if rxn == 1:
        print(f"    dH_rxn = {dHrxn1_0:+.4f}  kJ/mol")
        print(f"    dS_rxn = {dSrxn1_0:+.4f}  J/mol/K")
        print(f"    dG_rxn = {dGrxn1_0:+.4f}  kJ/mol")
        print(f"    Keq    = {Keq1_0:.4f}")
    else:
        print(f"    dH_rxn = {dHrxn2_0:+.4f}  kJ/mol")
        print(f"    dS_rxn = {dSrxn2_0:+.4f}  J/mol/K")
        print(f"    dG_rxn = {dGrxn2_0:+.4f}  kJ/mol")
        print(f"    Keq    = {Keq2_0:.4f}")

print(f"\nTube diameter  = {Dc:.6f} m")
print(f"Tube area      = {Ac:.8f} m2")
print(f"Volume/tube    = {V_tube:.6f} m3")
print(f"Number tubes   = {n_tubes}")
print(f"W_catalyst tot = {W_total:.2f} kg")
print(f"G              = {G:.6f} kg/m2/s")
print(f"\nFinal MeOH conversion = {XA[-1]:.4f}")
print(f"Final HCl  conversion = {XB[-1]:.4f}")
print(f"Outlet pressure       = {P_bar_sol[-1]:.5f} bar")
print(f"Total pressure drop   = {deltaP_bar:.5f} bar")
print(f"Inlet  reactor temperature   = {T_sol[0]-273.15:.2f} °C")
print(f"Outlet reactor temperature  = {T_sol[-1]-273.15:.2f} °C")
print(f"Inlet  coolant temperature  = {Ta_sol[0]-273.15:.2f} °C")
print(f"Outlet coolant temperature = {Ta_sol[-1]-273.15:.2f} °C")

print(f"\n{'='*55}")
print("OUTLET FLOWRATES (TOTAL PLANT, kmol/h)")
print(f"{'='*55}")
names   = ["Methanol (MeOH)", "HCl", "Methyl Chloride", "Water (H2O)", "DME"]
outlets = [FA[-1], FB[-1], FP[-1], FW[-1], FE[-1]]
for name, F in zip(names, outlets):
    print(f"{name:<20} {to_kmh(F):>12.4f}")
print(f"{'Total':<20} {sum(to_kmh(F) for F in outlets):>12.4f}")

print("\n" + "="*50)
print("HEAT DUTY")
print("="*50)
print(f"Total heat removed = {Q_total_kJs:.2f} kJ/s")
print(f"Total heat removed = {Q_total_MW:.4f} MW")


