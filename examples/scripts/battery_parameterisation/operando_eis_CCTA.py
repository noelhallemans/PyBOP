import matplotlib.pyplot as plt
import numpy as np
import pybamm
import pybop
from scipy.io import savemat

"""
Example impedance of the grouped parameter SPMe in steady-state compared to operation
"""

## Create model
model_options = {"surface form": "differential", "contact resistance": "true"}
var_pts = {"x_n": 100, "x_s": 20, "x_p": 100, "r_n": 100, "r_p": 100}
model = pybop.lithium_ion.GroupedSPMe(
    options=model_options,
)
## Group parameter set
R0 = 0.01

parameter_set = pybamm.ParameterValues("Chen2020")
parameter_set["Electrolyte diffusivity [m2.s-1]"] = 1.769e-10
parameter_set["Electrolyte conductivity [S.m-1]"] = 1e16
parameter_set["Negative electrode conductivity [S.m-1]"] = 1e16
parameter_set["Positive electrode conductivity [S.m-1]"] = 1e16

parameter_values = pybop.lithium_ion.GroupedSPMe.create_grouped_parameters(
    parameter_set
)
parameter_values["Series resistance [Ohm]"] = R0
parameter_values["Positive electrode charge transfer time scale [s]"] /= 1.2
parameter_values["Negative electrode charge transfer time scale [s]"] /= 1.2
parameter_values["Nominal cell capacity [A.h]"] *= 2
parameter_values["Measured cell capacity [A.s]"] *= 2
Q = parameter_values["Measured cell capacity [A.s]"] / 3600  # Cell capacity [A.h]

## Compute steady state EIS
SOC_EIS = 50  # SoC at which EIS will be compared
n_frequency = 60
fmin = 20e-3
fmax = 1e3

f_eval = np.logspace(np.log10(fmin), np.log10(fmax), n_frequency)  # Frequency vector
parameter_values["Initial SoC"] = SOC_EIS / 100

solution_steadystate = pybop.pybamm.EISSimulator(
    model, parameter_values=parameter_values, var_pts=var_pts, f_eval=f_eval
).solve()
impedance_steadystate = solution_steadystate["Impedance"].data

## Set up and run a charge experiment
SOC0 = 0
N = 2000  # Number of simulation points
C_rates = [0.25, 0.5, 1]  # C-rate
N_C_rates = len(C_rates)

parameter_values["Initial SoC"] = SOC0 / 100
impedance_operando = np.zeros((n_frequency, N_C_rates), dtype=complex)
for ii in range(N_C_rates):
    C_rate = C_rates[ii]
    i0 = Q * C_rate  # DC current [A]
    T = 3600 / C_rate  # Simulation time [s]
    dataset = pybop.Dataset(
        {
            "Time [s]": np.linspace(0, T, N),
            "Current function [A]": -i0 * np.ones(N),
        }
    )

    sim = pybop.pybamm.Simulator(
        model, parameter_values=parameter_values, var_pts=var_pts, protocol=dataset
    )
    solution = sim.solve()
    # solution.plot()

    ## Compute operando EIS
    solution_operando = pybop.pybamm.EISSimulator(
        model,
        parameter_values=parameter_values,
        var_pts=var_pts,
        f_eval=f_eval,
        protocol=dataset,
    ).solve()
    impedance_operando_tmp = solution_operando["Impedance"].data
    index_SOC_EIS = int(
        round((SOC_EIS - SOC0) / (100 - SOC0) * N)
    )  # Index at which SOC_EIS is reached
    impedance_operando[:, ii] = impedance_operando_tmp[
        index_SOC_EIS, :
    ]  # Operando impedance at C-rate and SOC_EIS


## Plot EIS
fig, ax = plt.subplots()
ax.plot(
    np.real(impedance_steadystate),
    -np.imag(impedance_steadystate),
    color="k",
)
for i in range(N_C_rates):
    ax.plot(
        np.real(impedance_operando[:, i]),
        -np.imag(impedance_operando[:, i]),
        color="gray",
    )
ax.set(xlabel=r"$Z_r(\omega)$ [$\Omega$]", ylabel=r"$-Z_j(\omega)$ [$\Omega$]")
ax.set_aspect("equal", "box")
ax.legend()
ax.set_ylim([0, ax.get_xlim()[1]])
plt.show()

savemat(
    "/Users/engs2621/Library/CloudStorage/OneDrive-Nexus365/Documents OneDrive/Conferences/CCTA2026/Impedance comparison figure/impedanceComparisonSteadyStateOperando.mat",
    {
        "f_eval": f_eval,
        "C_rates": C_rates,
        "SOC_EIS": SOC_EIS,
        "impedance_steadystate": impedance_steadystate,
        "impedance_operando": impedance_operando,
    },
)
print("Saved!")
