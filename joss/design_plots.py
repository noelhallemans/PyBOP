# A script to generate design optimisation plots for the JOSS paper.

import numpy as np
import pybamm
from pybamm import Parameter

import pybop
from pybop.plot import PlotlyManager

go = PlotlyManager().go
np.random.seed(8)

# Choose which plots to show and save
create_plot = {}
create_plot["gravimetric"] = True  # takes longer
create_plot["prediction"] = True


# Define parameter set and model
parameter_set = pybop.ParameterSet.pybamm("Chen2020", formation_concentrations=True)
parameter_set.update(
    {
        "Electrolyte density [kg.m-3]": Parameter("Separator density [kg.m-3]"),
        "Negative electrode active material density [kg.m-3]": Parameter(
            "Negative electrode density [kg.m-3]"
        ),
        "Negative electrode carbon-binder density [kg.m-3]": Parameter(
            "Negative electrode density [kg.m-3]"
        ),
        "Positive electrode active material density [kg.m-3]": Parameter(
            "Positive electrode density [kg.m-3]"
        ),
        "Positive electrode carbon-binder density [kg.m-3]": Parameter(
            "Positive electrode density [kg.m-3]"
        ),
        "Positive electrode porosity": 1.0
        - Parameter("Positive electrode active material volume fraction"),
    },
    check_already_exists=False,
)
model = pybop.lithium_ion.SPMe(
    parameter_set=parameter_set, solver=pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Positive electrode thickness [m]",
        initial_value=7.56e-05,
        prior=pybop.Gaussian(7.56e-05, 3e-05),
        bounds=[50e-06, 120e-06],
    ),
    pybop.Parameter(
        "Positive electrode active material volume fraction",
        initial_value=0.58,
        prior=pybop.Gaussian(0.58, 0.1),
        bounds=[0.3, 0.8],
    ),
)

# Define test protocol
experiment = pybop.Experiment(["Discharge at 1C until 2.5 V (10 second period)"])

# Generate problem, cost and optimiser classes
problem = pybop.DesignProblem(
    model,
    parameters,
    experiment,
    signal=["Voltage [V]", "Current [A]"],
    initial_state={"Initial SoC": 1.0},
    update_capacity=True,
)
cost = pybop.GravimetricEnergyDensity(problem)
optim = pybop.PSO(
    cost,
    verbose=True,
    allow_infeasible_solutions=False,
    max_iterations=250,
    max_unchanged_iterations=25,
)

# Run optimisation
result = optim.run()
print("Estimated parameters:", result.x)
print(f"Initial gravimetric energy density: {cost(optim.x0):.2f} Wh.kg-1")
print(f"Optimised gravimetric energy density: {cost(result.x):.2f} Wh.kg-1")

if create_plot["gravimetric"]:
    # Plot the cost landscape with optimisation path
    gravimetric_fig = pybop.plot.contour(optim, steps=65, title=None)
    gravimetric_fig.write_image("joss/figures/design_gravimetric.png")

if create_plot["prediction"]:
    # Plot the timeseries output
    figs = pybop.plot.quick(
        problem,
        problem_inputs=result.x,
        title=None,
        width=576,
        height=576,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        show=False,
    )
    prediction_fig = figs[0]
    prediction_fig.data[0].name = f"Initial result: {cost(optim.x0):.2f} Wh.kg-1"
    prediction_fig.data[1].name = f"Optimised result: {cost(result.x):.2f} Wh.kg-1"
    prediction_fig.show()
    prediction_fig.write_image("joss/figures/design_prediction.png")
