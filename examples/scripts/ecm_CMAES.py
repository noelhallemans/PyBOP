import numpy as np
from pybamm import Parameter

import pybop

# Import the ECM parameter set from JSON
parameter_set = pybop.ParameterSet.pybamm("ECM_Example")

# Add Tau as a Parameter
parameter_set.update(
    {
        "Tau1": 0.0,
        "Tau2": 0.0,
        "Element-2 initial overpotential [V]": 0,
        "R2 [Ohm]": Parameter("Tau2") / Parameter("C2 [F]") + 1e-30,
        "C1 [F]": 500,
        "R1 [Ohm]": Parameter("Tau1") / Parameter("C1 [F]") + np.finfo(float).eps,
        "C2 [F]": 5000,
    },
    check_already_exists=False,
)
model = pybop.empirical.Thevenin(
    parameter_set=parameter_set, options={"number of rc elements": 2}
)

# Fitting parameters
parameters = pybop.Parameters(
    pybop.Parameter(
        "Tau1",
        prior=pybop.Gaussian(1e-3, 2e-4),
        bounds=[1e-4, 1e-1],
    ),
    pybop.Parameter(
        "Tau2",
        prior=pybop.Gaussian(0.12, 0.02),
        bounds=[1e-5, 0.1],
    ),
)


sigma = 0.001
t_eval = np.arange(0, 900, 3)
values = model.predict(t_eval=t_eval)
corrupt_values = values["Voltage [V]"].data + np.random.normal(0, sigma, len(t_eval))

# Form dataset
dataset = pybop.Dataset(
    {
        "Time [s]": t_eval,
        "Current function [A]": values["Current [A]"].data,
        "Voltage [V]": corrupt_values,
    }
)

# Generate problem, cost function, and optimisation class
problem = pybop.FittingProblem(model, parameters, dataset)
cost = pybop.SumSquaredError(problem)
optim = pybop.CMAES(cost, max_iterations=100)

results = optim.run()

# Export the parameters to JSON
# parameter_set.export_parameters(
#     "examples/scripts/parameters/fit_ecm_parameters.json", fit_params=parameters
# )

# Plot the time series
pybop.plot.dataset(dataset)

# Plot the timeseries output
pybop.plot.quick(problem, problem_inputs=results.x, title="Optimised Comparison")

# Plot convergence
pybop.plot.convergence(optim)

# Plot the parameter traces
pybop.plot.parameters(optim)

# Plot the cost landscape
pybop.plot.surface(optim)
