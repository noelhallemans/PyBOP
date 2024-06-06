import numpy as np
import pytest

import pybop


class Test_SPM_Parameterisation:
    """
    A class to test the model parameterisation methods.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ground_truth = np.array([0.55, 0.55]) + np.random.normal(
            loc=0.0, scale=0.05, size=2
        )

    @pytest.fixture
    def model(self):
        parameter_set = pybop.ParameterSet.pybamm("Chen2020")
        return pybop.lithium_ion.SPM(parameter_set=parameter_set)

    @pytest.fixture
    def parameters(self):
        return pybop.Parameters(
            pybop.Parameter(
                "Negative electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                bounds=[0.375, 0.725],
            ),
            pybop.Parameter(
                "Positive electrode active material volume fraction",
                prior=pybop.Uniform(0.4, 0.7),
                # no bounds
            ),
        )

    @pytest.fixture(params=[0.4, 0.7])
    def init_soc(self, request):
        return request.param

    @pytest.fixture(
        params=[
            pybop.GaussianLogLikelihoodKnownSigma,
            pybop.GaussianLogLikelihood,
            pybop.RootMeanSquaredError,
            pybop.SumSquaredError,
            pybop.MAP,
        ]
    )
    def cost_class(self, request):
        return request.param

    def noise(self, sigma, values):
        return np.random.normal(0, sigma, values)

    @pytest.fixture
    def spm_costs(self, model, parameters, cost_class, init_soc):
        # Form dataset
        solution = self.get_data(model, parameters, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma0=0.002)
        elif cost_class in [pybop.GaussianLogLikelihood]:
            return cost_class(problem)
        elif cost_class in [pybop.MAP]:
            return cost_class(
                problem, pybop.GaussianLogLikelihoodKnownSigma, sigma0=0.002
            )
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.Adam,
            pybop.CMAES,
            pybop.IRPropMin,
            pybop.NelderMead,
            pybop.SNES,
            pybop.XNES,
        ],
    )
    @pytest.mark.integration
    def test_spm_optimisers(self, optimiser, spm_costs):
        x0 = spm_costs.x0
        # Some optimisers require a complete set of bounds

        # Test each optimiser
        if isinstance(spm_costs, pybop.GaussianLogLikelihood):
            optim = optimiser(
                cost=spm_costs,
                max_iterations=125,
            )
        else:
            optim = optimiser(cost=spm_costs, sigma0=0.05, max_iterations=250)
        if issubclass(optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)
        if issubclass(optimiser, pybop.Adam) and isinstance(
            spm_costs, pybop.GaussianLogLikelihood
        ):
            optim.set_min_iterations(50)

        initial_cost = optim.cost(x0)
        x, final_cost = optim.run()

        # Assertions
        if not isinstance(spm_costs, pybop.GaussianLogLikelihood):
            if not np.allclose(x0, self.ground_truth, atol=1e-5):
                if optim.minimising:
                    assert initial_cost > final_cost
                else:
                    assert initial_cost < final_cost

            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
        else:
            np.testing.assert_allclose(x[:-1], self.ground_truth, atol=1.5e-2)

    @pytest.fixture
    def spm_two_signal_cost(self, parameters, model, cost_class):
        # Form dataset
        init_soc = 0.5
        solution = self.get_data(model, parameters, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
                "Bulk open-circuit voltage [V]": solution[
                    "Bulk open-circuit voltage [V]"
                ].data
                + self.noise(0.002, len(solution["Time [s]"].data)),
            }
        )

        # Define the cost to optimise
        signal = ["Voltage [V]", "Bulk open-circuit voltage [V]"]
        problem = pybop.FittingProblem(
            model, parameters, dataset, signal=signal, init_soc=init_soc
        )

        if cost_class in [pybop.GaussianLogLikelihoodKnownSigma]:
            return cost_class(problem, sigma0=0.002)
        elif cost_class in [pybop.MAP]:
            return cost_class(problem, pybop.GaussianLogLikelihoodKnownSigma)
        else:
            return cost_class(problem)

    @pytest.mark.parametrize(
        "multi_optimiser",
        [
            pybop.SciPyDifferentialEvolution,
            pybop.IRPropMin,
            pybop.CMAES,
        ],
    )
    @pytest.mark.integration
    def test_multiple_signals(self, multi_optimiser, spm_two_signal_cost):
        x0 = spm_two_signal_cost.x0

        # Test each optimiser
        optim = multi_optimiser(
            cost=spm_two_signal_cost,
            sigma0=0.03,
            max_iterations=250,
        )
        if issubclass(multi_optimiser, pybop.BasePintsOptimiser):
            optim.set_max_unchanged_iterations(iterations=35, absolute_tolerance=1e-5)

        initial_cost = optim.cost(spm_two_signal_cost.x0)
        x, final_cost = optim.run()

        # Assertions
        if not isinstance(spm_two_signal_cost, pybop.GaussianLogLikelihood):
            if not np.allclose(x0, self.ground_truth, atol=1e-5):
                if optim.minimising:
                    assert initial_cost > final_cost
                else:
                    assert initial_cost < final_cost
            np.testing.assert_allclose(x, self.ground_truth, atol=1.5e-2)
        else:
            np.testing.assert_allclose(x[:-2], self.ground_truth, atol=1.5e-2)

    @pytest.mark.parametrize("init_soc", [0.4, 0.6])
    @pytest.mark.integration
    def test_model_misparameterisation(self, parameters, model, init_soc):
        # Define two different models with different parameter sets
        # The optimisation should fail as the models are not the same
        second_parameter_set = pybop.ParameterSet.pybamm("Ecker2015")
        second_model = pybop.lithium_ion.SPMe(parameter_set=second_parameter_set)

        # Form dataset
        solution = self.get_data(second_model, parameters, self.ground_truth, init_soc)
        dataset = pybop.Dataset(
            {
                "Time [s]": solution["Time [s]"].data,
                "Current function [A]": solution["Current [A]"].data,
                "Voltage [V]": solution["Voltage [V]"].data,
            }
        )

        # Define the cost to optimise
        problem = pybop.FittingProblem(model, parameters, dataset, init_soc=init_soc)
        cost = pybop.RootMeanSquaredError(problem)

        # Select optimiser
        optimiser = pybop.CMAES

        # Build the optimisation problem
        optim = optimiser(cost=cost)
        initial_cost = optim.cost(cost.x0)

        # Run the optimisation problem
        x, final_cost = optim.run()

        # Assertion for final_cost
        assert initial_cost > final_cost

        # Assertion for x
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(x, self.ground_truth, atol=2e-2)

    def get_data(self, model, parameters, x, init_soc):
        model.parameters = parameters
        experiment = pybop.Experiment(
            [
                (
                    "Discharge at 0.5C for 3 minutes (4 second period)",
                    "Charge at 0.5C for 3 minutes (4 second period)",
                ),
            ]
        )
        sim = model.predict(init_soc=init_soc, experiment=experiment, inputs=x)
        return sim
