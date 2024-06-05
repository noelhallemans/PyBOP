import pybop


class BaseProblem:
    """
    Base class for defining a problem within the PyBOP framework, compatible with PINTS.

    Parameters
    ----------
    parameters : pybop.Parameter or pybop.Parameters
        An object or list of the parameters for the problem.
    model : object, optional
        The model to be used for the problem (default: None).
    check_model : bool, optional
        Flag to indicate if the model should be checked (default: True).
    signal: List[str]
      The signal to observe.
    additional_variables : List[str], optional
        Additional variables to observe and store in the solution (default: []).
    init_soc : float, optional
        Initial state of charge (default: None).
    """

    def __init__(
        self,
        parameters,
        model=None,
        check_model=True,
        signal=["Voltage [V]"],
        additional_variables=[],
        init_soc=None,
    ):
        # Check if parameters is a list of pybop.Parameter objects
        if isinstance(parameters, list):
            if all(isinstance(param, pybop.Parameter) for param in parameters):
                parameters = pybop.Parameters(*parameters)
            else:
                raise TypeError(
                    "All elements in the list must be pybop.Parameter objects."
                )
        # Check if parameters is a single pybop.Parameter object
        elif isinstance(parameters, pybop.Parameter):
            parameters = pybop.Parameters(parameters)
        # Check if parameters is already a pybop.Parameters object
        elif not isinstance(parameters, pybop.Parameters):
            raise TypeError(
                "The input parameters must be a pybop Parameter, a list of pybop.Parameter objects, or a pybop Parameters object."
            )

        self.parameters = parameters
        self._model = model
        self.check_model = check_model
        if isinstance(signal, str):
            signal = [signal]
        elif not all(isinstance(item, str) for item in signal):
            raise ValueError("Signal should be either a string or list of strings.")
        self.signal = signal
        self.init_soc = init_soc
        self.n_outputs = len(self.signal)
        self._time_data = None
        self._target = None

        if isinstance(model, pybop.BaseModel):
            self.additional_variables = additional_variables
        else:
            self.additional_variables = []

        # Set initial values
        self.x0 = self.parameters.initial_value()

    @property
    def n_parameters(self):
        return len(self.parameters)

    def evaluate(self, x):
        """
        Evaluate the model with the given parameters and return the signal.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def evaluateS1(self, x):
        """
        Evaluate the model with the given parameters and return the signal and
        its derivatives.

        Parameters
        ----------
        x : np.ndarray
            Parameter values to evaluate the model at.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def time_data(self):
        """
        Returns the time data.

        Returns
        -------
        np.ndarray
            The time array.
        """
        return self._time_data

    def get_target(self):
        """
        Return the target dataset.

        Returns
        -------
        np.ndarray
            The target dataset array.
        """
        return self._target

    def set_target(self, dataset):
        """
        Set the target dataset.

        Parameters
        ----------
        target : np.ndarray
            The target dataset array.
        """
        if self.signal is None:
            raise ValueError("Signal must be defined to set target.")
        if not isinstance(dataset, pybop.Dataset):
            raise ValueError("Dataset must be a pybop Dataset object.")

        self._target = {signal: dataset[signal] for signal in self.signal}

    @property
    def model(self):
        return self._model
