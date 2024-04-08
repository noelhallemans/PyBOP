#
# Weppner Huggins Model
#
import numpy as np
import pybamm


class BaseWeppnerHuggins(pybamm.lithium_ion.BaseModel):
    """WeppnerHuggins Model for GITT.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Weppner & Huggins model"):
        super().__init__({}, name)
        # `self.param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        self.options["working electrode"] = "positive"

        param = self.param
        t = pybamm.t
        ######################
        # Parameters
        ######################

        d_s = pybamm.Parameter("Positive electrode diffusivity [m2.s-1]")

        c_s_max = pybamm.Parameter(
            "Maximum concentration in positive electrode [mol.m-3]"
        )

        i_app = self.param.current_density_with_time

        U = pybamm.Parameter("Reference OCP [V]")

        U_prime = pybamm.Parameter("Derivative of the OCP wrt stoichiometry [V]")

        epsilon = pybamm.Parameter("Positive electrode active material volume fraction")

        r_particle = pybamm.Parameter("Positive particle radius [m]")

        a = 3 * (epsilon / r_particle)


        l_w = self.param.p.L

        ######################
        # Governing equations
        ######################
        u_surf = (2 / (np.pi**0.5)) * (i_app / ((d_s**0.5) * a * self.param.F * l_w)) * (t**0.5)
        # Linearised voltage
        V = U + (U_prime * u_surf) / c_s_max
        ######################
        # (Some) variables
        ######################
        self.variables = {
            "Voltage [V]": V,
            "Time [s]": t,
        }

    @property
    def default_geometry(self):
        return {}

    @property
    def default_submesh_types(self):
        return {}

    @property
    def default_var_pts(self):
        return {}

    @property
    def default_spatial_methods(self):
        return {}

    @property
    def default_solver(self):
        return pybamm.DummySolver()
