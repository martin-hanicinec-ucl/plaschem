import numpy as np
import pandas as pd
from scipy import constants

from pygmo_fwork.pygmol.model import Model
from pygmo_fwork.pygmol.exceptions import InitialSolutionError
from pygmo_fwork.pygmol.equations import Equations
from plaschem.chemistry import Chemistry
from plaschem.exceptions import ChemistryConsistencyError


class NeutralChemistry(Chemistry):
    """A Chemistry subclass representing a plasma chemistry consisting of neutral species only.
    Electron or ions are not permitted in this class, as they will trigger the consistency error.
    """
    requires_charge = False  # this is used as a flag by the base class for consistency checks.

    def assert_consistency(self):
        """Redefinition of the base-class method. Supers the base and adds the assertion of all species being neutral"""
        # check that all species are neutral:
        species_charge = self.get_species_charge()
        if not (species_charge.values == 0).all():
            raise ChemistryConsistencyError('NeutralChemistry cannot contain charged species!')
        super().assert_consistency()

    def get_reactions_stoich_coefs_electron(self, disabled=False, method='net'):
        """Redefinition of the base-class method, which always returns a zero vector. It is here just for
        consistency, or for the possibility that the model or equation instances loading the chemistry actually
        look for it or need it.

        :param disabled: (bool) if to include disabled species
        :param method: (str) of {'lhs', 'rhs', 'net'} where 'lhs' will be positive coefs for LHS's of reactions,
                       'rhs' positive coefs for RHS's of reactions and 'net' will be 'rhs' - 'lhs' (net production
                       of species)
        :return: (pd.Series) with index of reaction ids.
        """
        reactions = self.get_reactions(disabled=disabled)
        reactions_stoich_coefs_electron = pd.Series(index=reactions.index, dtype='i')  # signed integer
        return reactions_stoich_coefs_electron.fillna(0)


# noinspection DuplicatedCode
class NeutralEquations(Equations):
    """Equations class modified for a neutral only chemistry set. The electron energy density and it's derivative are
    held at 0.
    """

    def __init__(self, chemistry, model_params):
        super().__init__(chemistry, model_params)
        assert not self.species_charge.any()

    # redefine functions which are built into the secondary solution:
    def get_electron_temperature(self, y, n_e=None, rho=None):
        return np.nan

    def get_electron_density(self, y, n=None):
        return 0.

    # redefine necessary methods without electrons and ions, only to speed it up...
    def get_reaction_rate_coefficients(self, y, temp_e=None):
        return self.k_r

    def get_mean_speeds(self, y, temp_i=None):
        return self.v_m

    def get_sigma_sc(self, y, v_m=None, debye_length=None):
        return self.sigma_sc

    def get_diffusivities(self, y, diff_c_free=None, diff_a_pos=None, diff_a_neg=None):
        return diff_c_free

    # redefine the objective function factory, so it does not calculate unnecessary clutter
    def get_objective_functions(self, jacobian=False):
        if jacobian:
            raise NotImplementedError('Jacobian function not implemented!')
        obj_function_jacobian = None

        drho_dt = 0.

        def obj_function(t, y):
            n = self.get_density_vector(y)
            n_tot = self.get_total_density(y, n=n)
            p = self.get_total_pressure(y, n_tot=n_tot)
            k = self.get_reaction_rate_coefficients(y)
            rates = self.get_reaction_rates(y, n=n, n_e=0., n_tot=n_tot, k_r=k)
            source_vol = self.get_volumetric_source_rates(y, rates=rates)
            source_flow = self.get_flow_source_rates(y, n=n, p=p)
            v_m = self.get_mean_speeds(y)
            sigma_sc = self.get_sigma_sc(y, v_m=v_m)
            mfp = self.get_mean_free_paths(y, n=n, sigma_sc=sigma_sc)
            diff_c_free = self.get_free_diffusivities(y, mfp=mfp, v_m=v_m)
            diff = self.get_diffusivities(y, diff_c_free=diff_c_free)
            wall_fluxes = self.get_wall_fluxes(y, n=n, diff_c=diff, v_m=v_m)
            source_diff_sinks = self.get_diffusion_sinks(y, wall_fluxes=wall_fluxes)
            source_diff_sources = self.get_diffusion_sources(y, diff_sinks=source_diff_sinks)
            source_diff = self.get_diffusion_source_rates(y, diff_sinks=source_diff_sinks,
                                                          diff_sources=source_diff_sources)
            min_n_cor = self.get_min_n_correction(y, n=n)
            dn_dt = self.get_dn_dt(y, vol_source_rates=source_vol, flow_source_rates=source_flow,
                                   diff_source_rates=source_diff, min_n_correction=min_n_cor)

            dy_dt = self.get_dy_dt(t, y, dn_dt=dn_dt, drho_dt=drho_dt)

            return dy_dt

        return obj_function, obj_function_jacobian


class NeutralModel(Model):
    """Base Model class modified for the use with NeutralEquations and NeutralChemistry instances - designed as an
    ad-hoc model for very simple thermal neutral-only chemistry sets.
    The methods here are not documented, refer to the base class documentation for all the description.
    """

    def __init__(self, chemistry, model_params):
        assert not any(chemistry.get_species_charge()), 'The passed chemistry needs to be neutral!'
        super().__init__(chemistry, model_params)

    def _build_equations(self):
        self.equations = NeutralEquations(self.chemistry, self.model_params)

    def _build_y0(self, *args, **kwargs):
        gas_temp = self.model_params['Tg']  # (initial) gas temperature
        pressure = self.model_params['p']  # initial gas pressure
        feeds_dict = self.model_params['feeds']  # dict of feeds in [sccm] with sp names as keys
        n_tot = pressure / (constants.k * gas_temp)  # total initial density in [m-3]
        non_feed_sp_fraction = 1.e-15

        feeds = np.array(
            [feeds_dict[sp] if sp in feeds_dict else 0. for sp in self.chemistry.get_species_name()]
        )  # feeds in [sccm]

        if feeds.sum() > 0:
            n0 = np.ones(self.chemistry.num_species()) * n_tot * non_feed_sp_fraction  # seed the non-feed densities
            n_residual = n_tot - n0.sum()
            n0[feeds > 0] = feeds[feeds > 0] / feeds.sum() * n_residual  # rest between feed densities in proportion
        else:  # if no feed flows defined, distribute density between all the species...
            n0 = np.ones(self.chemistry.num_species()) * n_tot / self.chemistry.num_species()

        return np.r_[n0, 0.]  # keep electron energy density as NaN

    def _y0_from_init_params(self, init_el_temp, init_n):
        """The arguments only to keep the signature of the base method, but the init_el_temp is not used here"""
        if init_n is not None:
            n_series = pd.Series(init_n)
            if list(sorted(n_series.index)) != list(sorted(list(self.chemistry.get_species_name()))):
                raise InitialSolutionError('The init_n dict needs to contain all the species names as keys!')
            n_array = np.array(n_series[list(self.chemistry.get_species_name())])  # reorder
            y0 = np.r_[n_array, 0.]
        else:
            y0 = self._build_y0()
        return y0

    def _validate_y0(self, y0):
        # check for the correct dimension:
        if len(y0) != self.dimension():
            raise InitialSolutionError('Incorrect dimension of the initial values vector!')
