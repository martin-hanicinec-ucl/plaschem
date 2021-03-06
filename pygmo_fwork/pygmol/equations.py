import numpy as np
from scipy import constants

from pygmo_fwork.pygmol.exceptions import EquationsParametersConsistencyError
from pygmo_fwork.pygmol.model_parameters import ModelParameters


class Equations(object):
    """
    This class contains methods for calculating the objective functions (systems of ODEs and their Jacobian) for
    a plasma global model. Contains methods calculating partial results which are used for the final solution
    calculation. This is by design, so the methods for partial results also can be used for diagnostics.

    The main purpose of this class is the get_objective_functions method which will return one of the following:
        - (objective function, None) {for jac=False}
        - (objective function, jacobian function) {for jac=True}
            - The objective function is a function f(t, y) returning the vector (dy_1/dt, dy_2/dt, ..., dy_N/dt).
              This function can be fed into various different ODE solvers. In this case, y = (n_1, n_2, ..., n_N, rho)
              where n is a vector of heavy species densities and rho is the electron energy density n_e*3/2*T_e.
              It's called objective function within this framework, since this is a function which would be minimised
              for a steady state solution.
            - The jacobian function takes the same parameters as the objective function and returns the jacobian
              matrix of the objective function. This might be needed for some ODE solvers.
    """

    # TODO: Might want to think about if I want to store the chemistry as an attribute and call a higher-level method
    #       of chemistry itself to get reaction rate coefficients with el_temp and gas_temp as parameters. This way
    #       I could overcome the restriction of the Arrhenius form and also it would simplify things...

    diffusion_model = 1  # selects a diffusion model - see self.get_wall_fluxes docs.

    def __init__(self, chemistry, model_params):
        """Equations initializer. One of the parameters is an instance of a Chemistry class. This class is describing
        the chemistry kinetic scheme and needs to define some mandatory methods, for the Equations to work properly.
        The Chemistry class needs to define the following methods:

        * Methods returning information about the species (heavy species) for which the continuity equations are
          solved. These methods all return 1D arrays of length of number of 'ordinary' heavy species tracked by
          the model (density of electron is determined by neutrality and density of arbitrary specie 'M' is a sum of
          densities of all species - these two 'special' species do not figure in the arrays returned by following
          methods):
            - Chemistry.get_species_name()
                Returns array of (atr) of species names
            - Chemistry.get_species_charge()
                Returns array of (int) of species charges (unit = elementary charge e)
            - Chemistry.get_species_mass()
                Returns array of (float) of species masses (unit = amu)
            - Chemistry.get_species_lj_sigma()
                Returns array of (float) of Lennard-Jones sigma parameters (unit = Angstrom)
            - Chemistry.get_species_stick_coef()
                Returns array of (float) of species sticking coefficients. These are all between 0. - 1. and denote
                ratio of each species which gets stuck to a surface if it reaches the surface.

        * Methods returning information about reactions (between the species defined above and electron and 'M'). All
          return 1D arrays of length of number of reactions between species. Reaction rate coefficients are
          described by the Arrhenius formula - by 3 parameters arrh_a, arrh_b and arrh_c. arrh_a is in [SI]
          For electron collisions: k = arrh_a * (el_temp/1eV)**arrh_b * exp(-el_temp/arrh_c), arrh_c is in [eV]
          For heavy particle collisions: k = arrh_a * (gas_temp/300K)**arrh_b * exp(-gas_temp/arrh_c), arrh_c is in [K]
            - Chemistry.get_reactions_arrh_a()
                Returns array of (float) of arrh_a parameters for all the reactions
            - Chemistry.get_reactions_arrh_b()
                Returns array of (float) of arrh_b parameters for all the reactions
            - Chemistry.get_reactions_arrh_c()
                Returns array of (float) of arrh_c parameters for all the reactions
            - Chemistry.get_reactions_el_en_loss()
                Returns array of (float) of electron energy losses in [eV] for all the reactions. These should be
                only non-zero for inelastic collisions featuring electron in both reactants and products. These will
                typically be the first non-zero energy in a cross section or the activation energy (arrh_c parameter)
            - Chemistry.get_reactions_elastic()
                Returns array of (bool) flags for elastic collisions. Note, that even a reaction with the same
                reactants and products might be non-elastic (simply modelling the electron energy loss channels by
                excitations without explicitly including the excited products for example)

            - Chemistry.get_reactions_stoich_coefs_electron(method=['net', 'lhs', 'rhs'])
                Returns array of (int) of stoichiometric coefficients of electron in all the reactions. This method
                takes a parameter 'method' from {'net', 'lhs', 'rhs'}. For lhs and rhs, the stoichiometric coefficients
                are non-negative only for left-hand-side or right-hand-side of the reaction, whereas for 'net',
                the stoichiometric coefficients are total production of species (equivalent to 'rhs' - 'lhs').
            - Chemistry.get_reactions_stoich_coefs_arbitrary(method=['net', 'lhs', 'rhs'])
                Returns array of (int) of stoichiometric coefficients of arbitrary species M in all the reactions.
                This method takes a parameter 'method' from {'net', 'lhs', 'rhs'}. For lhs and rhs,
                the stoichiometric coefficients are non-negative only for left-hand-side or right-hand-side of the
                reaction, whereas for 'net', the stoichiometric coefficients are total production of species
                (equivalent to 'rhs' - 'lhs'). If none of the reactions include arbitrary species 'M', the method
                still needs to be implemented and needs to return all zeros.

        * Methods returning 2D arrays linking either species to species (return species matrix) or species to reactions
          (stoichiometric matrix)
            - Chemistry.get_return_matrix()
                Returns 2D array of shape (Ns, Ns) of (float), where Ns is number of species (without electron and M).
                The return species matrix is built in a way that R[i, j] (ith row and jth column) denotes number of
                i-th species created by each one j-th species STUCK to the surface. Non-zero values of R[i, j]
                therefore only makes sense if Chemistry.get_reactions_stick_coef()[j] > 0.

            - Chemistry.get_stoichiomatrix(method=['net', 'lhs', 'rhs'])
                Returns 2D array of Nr rows and Ns columns of (float), where Ns is number of species
                (without electron and M) and Nr is number of reactions. In this matrix, S[i, j] (ith row and jth column)
                is a stoichiometric coefficient of j-th species in i-th reaction. This method takes argument 'method'.
                If method is 'lhs' or 'rhs', the stoichiometric coefficients are non-negative values of simple number
                of species in reactants (LHS) or products (RHS). If method=='net', the stoichiometric coefficient
                is the net production of the specie[j] (or consumption if negative) and equivalent to rhs - lhs.


        :param chemistry: (Chemistry instance) Look above into the __init__ docstring for a documentation on what
                          methods and attributes the Chemistry class needs to define!
        :param model_params: (dict or ModelParameters) There are two options regarding this parameter:
                             1 Supply this parameter by the means of the ModelParameters instance. Consult the
                               ModelParameters docstrings for more information. One example of the ModelParameters
                               instance might be:
                                    model_params = ModelParameters(
                                        radius=0.1,  # [m]
                                        length=0.1,  # [m]
                                        pressure=100,  # [Pa]
                                        power=(0, 0, 1000, 1000, 0, 0),  # [W]
                                        t_power=(0, 0.001, 0.001, 0.01, 0.01, 1000),  # [s]
                                        feeds={'Ar': 100, 'He', 300},  # [sccm]
                                        t_end=0.1  # [s]
                                    )
                               The ModelParameters class handles all the validation. The above case corresponds to
                               the 1kW power pulse from 0.001 - 0.01 sec and simulation over time domain 0 - 0.1 sec.
                               The t_power parameter needs to have at least 10% overlap over the t_end. Together
                               power and t_power form a "open polygon path" of the power pulse. For the constant
                               power, use e.g.:
                                    ModelParameters(
                                        ...
                                        power=(1000, 1000),
                                        t_power=(-1000, 1000),
                                        ...
                                    )
                               The model parameters need to be consistent with the passed chemistry instance,
                               meaning al the species names included in the feeds dict need to be in the chemistry
                               (need to be present in the chemistry.get_species_name() array). If they are not,
                               ChemistryConsistencyError will be raised
                             2 Supply the model parameters by the means of a dict. In this case the ModelParameters
                               object instance is created from the dict in the code, to ensure all the validators
                               defined in the ModelParameters class are met. in this case, the parameter might be
                                    model_params = {
                                        'radius': 0.1,
                                        'length': 0.1,
                                        'pressure': 100,
                                        'power': (1000, 1000),
                                        't_power': (0, 1),
                                        'feeds': {'Ar': 1000},
                                        't_end': 0.1
                                    }
                               all with the same units as above.
        """

        # ***************************************** MODEL PARAMETERS ************************************************* #
        # validate the model parameters:
        if not isinstance(model_params, ModelParameters):
            model_params = ModelParameters(model_params)
        self.model_params = model_params  # this is ModelParameters custom dict with validated consistent values

        # ***************************************** CHEMISTRY ******************************************************** #
        # species attributes:
        self.species_name = np.array(chemistry.get_species_name())
        self.species_charge = np.array(chemistry.get_species_charge())  # species charges in elementary charges
        self.species_mass = np.array(chemistry.get_species_mass()) * constants.atomic_mass  # species mass in [kg]
        self.species_lj_sigma = np.array(chemistry.get_species_lj_sigma()) * 1.e-10  # Lennard-Jones sigma in [m]
        self.species_stick_coef = np.array(chemistry.get_species_stick_coef())
        self.num_species = len(self.species_name)
        # reactions attributes:
        self.reactions_arrh_a = np.array(chemistry.get_reactions_arrh_a())  # arrhenius pre-exp. factor in [SI]
        self.reactions_arrh_b = np.array(chemistry.get_reactions_arrh_b())  # arrhenius n-factor
        self.reactions_arrh_c = np.array(chemistry.get_reactions_arrh_c())  # arrhenius activation en [eV] or [K]
        self.reactions_el_en_loss = np.array(chemistry.get_reactions_el_en_loss())  # [eV] for inelastic collisions
        self.num_reactions = len(self.reactions_arrh_a)

        self.stoichiovector_electron_net = np.array(chemistry.get_reactions_stoich_coefs_electron(method='net'))
        self.stoichiovector_electron_lhs = np.array(chemistry.get_reactions_stoich_coefs_electron(method='lhs'))
        self.stoichiovector_arbitrary_lhs = np.array(chemistry.get_reactions_stoich_coefs_arbitrary(method='lhs'))

        # multi-d attributes:
        self.return_matrix = np.array(chemistry.get_return_matrix())
        self.stoichiomatrix_net = np.array(chemistry.get_stoichiomatrix(method='net'))
        self.stoichiomatrix_lhs = np.array(chemistry.get_stoichiomatrix(method='lhs'))
        # stoich matrix of LHS for all reactions only with 1st col electrons and last columns arbitrary_sp 'M'
        self.stoichiomatrix_reactants_lhs = np.c_[
            self.stoichiovector_electron_lhs, self.stoichiomatrix_lhs, self.stoichiovector_arbitrary_lhs
        ]

        # ***************************************** MODEL PARAMETERS ************************************************* #
        self.feeds = model_params['feeds']  # dict with species names as keys
        self.temp_g = model_params['temp_gas']
        self.r, self.z = model_params['radius'], model_params['length']
        self.p0 = model_params['pressure']
        self.t_end = model_params['t_end']

        self.pow = np.array(model_params['power'])
        self.t_pow = np.array(model_params['t_power'])

        # adjust repeated values in P_t, since P_t needs to be strictly ascending - need to make it continuous func
        d_t = 0.00001 * self.t_end
        for i in range(len(self.t_pow) - 1):
            if self.t_pow[i] == self.t_pow[i + 1]:
                self.t_pow[i] -= d_t
                self.t_pow[i + 1] += d_t
        assert np.array_equal(self.t_pow, sorted(self.t_pow)), 'Time points for power definition ill-defined!'
        assert len(self.t_pow) == len(set(self.t_pow)), 'Time points for power definition ill-defined'
        # I want the border points to be included so I don't need to handle extrapolations...
        assert self.t_pow[0] <= 0. and self.t_pow[-1] >= self.t_end, 'The power definition needs to cover [0, t_end]'
        # treated as a non-constant power even if constant power defined e.g. by P=(1000, 1000), t_P=(-1e10, 1e10)

        # secondary and other model parameters:
        self.volume = constants.pi * self.r ** 2 * self.z  # plasma volume in [m3]
        self.area = 2 * constants.pi * self.r * (self.r + self.z)
        self.diff_l = ((constants.pi / self.z) ** 2 + (2.405 / self.r) ** 2) ** -0.5  # diffusion length in [m]

        # ********************************************** OTHERS ****************************************************** #
        self.num_unknowns = self.num_species + 1

        # useful constants:
        self.mtorr = 0.133322  # 1 mtorr in Pa

        # masking arrays:
        self.mask_pos = self.species_charge > 0  # positive species
        self.mask_neg = self.species_charge < 0  # negative species
        self.mask_neu = self.species_charge == 0  # neutral species
        self.mask_ion = self.mask_pos | self.mask_neg  # ion species
        self.mask_inl = np.array([sp in self.feeds for sp in self.species_name])  # feed gas species all species

        self.mask_elc = self.stoichiovector_electron_lhs > 0  # electron collisions
        self.mask_els = np.array(chemistry.get_reactions_elastic())  # elastic collisions for all reactions

        # ************************  STATIC QUANTITIES EXTRACTED FROM THE CHEMISTRY  ********************************** #
        # feed flows in [sccm]
        self.feed_flows = np.array(
            [self.feeds[sp] if sp in self.feeds else 0. for sp in self.species_name]
        )
        self.feed_flow_tot = np.sum(self.feed_flows)  # total gas flow in [sccm]

        # masses of collision partners in [kg] (for elastic collisions electron energy losses)
        self.col_partner_masses = np.zeros(self.num_reactions)  # defined for elastic collision only
        for j in range(self.num_reactions):
            if self.mask_elc[j] and self.mask_els[j]:  # only define for elastic electron processes, else 0.
                stoichiovector_lhs = self.stoichiomatrix_lhs[j]
                collision_partners_masses = self.species_mass[stoichiovector_lhs > 0]
                if not len(collision_partners_masses):
                    raise ValueError('The elastic reaction if={} does not have a collision partner')
                preference_collision_partner_mass = collision_partners_masses[0]
                self.col_partner_masses[j] = preference_collision_partner_mass

        # matrix of reduced masses:
        m_i = self.species_mass[:, np.newaxis]
        m_k = self.species_mass[np.newaxis, :]
        self.reduced_mass = m_i * m_k / (m_i + m_k)

        # mean mass of ions (used in the formula for the sheath voltage)
        self.mean_ion_mass = self.species_mass[self.mask_pos].mean() if any(self.mask_pos) else np.nan
        # static part of the sheath voltage formula (sheath voltage per 1eV of electron temperature)
        self.sheath_voltage_factor = np.log((self.mean_ion_mass / (2 * constants.pi * constants.m_e)) ** 0.5)

        # **************************************  DYNAMIC QUANTITIES  ************************************************ #
        # electron energy losses
        self.el_en_loss = np.array(self.reactions_el_en_loss)  # those stay static, the elastic losses change
        # reaction rate coefficients:
        self.k_r = np.empty(self.num_reactions)
        self.k_r[~self.mask_elc] = \
            self.reactions_arrh_a[~self.mask_elc] * (self.temp_g / 300) ** self.reactions_arrh_b[~self.mask_elc] * \
            np.e ** (-self.reactions_arrh_c[~self.mask_elc] / self.temp_g)  # this bit stays static
        # mean velocities
        self.v_m = np.empty(self.num_species)
        self.v_m[self.mask_neu] = \
            (8 * constants.k * self.temp_g / (
                    constants.pi * self.species_mass[self.mask_neu])) ** 0.5  # this stays static
        # scattering cross sections for sp-sp pair:
        # hard-sphere scattering xsecs seed [m2] - only should be defined for n-n and n-i collisions:
        self.sigma_sc = (self.species_lj_sigma[:, np.newaxis] + self.species_lj_sigma[np.newaxis, :]) ** 2
        self.sigma_sc[np.ix_(~self.mask_neu, ~self.mask_neu)] = np.nan  # placeholder for rutherford scattering
        self.sigma_sc[np.diag(len(self.sigma_sc) * [True, ])] = 0.  # diagonal elements are zero - collisions with self
        # are not contributing to diffusion of species

        # ****************************************** CONSISTENCY CHECKS ********************************************** #
        if not set(self.feeds).issubset(set(self.species_name)):
            raise EquationsParametersConsistencyError('Feed species from the model parameters inconsistent with the '
                                                      'chemistry!')
        # check the shapes of all the chemistry getter methods:
        for arr in (self.species_name, self.species_charge, self.species_mass, self.species_stick_coef,
                    self.species_lj_sigma):
            if len(arr) != self.num_species:
                raise EquationsParametersConsistencyError('Inconsistent Chemistry arrays!')
        for arr in (self.reactions_arrh_b, self.reactions_arrh_a, self.reactions_arrh_c, self.reactions_el_en_loss,
                    self.mask_els, self.stoichiovector_electron_lhs, self.stoichiovector_electron_net,
                    self.stoichiovector_arbitrary_lhs):
            if len(arr) != self.num_reactions:
                raise EquationsParametersConsistencyError('Inconsistent Chemistry arrays!')
        for mat in (self.stoichiomatrix_lhs, self.stoichiomatrix_net):
            if mat.shape != (self.num_reactions, self.num_species):
                raise EquationsParametersConsistencyError('Inconsistent Chemistry arrays!')
        if self.return_matrix.shape != (self.num_species, self.num_species):
            raise EquationsParametersConsistencyError('Inconsistent Chemistry arrays!')

    # ***************************************  HELPER FUNCTIONS  ***************************************************** #

    @staticmethod
    def get_density_vector(y):
        """Separates the vector of densities from vector of features y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :return: (np.array) of vector of densities
        """
        return y[:-1]

    @staticmethod
    def get_electron_energy_density(y):
        """Method to separate the electron energy density from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :return: (float) electron density [m-3]
        """
        return y[-1]

    def get_total_density(self, y, n=None):
        """Method to calculate the total pressure from the features vector y. Uses ideal gas equation and gas temp.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :return: (float) total density of the system in [m-3] as sum of all heavy sp densities.
        """
        if n is None:
            n = self.get_density_vector(y)

        n_tot = np.sum(n)
        return n_tot

    def get_total_pressure(self, y, n_tot=None):
        """Method to calculate the total pressure from the features vector y. Uses ideal gas equation and gas temp.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n_tot: (float) total density of heavy species. Optional, if not given, calculated from y.
        :return: (float) total pressure of the system in [Pa] calculated from ideal gas law.
        """
        if n_tot is None:
            n_tot = self.get_total_density(y)

        p = constants.k * self.temp_g * n_tot  # in [Pa] - total instantaneous pressure
        return p

    def get_ion_temperature(self, y, p=None):
        """Method to calculate the ion temperature for the purpose of the diffusivities calculation from the
        features vector y. The ion temperature is not used to evaluate reaction rate coefficient, pressure, nor
        anything else, but is also used as a lower cap for the electron temperature.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param p: (float) total pressure of the system in [Pa] calculated from ideal gas law. Optional, if not given,
                  calculated from y.
        :return: (float) ion temperature [K] for diffusivities calculation.
        """
        if p is None:
            p = self.get_total_pressure(y)

        if p > self.mtorr:
            temp_i = (0.5 * constants.e / constants.k - self.temp_g) / (p / self.mtorr) + self.temp_g
        else:
            temp_i = 0.5 * constants.e / constants.k
        return temp_i  # in [K]

    def get_electron_density(self, y, n=None):
        """Method to calculate the electron density from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :return: (float) electron density [m-3].
        """
        if n is None:
            n = self.get_density_vector(y)

        n_e = np.sum(n * self.species_charge)
        return n_e

    def get_electron_temperature(self, y, n_e=None, rho=None):
        """Method to calculate the electron temperature from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n_e: (float) electron density [m-3]. Optional, if not given, calculated from y.
        :param rho: (float) electron energy density [eV.m-3]. Optional, if not given, calculated from y.
        :return: (float) electron temperature [eV].
        """
        if n_e is None:
            n_e = self.get_electron_density(y)
        if rho is None:
            rho = self.get_electron_energy_density(y)

        temp_e = rho / n_e * 2 / 3
        temp_g_ev = np.float64(self.temp_g * constants.k / constants.e)  # convert to np.float64 so it supports copy()

        # The gas temperature acts as a lower limit for the electron temperature
        temp_e = max(temp_e, temp_g_ev)

        return temp_e

    def get_debye_length(self, y, n_e=None, temp_e=None):
        """Method to calculate the debye length for this plasma model.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n_e: (float) electron density [m-3]. Optional, if not given, calculated from y.
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :return: (float) debye length in [m]
        """
        if n_e is None:
            n_e = self.get_electron_density(y)
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        return (constants.epsilon_0 * temp_e / constants.e / n_e) ** 0.5

    def get_reaction_rate_coefficients(self, y, temp_e=None):
        """Method to calculate the vector of reaction rate coefficients for all the reactions from the features
        vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :return: (np.array) vector of reaction rate coefficients [SI] for all the reactions in the chemistry.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)

        self.k_r[self.mask_elc] = \
            self.reactions_arrh_a[self.mask_elc] * temp_e ** self.reactions_arrh_b[self.mask_elc] * \
            np.e ** (-self.reactions_arrh_c[self.mask_elc] / temp_e)
        return self.k_r

    def get_reaction_rates(self, y, n=None, n_e=None, n_tot=None, k_r=None):
        """Method to calculate the vector of reaction rates for all the reactions from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :param n_e: (float) electron density [m-3]. Optional, if not given, calculated from y.
        :param n_tot: (float) total density of heavy species. Optional, if not given, calculated from y.
        :param k_r: (np.array) vector of reaction rate coefficients [SI] for all the reactions in the chemistry.
                    Optional, if not given, calculated from y.
        :return: (np.array) vector of reaction rates [m-3/s] for all the reactions in the chemistry.
                 Optional, if not given, calculated from y.
        """
        if n is None:
            n = self.get_density_vector(y)
        if n_e is None:
            n_e = self.get_electron_density(y, n=n)
        if n_tot is None:
            n_tot = self.get_total_density(y, n=n)
        if k_r is None:
            k_r = self.get_reaction_rate_coefficients(y)

        # density vector with the electron density (1st) and 'M' density (last)
        n_reactants = np.r_[n_e, n, n_tot]
        # product of all reactants densities for all reactions:
        n_reactants_prod = np.prod(n_reactants[np.newaxis, :] ** self.stoichiomatrix_reactants_lhs, axis=1)
        rates = k_r * n_reactants_prod  # reaction rate for each reaction
        return rates

    def get_volumetric_source_rates(self, y, rates=None):
        """Method to calculate the contributions to the densities time derivatives due to the volumetric species
        gains and losses from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param rates: (np.array) vector of reaction rates [m-3/s] for all the reactions in the chemistry.
                      Optional, if not given, calculated from y.
        :return: (np.array) vector of densities time derivatives due to the volumetric gains and losses. Elements
                 for all heavy species.
        """
        if rates is None:
            rates = self.get_reaction_rates(y)

        # volumetric reactions production rates for all species [m-3/s]
        source_vol = np.sum(self.stoichiomatrix_net * rates[:, np.newaxis], axis=0)
        return source_vol

    def get_flow_source_rates(self, y, n=None, p=None):
        """Method to calculate the contributions to the densities time derivatives due to the gas flows from the
        features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :param p: (float) total pressure of the system in [Pa] calculated from ideal gas law. Optional, if not given,
                  calculated from y.
        :return: (np.array) vector of densities time derivatives due to the flows (in and out). Elements for all
                 heavy species.
        """
        if n is None:
            n = self.get_density_vector(y)
        if p is None:
            p = self.get_total_pressure(y)

        t_flow = 1e-3  # pressure recovery time in [s]
        # initialise the flow source/sink rates:
        source_flow = np.zeros(self.num_species)  # sources/sinks due to flow - in/out
        # loss rate due to the pressure regulation - only acting upon neutral species:
        source_flow[self.mask_neu] -= (n[self.mask_neu] * (1 / t_flow) * (p - self.p0) / self.p0)
        # gain rate due to the inflow:
        source_flow += self.feed_flows * 4.485E+17 / self.volume  # converting from sccm -> particles/sec -> m-3/sec
        # loss rate due to outflow (sum is the same as the sum of inflow, distributed amongst neutral species)
        source_flow[self.mask_neu] -= \
            sum(self.feed_flows * 4.485E+17 / self.volume) * n[self.mask_neu] / sum(n[self.mask_neu])

        return source_flow

    def get_mean_speeds(self, y, temp_i=None):
        """Method to calculate mean thermal speeds for all the species from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_i: (float) ion temperature [K] for diffusivities calculation. Optional, if not given,
                       calculated from y.
        :return: (np.array) of mean thermal speeds for heavy species [m/s].
        """
        if temp_i is None:
            temp_i = self.get_ion_temperature(y)

        # mean speeds (therm.) - values for the neutrals are static, only need to update values for +/- ions
        self.v_m[~self.mask_neu] = (8 * constants.k * temp_i / (
                constants.pi * self.species_mass[~self.mask_neu])) ** 0.5
        return self.v_m

    def get_sigma_sc(self, y, v_m=None, debye_length=None):
        """Method to populate the dynamic values of the matrix for sp-2-sp momentum transfer scattering cross sections
        (only the ion-ion pairs are dynamically calculated, rest are static.)

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param v_m: (np.array) of mean thermal speeds for heavy species [m/s]. Optional, if not given, calculated
                    from y.
        :param debye_length: (float) Debye length in the plasma. Optional, if not given, calculated from y.
        :return: (np.array 2D) matrix of cross sections for momentum transfer between each pair species
        """
        if v_m is None:
            v_m = self.get_mean_speeds(y)
        if debye_length is None:
            debye_length = self.get_debye_length(y)

        # classical distance of closest approach matrix for each ion-ion pair:
        b_0_ii = \
            constants.e ** 2 * \
            abs(self.species_charge[~self.mask_neu, np.newaxis] * self.species_charge[np.newaxis, ~self.mask_neu]) / \
            (2 * constants.pi * constants.epsilon_0) / \
            (self.reduced_mass[np.ix_(~self.mask_neu, ~self.mask_neu)] * v_m[~self.mask_neu, np.newaxis] ** 2)
        # populate the dynamic part of the self.sigma_sc matrix: only values for ion-ion pairs:
        self.sigma_sc[np.ix_(~self.mask_neu, ~self.mask_neu)] = \
            constants.pi * b_0_ii ** 2 * np.log(2 * debye_length / b_0_ii)
        self.sigma_sc[np.diag(len(self.sigma_sc) * [True, ])] = 0.  # diagonal elements are zero - collisions with self
        # are not contributing to diffusion of species
        return self.sigma_sc

    def get_mean_free_paths(self, y, n=None, sigma_sc=None):
        """Method to calculate the vector of mean free paths for all the species from the features vector y.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :param sigma_sc: (np.array 2D) matrix of heavy species momentum transfer cross sections. Optional, if not
                         given, calculated from y.
        :return: (np.array) of mean free paths for heavy species [m].
        """
        if n is None:
            n = self.get_density_vector(y)
        if sigma_sc is None:
            sigma_sc = self.get_sigma_sc(y)

        mfp = 1 / np.sum(n[np.newaxis, :] * sigma_sc, axis=1)
        return mfp

    def get_free_diffusivities(self, y, mfp=None, v_m=None):
        """Method to calculate the free diffusion coefficients for all the heavy species in the model.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param mfp: (np.array) of mean free paths for heavy species [m]. Optional, if not given, calculated from y.
        :param v_m: (np.array) of mean thermal speeds for heavy species [m/s]. Optional, if not given, calculated
                    from y.
        :return: (np.array) of coefficients of free diffusion for species [SI].
        """
        if mfp is None:
            mfp = self.get_mean_free_paths(y)
        if v_m is None:
            v_m = self.get_mean_speeds(y)

        diff_c_free = constants.pi / 8 * mfp * v_m
        return diff_c_free

    def get_ambipolar_diffusivity_pos(self, y, n=None, n_e=None, temp_i=None, temp_e=None, diff_c_free=None):
        """Method to calculate the coefficient of ambipolar diffusion for positive ions.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :param n_e: (float) electron density [m-3]. Optional, if not given, calculated from y.
        :param temp_i: (float) ion temperature [K] for diffusivities calculation. Optional, if not given,
                       calculated from y.
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :param diff_c_free: (np.array) coefficients of free diffusion [SI]. Optional, if not given, calculated from y.
        :return: (float) coefficient of ambipolar diffusion for positive ions.
        """
        if n is None:
            n = self.get_density_vector(y)
        if n_e is None:
            n_e = self.get_electron_density(y, n=n)
        if temp_i is None:
            temp_i = self.get_ion_temperature(y)
        if temp_e is None:
            temp_e = self.get_electron_temperature(y, n_e=n_e)
        if diff_c_free is None:
            diff_c_free = self.get_free_diffusivities(y)

        gamma = temp_e * constants.e / constants.k / temp_i
        alpha = n[self.mask_neg].sum() / n_e
        diff_free_pos = diff_c_free[self.mask_pos].mean()  # mean coefficient for positive ions free diffusion
        # NOTE: this only holds for alpha << mu_e/mu_i
        diff_a_pos = diff_free_pos * (1 + gamma * (1 + 2 * alpha)) / (
                1 + alpha * gamma)  # ambipolar diffusion coef for +ions
        return diff_a_pos

    # noinspection PyMethodMayBeStatic
    def get_ambipolar_diffusivity_neg(self, y):
        """Method to calculate the coefficient of ambipolar diffusion for negative ions.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :return: (float) in [SI]
        """
        # Global_Kin apparently calculates this as D^-_free * exp(-V_sh/(k*T_i)), though this is perhaps not needed,
        # since the factor will be about 1e-100 for 300K and about 1e-10 for 3000K
        _ = y  # this is not used here currently
        diff_a_neg = 0  # so far not really counting this, zero should be valid for alpha << mu_e/mu_i
        return diff_a_neg

    def get_diffusivities(self, y, diff_c_free=None, diff_a_pos=None, diff_a_neg=None):
        """Method to calculate the diffusion coefficients from the features vector y. The neutrals diffusivities
        are free diffusivities for neutrals (given by the mixture rules using LJ potentials) and the
        ion diffusivities are equal to ambipolar diffusion coefficients for positive and negative ions.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param diff_c_free: (np.array) coefficients of free diffusion [SI]. Optional, if not given, calculated from y.
        :param diff_a_pos: (float) coefficient of ambipolar diffusion for positive ions [SI]
        :param diff_a_neg: (float) coefficient of ambipolar diffusion for negative ions [SI]
        :return: (np.array) of diffusion coefficients for all species [SI].
        """
        if diff_c_free is None:
            diff_c_free = self.get_free_diffusivities(y)
        if diff_a_pos is None:
            diff_a_pos = self.get_ambipolar_diffusivity_pos(y, diff_c_free=diff_c_free)
        if diff_a_neg is None:
            diff_a_neg = self.get_ambipolar_diffusivity_neg(y)

        diff_c = np.empty(self.num_species)
        diff_c[self.mask_neu] = diff_c_free[self.mask_neu]  # diffusion coefficients for neutrals
        # populate diffusivities for ions with their ambipolar diffusion coefficients
        diff_c[self.mask_pos] = diff_a_pos
        diff_c[self.mask_neg] = diff_a_neg
        return diff_c

    def get_wall_fluxes(self, y, n=None, diff_c=None, v_m=None):
        """Method to calculate the vector of fluxes of particles sticking to the walls for each species.
        The fluxes therefore already take into account the sticking coefficients - if sticking coefficients are null for
        certain species, fluxes for those will be null also.
        This method ONLY takes into account STICKING fluxes. If any species is getting stuck to the surface, it will
        have a negative wall flux returned by this method. But the same species might have return coefficient defined
        as 1.0 with the return species of itself, which will mean that the rate of density change due to the surface
        interactions for this species will still be null. This is simply how the WALL FLUXES are defined in this work.
        The 'in-fluxes' of returned species are not at all taken into account by this method!
        All the fluxes are negative by convention (particles "moving out of the system").
        The fluxes depend on the diffusion model (class attribute) - 0: wall flux is a pure diffusive flux (Lietz2016)
                                                                     1: wall flux is a combination of diffusive
                                                                        and thermal flux (Schroter2018)

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :param diff_c: (np.array) of diffusion coefficients for all species [SI].
                       Optional, if not given, calculated from y.
        :param v_m: (np.array) of mean thermal speeds for heavy species [m/s]. Optional, if not given, calculated
                    from y.
        :return: (np.array) vector of particle fluxes in [m-2.s-1].
        """
        if n is None:
            n = self.get_density_vector(y)
        if diff_c is None:
            diff_c = self.get_diffusivities(y)
        if v_m is None:
            v_m = self.get_mean_speeds(y)

        s = self.species_stick_coef
        if self.diffusion_model == 0:
            return -diff_c * n * s / self.diff_l ** 2 * self.volume / self.area
        elif self.diffusion_model == 1:
            return -diff_c * n * s / (s * self.diff_l + (4 * diff_c / v_m))
        else:
            raise ValueError('Unsupported diffusion model!')

    def get_diffusion_sinks(self, y, wall_fluxes=None):
        """Method to calculate a vector of densities time derivatives contributions due to diffusion sinks to the walls
        from the features vector y. Only sinks, no sources.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param wall_fluxes: (np.array) vector of wall fluxes in [m-2/s].
        :return: (np.array) vector of density time derivatives due to diffusion losses for all heavy species.
        """
        if wall_fluxes is None:
            wall_fluxes = self.get_wall_fluxes(y)

        source_diff_sinks = wall_fluxes * self.area / self.volume
        return source_diff_sinks

    def get_diffusion_sources(self, y, diff_sinks=None):
        """Method to calculate a vector of densities time derivatives contributions due to diffusion sources from the
        walls, that is return species re-injection from the walls to the plasma.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param diff_sinks: (np.array) vector of density time derivatives due to diffusion losses for all heavy species.
                           Optional, if not given, calculated from y.
        :return: (np.array) vector of density time derivatives due to returned species from walls.
                 Elements for all heavy species.
        """
        if diff_sinks is None:
            diff_sinks = self.get_diffusion_sinks(y)

        return np.sum(-diff_sinks[np.newaxis, :] * self.return_matrix, axis=1)

    def get_diffusion_source_rates(self, y, diff_sinks=None, diff_sources=None):
        """Method to calculate a vector of densities time derivatives contributions due to diffusion sinks and
        return sources.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param diff_sinks: (np.array) vector of density time derivatives due to diffusion losses for all heavy species.
                           Optional, if not given, calculated from y.
        :param diff_sources: (np.array) vector of density time derivatives due to returned species from walls.
                             Elements for all heavy species. Optional, if not given, calculated from y.
        :return: (np.array) vector of densities time derivatives due to the diffusion losses and
                 returned fluxes from the walls. Elements for all heavy species.
        """
        if diff_sinks is None:
            diff_sinks = self.get_diffusion_sinks(y)
        if diff_sources is None:
            diff_sources = self.get_diffusion_sources(y, diff_sinks=diff_sinks)

        return diff_sinks + diff_sources

    def get_min_n_correction(self, y, n=None):
        """This is an artificial (unphysical) correction applied to the RHS of the densities ODE system preventing
        the densities to go under a minimal value (and ultimately from reaching unphysical negative densities.) It
        supplies a nudge for all the densities below a lower limit which is proportional to the difference of the
        densities and the limit.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param n: (np.array) vector of densities of heavy species. Optional, if not given, calculated from y.
        :return: (np.array) vector of corrections preventing the densities reaching values lower than n_min.
        """
        if n is None:
            n = self.get_density_vector(y)

        n_min = 1.e0
        t_rec = 1.e-10  # recovery time - should be approx the solver time step time
        below_min_mask = n < n_min
        min_n_correction = np.zeros(len(n))
        min_n_correction[below_min_mask] = (n_min - n[below_min_mask]) / t_rec

        return min_n_correction

    def get_dn_dt(self, y, vol_source_rates=None, flow_source_rates=None, diff_source_rates=None,
                  min_n_correction=None):
        """Method to calculate the vector of densities time derivatives for all the heavy species.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param vol_source_rates: (np.array) vector of densities time derivatives due to the volumetric gains and
                                 losses. Elements for all heavy species. Optional, if not given, calculated from y.
        :param flow_source_rates: (np.array) vector of densities time derivatives due to the flows (in and out).
                                  Elements for all heavy species. Optional, if not given, calculated from y.
        :param diff_source_rates: (np.array) vector of densities time derivatives due to the diffusion losses and
                                  returned fluxes from the walls. Elements for all heavy species. Optional, if not
                                  given, calculated from y.
        :param min_n_correction: (np.array) vector of correction factors nudging the solver away from two low densities
                                 and preventing reaching negative densities.
        :return: (np.array) vector of time derivatives of densities of all heavy species.
        """
        if vol_source_rates is None:
            vol_source_rates = self.get_volumetric_source_rates(y)
        if flow_source_rates is None:
            flow_source_rates = self.get_flow_source_rates(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)
        if min_n_correction is None:
            min_n_correction = self.get_min_n_correction(y)

        return vol_source_rates + flow_source_rates + diff_source_rates + min_n_correction

    def get_power_ext(self, t):
        """Method returning the instant absorbed power P(t) in the time t.
        This unlike other methods is only dependent on time.

        :param t: (float )time in [sec].
        :return: (float) Instant absorbed power P(t) in [W]
        """
        return np.interp(t, self.t_pow, self.pow)

    def get_drho_dt_ext(self, t, power_ext=None):
        """Method to calculate the contribution of absorbed power to the time derivative of electron energy density.
        This unlike other methods is only dependent on time.

        :param t: (float) time in [sec].
        :param power_ext: (float) instant absorbed power in [W]. Optional, if not passed, calculated from t
        :return: (float) time derivative of the energy density due to the absorbed power contribution
        """
        if power_ext is None:
            power_ext = self.get_power_ext(t=t)
        return power_ext / self.volume / constants.e

    def get_el_en_losses(self, y, temp_e=None):
        """Method to calculate the vector of electron energy losses per a reaction for all reactions. Only non-zero
        values (and only used) for electron collisions elastic and inelastic, with at least one electron on the LHS.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :return: (np.array) vector of electron energy losses per reaction for all reactions. Only relevant
                 for elastic and inelastic electron collisions with electrons also on LHS.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)

        mask = self.mask_elc & self.mask_els
        self.el_en_loss[mask] = \
            3 * constants.m_e / self.col_partner_masses[mask] * (temp_e - self.temp_g * constants.k / constants.e)
        return self.el_en_loss

    def get_drho_dt_el_inel(self, y, el_en_losses=None, reaction_rates=None):
        """Method to calculate the contribution of elastic and inel collisions to the time derivative of the
        electron energy density.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param el_en_losses: (np.array) vector of electron energy losses per reaction for all reactions. Only relevant
                             for elastic and inelastic electron collisions with electrons also on LHS.
        :param reaction_rates: (np.array) vector of reaction rates [m-3/s] for all the reactions in the chemistry.
                               Optional, if not given, calculated from y.
        :return: (float) time derivative of the energy density due to the elastic and inelastic
                 collisions.
        """
        if el_en_losses is None:
            el_en_losses = self.get_el_en_losses(y)
        if reaction_rates is None:
            reaction_rates = self.get_reaction_rates(y)

        return np.sum((el_en_losses[self.mask_elc] * reaction_rates[self.mask_elc]))

    def get_drho_dt_gain_loss(self, y, temp_e=None, reaction_rates=None):
        """Method to calculate the contribution of volumetric creation and destruction of electrons to the time
        derivative of the electron energy density.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :param reaction_rates: (np.array) vector of reaction rates [m-3/s] for all the reactions in the chemistry.
                               Optional, if not given, calculated from y.
        :return: (float) time derivative of electron energy density due to volumetric losses and gains of
                 electrons.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if reaction_rates is None:
            reaction_rates = self.get_reaction_rates(y)

        return 3 / 2 * temp_e * np.sum(self.stoichiovector_electron_net * reaction_rates)

    def get_drho_dt_el_walls(self, y, temp_e=None, diff_source_rates=None):
        """Method to calculate the contribution of the electrons loss to the walls to the time derivative of
        the electron energy density.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :param diff_source_rates: (np.array) vector of densities time derivatives due to the diffusion losses and
                                  returned fluxes from the walls. Elements for all heavy species. Optional, if not
                                  given, calculated from y.
        :return: (float) time derivative of electron energy density due to losses of electrons to the walls.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)

        tot_el_outflux = np.sum(self.species_charge * diff_source_rates)
        return -2 * temp_e * tot_el_outflux

    def get_sheath_voltage(self, y, temp_e=None):
        """Method to estimate the sheath voltage of the modeled plasma.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :return: (float) sheath voltage in [V].
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)

        return temp_e * self.sheath_voltage_factor

    def get_drho_dt_ions_walls(self, y, temp_e=None, sh_pot=None, diff_source_rates=None):
        """Method to calculate the contribution of the loss of ions to the walls to the time derivative of
        the electron energy density.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param temp_e: (float) electron temperature [eV]. Optional, if not given, calculated from y.
        :param sh_pot: (float) sheath voltage in [V]. Optional, if not given, calculated from y.
        :param diff_source_rates: (np.array) vector of densities time derivatives due to the diffusion losses and
                                  returned fluxes from the walls. Elements for all heavy species. Optional, if not
                                  given, calculated from y.
        :return: (float) time derivative of electron energy density due to losses of ions to the
                 walls.
        """
        if temp_e is None:
            temp_e = self.get_electron_temperature(y)
        if sh_pot is None:
            sh_pot = self.get_sheath_voltage(y)
        if diff_source_rates is None:
            diff_source_rates = self.get_diffusion_source_rates(y)

        return - 0.5 * temp_e * np.sum(diff_source_rates[self.mask_pos]) \
               - sh_pot * np.sum(self.species_charge[self.mask_pos] * diff_source_rates[self.mask_pos])

    def get_min_rho_correction(self, y, rho=None):
        """This is an artificial (unphysical) correction applied to the RHS of the el. en. dens. ODE, preventing
        the rho to go under a minimal value (and ultimately from reaching unphysical negative value.) It
        supplies a nudge if rho is below a lower limit which is proportional to the difference of the rho and the limit.

        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param rho: (float) electron energy density. Optional, if not given, calculated from y.
        :return: (float) correction preventing the el. en. density from reaching values lower than rho_min.
        """
        if rho is None:
            rho = self.get_electron_energy_density(y)

        rho_min = 1.e0
        t_rec = 1.e-10  # recovery time scale - approx solver step time
        min_rho_correction = (rho_min - rho) / t_rec if rho < rho_min else 0.

        return np.float64(min_rho_correction)  # convert explicitly to np.float so it supports copy as others do...

    def get_drho_dt(self, t, y, ext=None, el_inel=None, gain_loss=None, el_walls=None, ions_walls=None,
                    min_rho_correction=None):
        """Method to calculate the time derivative of the electron energy density.

        :param t: (float) time in [sec].
        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param ext: (float) time derivative of the energy density due to the external absorbed power.
                    Optional, if not given, calculated from t.
        :param el_inel: (float) time derivative of the energy density due to the elastic and inelastic
                        collisions. Optional, if not given, calculated for y.
        :param gain_loss: (float) time derivative of electron energy density due to volumetric losses
                          and gains of electrons.  Optional, if not given, calculated for y.
        :param el_walls: (float) time derivative of electron energy density due to losses of electrons
                         to the walls. Optional, if not given, calculated for y.
        :param ions_walls: (float) time derivative of electron energy density due to losses of ions to the
                           walls. Optional, if not given, calculated for y.
        :param min_rho_correction: (float) correction factor nudging the solver away from two low electron energy
                                   density and preventing reaching a negative value.
        :return: (float) time derivative of electron energy density.
        """
        if ext is None:
            ext = self.get_drho_dt_ext(t)
        if el_inel is None:
            el_inel = self.get_drho_dt_el_inel(y)
        if gain_loss is None:
            gain_loss = self.get_drho_dt_gain_loss(y)
        if el_walls is None:
            el_walls = self.get_drho_dt_el_walls(y)
        if ions_walls is None:
            ions_walls = self.get_drho_dt_ions_walls(y)
        if min_rho_correction is None:
            min_rho_correction = self.get_min_rho_correction(y)

        return ext - el_inel - gain_loss - el_walls - ions_walls + min_rho_correction

    def get_dy_dt(self, t, y, dn_dt=None, drho_dt=None):
        """Method to calculate the vector of LHSs of the system of ODS which is being solved.

        :param t: (float) time in [sec].
        :param y: (np.array) vector of features (n0, ..., nN, rho) of densities and electron energy density in
                  [m-3] and [m-3.eV].
        :param dn_dt: (np.array) vector of time derivatives of densities of all heavy species. Optional, if not given,
                      calculated for y.
        :param drho_dt: (float) time derivative of electron energy density. Optional, if not given, calculated from y.
        :return: (np.array) vector of the right-hand-sides of all the ODEs in the system being solved.
        """
        if dn_dt is None:
            dn_dt = self.get_dn_dt(y)
        if drho_dt is None:
            drho_dt = self.get_drho_dt(t, y)

        return np.r_[dn_dt, drho_dt]

    # ***************************************  OBJECTIVE FUNCTIONS  ************************************************** #

    def get_objective_functions(self, jacobian=False):
        """Method to calculate the objective (master) function for a ODE solver and it's jacobian.

        :param jacobian: (bool) if to calculate the jacobian or not.
        :return: (func, func) - if jacobian == True, tuple of functions taking time and features vectors and
                 returning first LHSs of the ODE system and second Jacobian matrix of the ODE system. If the
                 jacobian == False, (func, None) is returned only with objective function.
        """

        obj_function_jacobian = None

        # noinspection DuplicatedCode
        def obj_function(t, y):
            n = self.get_density_vector(y)
            rho = self.get_electron_energy_density(y)
            n_tot = self.get_total_density(y, n=n)
            p = self.get_total_pressure(y, n_tot=n_tot)
            temp_i = self.get_ion_temperature(y, p=p)
            n_e = self.get_electron_density(y, n=n)
            temp_e = self.get_electron_temperature(y, n_e=n_e, rho=rho)
            debye_length = self.get_debye_length(y, n_e=n_e, temp_e=temp_e)
            k = self.get_reaction_rate_coefficients(y, temp_e=temp_e)
            rates = self.get_reaction_rates(y, n=n, n_e=n_e, n_tot=n_tot, k_r=k)
            source_vol = self.get_volumetric_source_rates(y, rates=rates)
            source_flow = self.get_flow_source_rates(y, n=n, p=p)
            v_m = self.get_mean_speeds(y, temp_i=temp_i)
            sigma_sc = self.get_sigma_sc(y, v_m=v_m, debye_length=debye_length)
            mfp = self.get_mean_free_paths(y, n=n, sigma_sc=sigma_sc)
            diff_c_free = self.get_free_diffusivities(y, mfp=mfp, v_m=v_m)
            diff_a_pos = self.get_ambipolar_diffusivity_pos(y, n=n, n_e=n_e, temp_i=temp_i, temp_e=temp_e,
                                                            diff_c_free=diff_c_free)
            diff_a_neg = self.get_ambipolar_diffusivity_neg(y)
            diff = self.get_diffusivities(y, diff_c_free=diff_c_free, diff_a_pos=diff_a_pos, diff_a_neg=diff_a_neg)
            wall_fluxes = self.get_wall_fluxes(y, n=n, diff_c=diff, v_m=v_m)
            source_diff_sinks = self.get_diffusion_sinks(y, wall_fluxes=wall_fluxes)
            source_diff_sources = self.get_diffusion_sources(y, diff_sinks=source_diff_sinks)
            source_diff = self.get_diffusion_source_rates(y, diff_sinks=source_diff_sinks,
                                                          diff_sources=source_diff_sources)
            min_n_cor = self.get_min_n_correction(y, n=n)
            dn_dt = self.get_dn_dt(y, vol_source_rates=source_vol, flow_source_rates=source_flow,
                                   diff_source_rates=source_diff, min_n_correction=min_n_cor)

            power_ext = self.get_power_ext(t)
            drho_dt_ext = self.get_drho_dt_ext(t, power_ext=power_ext)
            el_en_losses = self.get_el_en_losses(y, temp_e=temp_e)
            drho_dt_el_inel = self.get_drho_dt_el_inel(y, el_en_losses=el_en_losses, reaction_rates=rates)
            drho_dt_gain_loss = self.get_drho_dt_gain_loss(y, temp_e=temp_e, reaction_rates=rates)
            drho_dt_el_walls = self.get_drho_dt_el_walls(y, temp_e=temp_e, diff_source_rates=source_diff)
            sh_pot = self.get_sheath_voltage(y, temp_e=temp_e)
            drho_dt_ions_walls = self.get_drho_dt_ions_walls(y, temp_e=temp_e, sh_pot=sh_pot,
                                                             diff_source_rates=source_diff)
            min_rho_correction = self.get_min_rho_correction(y, rho=rho)
            drho_dt = self.get_drho_dt(
                t, y, ext=drho_dt_ext, el_inel=drho_dt_el_inel, gain_loss=drho_dt_gain_loss, el_walls=drho_dt_el_walls,
                ions_walls=drho_dt_ions_walls, min_rho_correction=min_rho_correction)
            dy_dt = self.get_dy_dt(t, y, dn_dt=dn_dt, drho_dt=drho_dt)

            return dy_dt

        if jacobian:
            raise NotImplementedError('Jacobian function not implemented!')

        return obj_function, obj_function_jacobian
