import copy
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
import yaml

from plaschem.classification import ReactionClassifier
from plaschem.exceptions import \
    ChemistryConsistencyError, ReactionAttributeError, RPAttributeError, ChemistrySpeciesNotPresentError, \
    ChemistryReactionNotPresentError, ChemistryDisableError, ChemistryInitError, ChemistryAttributeError
from plaschem.pretty_print import ColorValue
from plaschem.reactions import Reaction
from plaschem.species import RP
from plaschem.xml_inputs import XmlParser


# noinspection DuplicatedCode
class Chemistry(object):
    """
    Class representing a plasma chemistry. Consists of a load of RP and Reaction instances and defines several
    useful methods on them, such as compiling stoichiometric matrices, rates matrices, various parameters printouts
    as well as a framework for self-consistent disabling certain species and reactions.
    """
    requires_charge = True

    def __init__(self, species=None, reactions=None, xml_path=None):
        """Initiates the Chemistry instance with species and reactions.

        :param species: (iterable) of RP instances. Has to include also the special species 'e' and 'M'
        :param reactions: (iterable) of Reaction instances. All the reactions need to point to the same RP instances
                          as passed in species as reactants and products. This is asserted in the code.
        :param xml_path: (str) optional path to a xml input file - if species and reactions are None (not supplied),
                         then they will be created from xml by plaschem.xml.XmlLoader. Even if species and reactions
                         are supplied, xml_path can be passed if those were created from xml elsewhere.
        """
        # check consistency of the inputs:
        legal = (
                (species is not None and reactions is not None) or
                (species is None and reactions is None and xml_path is not None)
        )
        if not legal:
            raise ChemistryInitError('Unsupported combination of attributes and values!')

        # build reactions and species arrays if not passed:
        if reactions is None:
            species, reactions = XmlParser().get_species_and_reactions(xml_path=xml_path)

        # base consistency check: IDs need to be unique:
        sp_ids = [sp.id for sp in species]
        r_ids = [r.id for r in reactions]
        if len(sp_ids) != len(set(sp_ids)) or len(r_ids) != len(set(r_ids)):
            raise ChemistryConsistencyError('The ids of species and reactions need to be unique!!')

        # store the xml path if used
        self.xml_path = xml_path

        self._species = pd.Series(species, index=[sp.id for sp in species])  # also including 'e' and 'M' and all ign.
        self._reactions = pd.Series(reactions, index=[r.id for r in reactions], name='Reactions')  # including all ign.

        self._special_sp = pd.Series([sp.is_special() for sp in species], index=self._species.index)

        # species masks:
        self._enabled_sp = pd.Series(True, index=self._species.index)
        self._protected_sp = pd.Series(False, index=self._species.index)  # these are explicitly set protected species
        self._adhoc_protected_sp = pd.Series(False, index=self._species.index)  # gets set eg if sec. to last ion is out
        # adhoc protected species are protected as a result disabling second-to-last positive ion...

        # reactions masks:
        self._enabled_r = pd.Series(True, index=self._reactions.index)

        # AD-HOC CHANGES TO THE SPECIES/REACTIONS KINETIC DATA
        # only attributes defined by keys in the following dicts can be adhoc-changed!
        self.adhoc_species_attributes = {
            'stick_coef': pd.Series(dtype='O'),
            'ret_coefs': pd.Series(dtype='O'),
        }
        self.adhoc_reactions_attributes = {
            'arrh_a': pd.Series(dtype='O'),
        }

        self.assert_consistency()

        if self.requires_charge:
            # automatically set electron as protected:
            self.set_protected_species('e')
            # if there is only one +ion, protect it as well!
            positive_ions = self.get_species()[(self.get_species_charge().values > 0)]
            if len(positive_ions) == 1:
                self.set_protected_species(positive_ions.iat[0])

    def copy(self):
        """Return a copy of the chemistry. It's not a very deep copy, meaning that both copy and original chemistry
        will still be sharing species and reactions objects.
        :return: (self instance)
        """
        self_copy = Chemistry(species=self._species.values, reactions=self._reactions.values, xml_path=self.xml_path)

        # ENFORCE IDENTICAL STATE:
        # masks:
        self_copy._enabled_sp = self._enabled_sp.copy()
        self_copy._protected_sp = self._protected_sp.copy()
        self_copy._adhoc_protected_sp = self._adhoc_protected_sp.copy()
        self_copy._enabled_r = self._enabled_r.copy()

        # ad-hoc attributes:
        for key in self.adhoc_species_attributes:
            self_copy.adhoc_species_attributes[key] = self.adhoc_species_attributes[key].copy()
            if key == 'ret_coefs':  # need to manually create copies of the dicts, since these were not deeply made
                for i in self_copy.adhoc_species_attributes[key].index:  # i is index of the stuck sp with return coefs
                    # copies of the dicts describing the return species and coefficients
                    self_copy.adhoc_species_attributes[key].at[i] = \
                        copy.deepcopy(self_copy.adhoc_species_attributes[key].at[i])
        for key in self.adhoc_reactions_attributes:
            self_copy.adhoc_reactions_attributes[key] = self.adhoc_reactions_attributes[key].copy()

        return self_copy

    def assert_consistency(self):
        """This function checks the consistency of species and reactions and raises exceptions if the
        species set is not consistent with reaction set

        :return: None
        """
        sp_names = self.get_species_name(special=True)

        if self.requires_charge:
            # check if 'e' in species and if at least one +ion in species.
            if 'e' not in list(sp_names.values):
                raise ChemistryConsistencyError('Chemistry needs the electron species!')
            species_charge = self.get_species_charge()
            if not (species_charge.values > 0).any():
                raise ChemistryConsistencyError('Chemistry needs at least one positive ion!')

        # check for duplicates:
        sp_ids = sp_names.index
        if len(sp_names) != len(set(sp_names)):
            raise ChemistryConsistencyError('Species with duplicate names present!')
        if len(sp_ids) != len(set(sp_ids)):
            raise ChemistryConsistencyError('Species with duplicate ids present!')

        # check if the species passed and species involved in reactions are the same instances:
        species = self.get_species(special=True)
        confirmed = set([])
        for r in self.get_reactions():
            for rp_set in r.get_reactants(), r.get_products():
                for rp in rp_set:
                    if any([rp is sp for sp in species]):
                        confirmed.add(rp.get_name())
                    else:
                        raise ChemistryConsistencyError(
                            '{} from {} was not passed to the chemistry in "species"'.format(rp, r)
                        )
        if len(species) != len(confirmed):
            raise ChemistryConsistencyError(
                'Species {} specified in the chemistry is not in any of the reactions!'.format(
                    (list(set([sp.get_name() for sp in species]) - confirmed))[0]
                )
            )

        # check if all the species names in return coefficients dicts are species present in the chemistry
        ret_sp_names = set()
        for ret_coefs in self.get_species_ret_coefs(special=True):
            if ret_coefs is not None:
                for sp_name in ret_coefs:
                    ret_sp_names.add(sp_name)
        if not ret_sp_names.issubset(set(sp_names)):
            raise ChemistryConsistencyError(
                'Some of the return species are not present in the chemistry: {}!'.format(ret_sp_names - set(sp_names))
            )

    def get_rp(self, rp):
        """Returns an RP instance from the chemistry. Raises a ValueError if the species is not found in the chemistry.
        Does not reflect filtering with disabled species or special species, will return RP even if it's set disabled.
        :param rp: (int or str or RP instance) species id, name or instance itself.
        :return: RP instance from the chemistry
        """
        try:  # is rp species id?
            if type(rp) == int:
                return self._species.at[rp]
            else:
                raise KeyError
        except KeyError:
            for sp in self._species:
                if sp.get_name() == rp or sp is rp:  # is rp species name or RP instance?
                    return sp
            raise ChemistrySpeciesNotPresentError('Unrecognised species identifier passed: {}!'.format(rp))

    def get_reaction(self, reaction):
        """Returns an Reaction instance from the chemistry. Raises a ValueError if the reaction is not found in the
        chemistry. Does not reflect filtering with disabled reactions, will return Reaction even if it's set disabled.
        This method is here only for consistency with it's RP counterpart, which can be identified with id or name.
        Reactions are to date only identified by id, so it's essentially wrapper around pd.Series.at[]...
        :param reaction: (int) reaction id.
        :return: Reaction instance from the chemistry
        """
        try:
            return self._reactions.at[reaction]
        except KeyError:
            raise ChemistryReactionNotPresentError('Unrecognised reaction identifier passed: {}!'.format(reaction))

    def num_species(self, disabled=False, special=False, protected=True):
        """Returns list of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (int) number of species in the chemistry which meet the disabled/special/protected criteria
        """
        return len(self.get_species(disabled=disabled, special=special, protected=protected))

    def num_reactions(self):
        return len(self.get_reactions())

    # *********************************** DISABLING/ENABLING SPECIES/REACTIONS *************************************** #

    def set_protected_species(self, *args):
        """Sets protected species. Protected species are the ones which cannot be disabled from the chemistry (attempt
        to do so will rise a ChemistryConsistencyError. Examples of protected species might be for example feed gases
        or electron and the last remaining positive ion.

        :param args: (str or RP or int) for species names, species ids or species. Allows mix of all
        :return: None
        """
        args = list(args)
        try:  # are all args species ids?
            self._protected_sp[args] = True
        except (ValueError, KeyError):  # different versions of pandas package raise different errors it seems.
            # are args species names or species instances or all mixed??
            protected_ids = [self.get_rp(arg).id for arg in args]
            self.set_protected_species(*protected_ids)

    def _disable_recursively(self, species_ids=(), reactions_ids=()):
        """Method follows the "disabling cascade" all the way to the bottom, ensuring that the chemistry left
        is the largest possible one with passed reactions and species disabled and still consistent (the network
        connected). If it hits a protected species, raises the ChemistryDisableError.
        :param species_ids: (tuple of int)
        :param reactions_ids: (tuple of int)
        :return: None
        """
        dis_species_ids = [sp_id for sp_id in species_ids if self._enabled_sp.at[sp_id]]
        dis_reactions_ids = [r_id for r_id in reactions_ids if self._enabled_r.at[r_id]]
        # base case #1 (nothing to disable):
        if not len(dis_species_ids) and not len(dis_reactions_ids):
            return
        # base case #2 (some species cannot be disabled since they are protected):
        if len(set(species_ids) & set(self.get_protected_species().index)):
            raise ChemistryDisableError('Some of the species cannot be disabled!')
        # disable everything:
        self._enabled_sp[dis_species_ids] = False
        self._enabled_r[dis_reactions_ids] = False
        if self.requires_charge:
            pos_sp_mask = self.get_species_charge().values > 0
            # base case #3 (no positive ion left)
            if sum(pos_sp_mask) == 0:
                raise ChemistryDisableError('The last positive ion cannot be disabled!')
            elif sum(pos_sp_mask) == 1:  # if the last positive ion remaining, put it to protected species:
                last_ion_id = self.get_species()[pos_sp_mask].iat[0].id
                self._adhoc_protected_sp[last_ion_id] = True
        # get the stoichiometric matrices, just to see if any species or reactions are disconnected from the network
        network_matrix = \
            self.get_stoichiomatrix(disabled=True, special=True, method='lhs') + \
            self.get_stoichiomatrix(disabled=True, special=True, method='rhs')

        disconnected_species_mask = network_matrix.loc[self._enabled_r, self._enabled_sp].sum(axis=0) == 0
        disconnected_species_ids = disconnected_species_mask[disconnected_species_mask].index
        disconnected_reactions_mask = network_matrix.loc[self._enabled_r, ~self._enabled_sp].sum(axis=1) > 0
        disconnected_reactions_ids = disconnected_reactions_mask[disconnected_reactions_mask].index

        # base case #4 (the reaction network is all connected, nothing else needs to be disabled)
        if not len(disconnected_species_ids) and not len(disconnected_reactions_ids):
            return
        else:
            # recursive case:
            self._disable_recursively(
                species_ids=tuple(disconnected_species_ids), reactions_ids=tuple(disconnected_reactions_ids)
            )

    def disable(self, species=(), reactions=()):
        """Method to self-consistently and recursively disable species and reactions passed. The chemistry after
        a successful disabling event may have more species and reactions disabled than passed - this comes from the fact
        that if passed reactions contain all reactions involving certain species, this species is automatically
        disabled. Also disabling one species will disable all it's reactions, which in turn might disable another
        species by the mechanism described above.
        If during the "disabling cascade" a species needs to be disabled which is protected, the disabling event will
        not occur, all disabling is back-tracked and a ChemistryDisableError is raised.
        WARNING: if any of the disabled species feature as return species for any active species, the disabling event
        goes ahead and the return species with its coefficient is removed!
        :param species: (tuple of species) identified by id or name
        :param reactions: (tuple of reactions) identified by id
        :return: None
        """
        bu_enabled_reactions = self._enabled_r.copy()
        bu_enabled_species = self._enabled_sp.copy()
        bu_protected_species = self._protected_sp.copy()
        bu_adhoc_protected_species = self._adhoc_protected_sp.copy()

        dis_species_ids = [self.get_rp(rp).id for rp in species]
        dis_reactions_ids = [self.get_reaction(r).id for r in reactions]

        try:
            self._disable_recursively(species_ids=tuple(dis_species_ids), reactions_ids=tuple(dis_reactions_ids))
        except ChemistryDisableError:
            # back track
            self._enabled_sp = bu_enabled_species
            self._enabled_r = bu_enabled_reactions
            self._protected_sp = bu_protected_species
            self._adhoc_protected_sp = bu_adhoc_protected_species
            raise ChemistryDisableError('Some of the species cannot be disabled!')

        # just to be on a safe side, assert consistency:
        self.assert_consistency()

    def reset(self):
        """Resets all the disabled species and reactions.
        :return: None
        """
        self._enabled_sp = pd.Series(True, index=self._species.index)
        self._enabled_r = pd.Series(True, index=self._reactions.index)
        self._adhoc_protected_sp = pd.Series(False, index=self._species.index)

    def get_protected_species(self):
        """Returns series of species in the chemistry which are protected (cannot be disabled)
        :return: (pd.Series) of RP instances of all protected with species ids as indices
        """
        return self._species[np.array(self._protected_sp | self._adhoc_protected_sp)]

    def get_special_species(self):
        """Returns series of species in the chemistry which special (not ordinary heavy species) - 'e' and 'M' are
        special, others might be added in the future.
        :return:
        """
        return self._species[np.array(self._special_sp)]

    def get_disabled_species(self):
        """Returns list of species in the chemistry which are disabled
        :return: (pd.Series) of RP instances with species ids as indices
        """
        return self._species[~np.array(self._enabled_sp)]

    def get_disabled_reactions(self):
        """Returns list of reactions in the chemistry which are disabled
        :return: (pd.Series) of Reaction instances with reactions ids as indices
        """
        return self._reactions[~np.array(self._enabled_r)]

    # ***************************************** AD-HOC ATTRIBUTES CHANGES ******************************************** #

    def set_adhoc_species_attributes(self, attrib, rp, value):
        """This method is for setting some ad-hoc changes to species attributes on the chemistry level. These
        attributes values will override the species instances attributes in the chemistry-level getter methods.
        Only the attributes already defined in self.adhoc_species_attributes dict in __init__ are allowed to be set,
        will raise the RPAttributeError, if trying to set an unsupported attribute. An example might be setting
        different surface coefficients when trying to optimise the surface chemistry.
        Care must be taken about the units of the attribute being set. The set value is ALWAYS in the units defined
        by the RP.get_unit[attrib]

        :param attrib: (str) only attributes already set as keys in self.adhoc_species_attributes are allowed.
        :param rp: (int, str or RP instance) species id for which the chemistry level attribute should be set
        :param value: (object) the new attribute value which will be returned for the species in chemistry level getters
                      Only types specified in RP.allowed_attribute_types are allowed.
        :return: None
        """
        value = copy.deepcopy(value)  # just if the value is mutable...
        sp_id = self.get_rp(rp).id
        if sp_id not in self._species.index:
            raise ChemistrySpeciesNotPresentError('Passed ID does not belong to any species in the chemistry!')
        if attrib not in self.adhoc_species_attributes:
            raise RPAttributeError('Adhoc changes to the {} attribute cannot be made'.format(attrib))
        if type(value) not in RP.allowed_attribute_types[attrib]:
            raise RPAttributeError('Unsupported value type for {} attribute: {}'.format(attrib, type(value)))
        # need to convert np.float64 explicitly to float, otherwise the yaml dump will be a mess!
        if type(value) == np.float64:
            value = float(value)
        if attrib == 'ret_coefs':
            # verify the consistency of the chemistry with new return species
            species_names_pool = set(self.get_species_name(special=True).values)
            for ret_sp_name in value.keys():
                if ret_sp_name not in species_names_pool:
                    raise ChemistryConsistencyError('The return species incompatible with the chemistry species.')
            # also convert all the return coefficients explicitly to floats, as np types cause problems with yaml dumps
            value = {ret_sp: float(value[ret_sp]) for ret_sp in value}
        self.adhoc_species_attributes[attrib].at[sp_id] = value
        # just in case:
        self.assert_consistency()

    def reset_adhoc_species_attributes(self, attrib):
        """This method will reset all the ad-hoc chemistry level attributes for species.

        :param attrib: (str) only attributes already set as keys in self.adhoc_species_attributes are allowed.
                       also 'all' is allowed which will reset all of them defined in self.adhoc_species_attributes.
        :return: None
        """
        if attrib == 'all':
            for attr in self.adhoc_species_attributes:
                self.adhoc_species_attributes[attr] = pd.Series(dtype='O')
        else:
            if attrib not in self.adhoc_species_attributes:
                raise RPAttributeError('Adhoc changes to the {} attribute cannot be made'.format(attrib))
            self.adhoc_species_attributes[attrib] = pd.Series(dtype='O')

    def set_adhoc_reactions_attributes(self, attrib, r_id, value):
        """This method is for setting some ad-hoc changes to reactions attributes on the chemistry level. These
        attributes values will override the reactions instances attributes in the chemistry-level getter methods.
        Only the attributes already defined in self.adhoc_reactions_attributes dict in __init__ are allowed to be set,
        will raise the ReactionAttributeError, if trying to set an unsupported attribute. An example might be setting
        different arrh_a attribute for different reactions in Morris method of sensitivity analysis.
        Care must be taken about the units of the attribute being set. The set value is ALWAYS in the units defined
        by the Reaction.get_unit[attrib]

        :param attrib: (str) only attributes already set as keys in self.adhoc_reactions_attributes are allowed.
        :param r_id: (int) reaction id for which the chemistry level attribute should be set
        :param value: (object) the new attribute value which will be returned for the react. in chemistry level getters
                      Only types specified in RP.allowed_attribute_types are allowed.
        :return: None
        """
        value = copy.deepcopy(value)  # just if the value is mutable...
        if r_id not in self._reactions.index:
            raise ChemistryReactionNotPresentError('Passed ID does not belong to any reaction in the chemistry!')
        if attrib not in self.adhoc_reactions_attributes:
            raise ReactionAttributeError('Adhoc changes to the {} attribute cannot be made'.format(attrib))
        if type(value) not in Reaction.allowed_attribute_types[attrib]:
            raise ReactionAttributeError('Unsupported value type for {} attribute: {}'.format(attrib, type(value)))
        # need to convert np.float64 explicitly to float, otherwise the yaml dump will be a mess!
        if type(value) == np.float64:
            value = float(value)
        self.adhoc_reactions_attributes[attrib].at[r_id] = value

    def reset_adhoc_reactions_attributes(self, attrib):
        """This method will reset all the ad-hoc chemistry level attributes for reactions.

        :param attrib: (str) only attributes already set as keys in self.adhoc_reactions_attributes are allowed.
                       also 'all' is allowed which will reset all of them defined in self.adhoc_reactions_attributes.
        :return: None
        """
        if attrib == 'all':
            for attr in self.adhoc_reactions_attributes:
                self.adhoc_reactions_attributes[attr] = pd.Series(dtype='O')
        else:
            if attrib not in self.adhoc_reactions_attributes:
                raise ReactionAttributeError('Adhoc changes to the {} attribute cannot be made'.format(attrib))
            self.adhoc_reactions_attributes[attrib] = pd.Series(dtype='O')

    # ******************************** DUMPING CHEMISTRY INFORMATION TO A FILE *************************************** #

    def _chemistry_attributes(self):
        """Creates a dictionary full of all relevant attributes of the Chemistry instance, which fully define it's
        state. The saved attributes are:
        - enabled/disabled, protected species ids
        - enabled/disabled reactions ids
        - all chemistry ad-hoc changed parameters
        - path to the xml chemistry input file (if used) - this is for consistency validation purpose
        - xml chemistry input file hash - this is for consistency validation purpose
        WARNING: Any changes made here need to also be made in self.load_chemistry_attributes
        :return: (dict)
        """
        species_ids = {
            'enabled': list(self._enabled_sp[self._enabled_sp.values].index),
            'disabled': list(self._enabled_sp[~self._enabled_sp.values].index),
            'protected': list(self._protected_sp[self._protected_sp.values].index),
            'adhoc_protected': list(self._adhoc_protected_sp[self._adhoc_protected_sp.values].index),
        }
        reactions_ids = {
            'enabled': list(self._enabled_r[self._enabled_r.values].index),
            'disabled': list(self._enabled_r[~self._enabled_r.values].index),
        }

        adhoc_species_attributes = []
        for attr in self.adhoc_species_attributes:
            for sp_i in self.adhoc_species_attributes[attr].index:
                attr_value = self.adhoc_species_attributes[attr].at[sp_i]
                # need to explicitly convert the value into the correct datatype (numpy data-types play up in yaml)
                attr_value = RP.allowed_attribute_types[attr][0](attr_value)  # using RP.allowed_types lists
                adhoc_species_attributes.append([str(attr), int(sp_i), attr_value])

        adhoc_reactions_attributes = []
        for attr in self.adhoc_reactions_attributes:
            for r_i in self.adhoc_reactions_attributes[attr].index:
                attr_value = self.adhoc_reactions_attributes[attr].at[r_i]
                attr_value = Reaction.allowed_attribute_types[attr][0](attr_value)
                adhoc_reactions_attributes.append([str(attr), int(r_i), attr_value])

        xml_path = str(self.xml_path) if self.xml_path else None
        xml_hash = str(self.xml_hash(self.xml_path)) if self.xml_path else None

        chemistry_attributes = {
            'species_ids': species_ids,
            'reactions_ids': reactions_ids,
            'adhoc_species_attributes': adhoc_species_attributes,
            'adhoc_reactions_attributes': adhoc_reactions_attributes,
            'xml_path': xml_path,
            'xml_hash': xml_hash,
        }

        return chemistry_attributes

    def dump_chemistry_attributes(self, yaml_path):
        """Dump attributes of the chemistry such as enabled/disabled species and reactions ids,
        Dumps all into a yaml file in supplied path.
        Attributes and values stored are such that the chemistry can be restored from the yaml_file into the
        exactly same state as when the yaml_file was dumped. These are:
        - enabled/disabled, protected species ids
        - enabled/disabled reactions ids
        - all chemistry ad-hoc changed parameters
        - path to the xml chemistry input file (if used) - this is for consistency validation purpose
        - xml chemistry input file hash - this is for consistency validation purpose
        WARNING: Any changes made here need to also be made in self.load_chemistry_attributes

        :param yaml_path: (str) full path where to dump the yaml file
        :return: None
        """
        # dump chemistry attributes to a yaml file:
        with open(yaml_path, 'w') as stream:
            yaml.dump(self._chemistry_attributes(), stream=stream)

    def load_chemistry_attributes(self, yaml_path, check_hash=True):
        """Restores chemistry attributes from a yaml file. Raises ChemistryAttributeError if the yaml file does not
        exist or if is incompatible/inconsistent with current chemistry.
        The consistency check only looks at top level keys in the yaml, if they are there, it is expected that all the
        nested keys are correct and correct value types.
        All attributes dumped in the self.dump_chemistry_attributes method are read and restored.
        WARNING: Any changes made here need to also be made in self.dump_chemistry_attributes

        :param yaml_path: (str) full path to the yaml file from which to read attributes
        :param check_hash: (bool) if True, the chemistry xml hash from the attributes yaml is checked against the
                           one created from self.xml_path. If they do not match, error is raised. Defaults to True.
                           Ich check_hash is False, then the chemistry_xml path from the yaml is not checked either.
        :return: None
        """
        # validate consistency:
        if not os.path.isfile(yaml_path):
            raise ChemistryAttributeError('Invalid path to the chemistry attributes yaml file!')
        try:
            with open(yaml_path) as stream:
                attributes = yaml.load(stream=stream, Loader=yaml.FullLoader)

                enabled_sp_ids = attributes['species_ids']['enabled']
                disabled_sp_ids = attributes['species_ids']['disabled']
                protected_sp_ids = attributes['species_ids']['protected']
                try:  # need to ensure backwards compatibility, this did not use to get saved into chemistry attributes.
                    adhoc_protected_sp_ids = attributes['species_ids']['adhoc_protected']
                except KeyError:
                    warnings.warn('Ad-hoc protected species not found in the chemistry attributes! Assumed empty!')
                    adhoc_protected_sp_ids = []

                enabled_r_ids = attributes['reactions_ids']['enabled']
                disabled_r_ids = attributes['reactions_ids']['disabled']

                adhoc_species_attributes = attributes['adhoc_species_attributes']
                adhoc_reactions_attributes = attributes['adhoc_reactions_attributes']

                xml_path = attributes['xml_path']
                xml_hash = attributes['xml_hash']
        except KeyError:
            raise ChemistryAttributeError('Invalid yaml file!')
        if check_hash:
            if xml_path != self.xml_path:
                raise ChemistryAttributeError('Chemistry xml input file path is not matching!')
            if xml_hash != self.xml_hash(self.xml_path):
                raise ChemistryAttributeError('Chemistry xml input file content is not matching!')

        # all appears valid, so can start restoring the attributes:
        # masks:
        self._enabled_sp[enabled_sp_ids] = True
        self._enabled_sp[disabled_sp_ids] = False
        self._protected_sp[:] = False
        self._protected_sp[protected_sp_ids] = True
        self._adhoc_protected_sp[:] = False
        self._adhoc_protected_sp[adhoc_protected_sp_ids] = True
        self._enabled_r[enabled_r_ids] = True
        self._enabled_r[disabled_r_ids] = False
        # ad-hoc attributes:
        self.reset_adhoc_species_attributes('all')
        for params_list in adhoc_species_attributes:
            self.set_adhoc_species_attributes(*params_list)
        self.reset_adhoc_reactions_attributes('all')
        for params_list in adhoc_reactions_attributes:
            self.set_adhoc_reactions_attributes(*params_list)

        # assert consistency to be on a safe side:
        self.assert_consistency()

    @staticmethod
    def xml_hash(xml_path):
        """A method returning a hash of the xml file content. Used to verify consistency of the xml chemistry
        input files between different runs
        :param xml_path: (str) path to the xml chemistry input file
        :return: (str) hexdigest md5 hash
        """
        if xml_path is None:
            return None
        if not os.path.isfile(xml_path):
            raise ChemistryAttributeError('Invalid xml_path attribute!')

        with open(xml_path) as f:
            xml_hash = hashlib.md5(f.read().encode()).hexdigest()
        return xml_hash

    def check_coherence(self, yaml_path):
        """Checks if the self (Chemistry instance) is entirely coherent with a dumped chemistry attributes yaml file.
        If the attributes were dumped and in the meantime the chemistry state changed (disabling/enabling, ah-hoc
        changes, different chemistry xml file etc...), then this method will return False when the original yaml_path
        is passed...
        :param yaml_path: (str) full path where to dump the yaml file
        :return: (bool) True, if the chemistry state is identical to the one which dumped the yaml file.
        """
        with open(yaml_path) as stream:
            file_attributes = yaml.load(stream=stream, Loader=yaml.FullLoader)
        return file_attributes == self._chemistry_attributes()

    # **************************************** SPECIES GETTERS ******************************************************* #

    def get_species_mask(self, disabled, special, protected):
        """Returns a boolean mask for species in the chemistry.
        :param disabled: (bool) if True, unmasks disabled species.
        :param special: (bool) if True, unmasks special species 'e' and 'M'.
        :param protected: (bool) if True, protected species. Defaults to True.
        :return: (np.array) of (bool) of same len as self._species
        """
        if disabled:
            mask = np.ones(len(self._species), dtype=bool)
        else:
            mask = self._enabled_sp.values.copy()
        if not special:
            mask &= ~self._special_sp.values
        if not protected:
            mask &= ~(self._protected_sp | self._adhoc_protected_sp).values
        return mask

    def get_species(self, disabled=False, special=False, protected=True):
        """Returns list of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of RP instances with species ids as indices
        """
        mask = self.get_species_mask(disabled=disabled, special=special, protected=protected)
        return self._species[mask]  # masking it always (even with all True mask) makes a top-level copy - desired!

    def get_species_name(self, disabled=False, special=False, protected=True):
        """Returns list of names of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (str) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_name() for sp in species], index=species.index)

    def get_species_charge(self, disabled=False, special=False, protected=True):
        """Returns list of charges of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (int) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_charge() for sp in species], index=species.index)

    def get_species_mass(self, disabled=False, special=False, protected=True):
        """Returns list of masses of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (float) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_mass() for sp in species], index=species.index)

    def get_species_h_form(self, disabled=False, special=False, protected=True):
        """Returns list of enthalpies of formation of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (float) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_h_form() for sp in species], index=species.index)

    def get_species_lj_epsilon(self, disabled=False, special=False, protected=True):
        """Returns list of lj_epsilon parameters of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (float) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_lj_epsilon() for sp in species], index=species.index)

    def get_species_lj_sigma(self, disabled=False, special=False, protected=True):
        """Returns list of lj_sigmas of species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (float) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        return pd.Series([sp.get_lj_sigma() for sp in species], index=species.index)

    def get_species_stick_coef(self, disabled=False, special=False, protected=True):
        """Returns list of sticking coefficients of species in the chemistry. Overwrites values with adhoc changes.
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (float) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        species_stick_coef = pd.Series([sp.get_stick_coef() for sp in species], index=species.index)
        species_stick_coef.update(self.adhoc_species_attributes['stick_coef'])
        return species_stick_coef

    def get_species_ret_coefs(self, disabled=False, special=False, protected=True):
        """Returns list of return coefficients of species in the chemistry. Overwrites values with adhoc changes.
        Also if stick_coef for certain species are 0., return coefs automatically collapse to {} on the chemistry
        level. This action overrides the adhoc changes.
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (pd.Series) of (dict or None) with species ids as indices
        """
        species = self.get_species(disabled=disabled, special=special, protected=protected)
        species_ret_coefs = pd.Series([copy.deepcopy(sp.get_ret_coefs()) for sp in species], index=species.index)
        # update with the adhoc ret_coefs:
        for sp_id in self.adhoc_species_attributes['ret_coefs'].index:
            species_ret_coefs.at[sp_id] = copy.deepcopy(self.adhoc_species_attributes['ret_coefs'].at[sp_id])
        # override for all species whose stick_coefs are 0.:
        species_stick_coefs = self.get_species_stick_coef(disabled=disabled, special=special)
        for sp_id in species_stick_coefs.index:
            if species_stick_coefs.at[sp_id] == 0.:
                species_ret_coefs.at[sp_id] = {}

        # after disabling event, some returned species may no longer be in chemistry => remove those from the ret_coefs
        if sum(~self._enabled_sp.values) > 0:
            # that's only relevant if there are any disabled species. if not, consistency was asserted at __init__
            species_names_pool = set(self.get_species_name(disabled=False, special=True).values)
            for sp_id in species_ret_coefs.index:
                ret_coefs = species_ret_coefs.at[sp_id]
                if ret_coefs is not None:
                    disabled_ret_species = []
                    for ret_sp_name in ret_coefs.keys():
                        if ret_sp_name not in species_names_pool:
                            disabled_ret_species.append(ret_sp_name)
                    for ret_sp_name in disabled_ret_species:
                        species_ret_coefs.at[sp_id].pop(ret_sp_name)
        return species_ret_coefs

    def get_species_id(self, disabled=False, special=False, protected=True):
        """Returns list of IDs of all species in the chemistry
        :param disabled: (bool) if True, also returns disabled species. Defaults to False
        :param special: (bool) if True, also returns special species 'e' and 'M'. Defaults to False
        :param protected: (bool) if True, also returns all protected species. Defaults to True
        :return: (np.array) of int RP IDs.
        """
        return np.array(self.get_species(disabled=disabled, special=special, protected=protected).index)

    # ***************************************** REACTIONS GETTERS **************************************************** #

    def get_reactions_mask(self, disabled):
        """Returns a boolean mask for reactions in the chemistry.
        :param disabled: (bool) if True, unmasks disabled reactions. Defaults to False
        :return: (np.array) of (bool) of same len as self._species
        """
        if disabled:
            return np.ones(len(self._reactions), dtype=bool)
        else:
            return self._enabled_r.values.copy()

    def get_reactions(self, disabled=False):
        """Returns list of reactions in the chemistry
        :param disabled: (bool) if True, also returns disabled reactions. Defaults to False
        :return: (pd.Series) of Reaction instances with reactions ids as indices
        """
        mask = self.get_reactions_mask(disabled=disabled)
        return self._reactions[mask]

    def get_reactions_arrh_a(self, disabled=False, si_units=True):
        """Returns list of Arrhenius pre-exp factors for reactions in the chemistry. Overwritten here by the values
        stored in ad-hoc attributes.
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :param si_units: (bool) if True, arrhenius pre-exp factors returned in SI, if False => in nominal units
        :return: (pd.Series) of (float) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        reactions_arrh_a = pd.Series([r.get_arrh_a(si_units=si_units) for r in reactions], index=reactions.index)
        # adhoc changes (here I need to take care of the SI conversion as well!):
        adhoc_arrh_a = self.adhoc_reactions_attributes['arrh_a'].copy()
        if si_units:
            for r_id in adhoc_arrh_a.index:
                orig_unit = self.get_reaction(r_id).get_unit('arrh_a')
                adhoc_arrh_a[r_id] *= Reaction.si_conversion_factors['arrh_a'][orig_unit]
        reactions_arrh_a.update(adhoc_arrh_a)
        return reactions_arrh_a

    def get_reactions_arrh_b(self, disabled=False):
        """Returns list of Arrhenius n factors for reactions in the chemistry
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :return: (pd.Series) of (float) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        return pd.Series([r.get_arrh_b() for r in reactions], index=reactions.index)

    def get_reactions_arrh_c(self, disabled=False):
        """Returns list of Arrhenius activation energy (eV/K) factors for reactions in the chemistry
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :return: (pd.Series) of (float) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        return pd.Series([r.get_arrh_c() for r in reactions], index=reactions.index)

    def get_reactions_el_en_loss(self, disabled=False):
        """Returns list of electron energy losses (eV) for reactions in the chemistry
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :return: (pd.Series) of (float) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        return pd.Series([r.get_el_en_loss() for r in reactions], index=reactions.index)

    def get_reactions_ges_heat_contrib(self, disabled=False):
        """Returns list of changes in enthalpy manifesting as gas heating for reactions in the chemistry
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :return: (pd.Series) of (float) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        return pd.Series([r.get_gas_heat_contrib() for r in reactions], index=reactions.index)

    def get_reactions_elastic(self, disabled=False):
        """Returns list of bool values if a reaction is elastic for reactions in the chemistry
        :param disabled: (bool) if True, also returns values for disabled reactions. Defaults to False
        :return: (pd.Series) of (bool) with reaction ids as indices
        """
        reactions = self.get_reactions(disabled=disabled)
        return pd.Series([r.get_elastic() for r in reactions], index=reactions.index)

    def get_reactions_id(self, disabled=False):
        """Returns list of IDs of all reactions in the chemistry
        :param disabled: (bool) if True, also returns disabled reactions. Defaults to False
        :return: (np.array) of int with reactions ids
        """
        return np.array(self.get_reactions(disabled=disabled).index)

    # ************************************* HIGHER DIMENSIONAL GETTERS *********************************************** #

    def get_return_matrix(self, disabled=False, special=False):
        """Constructs and returns a matrix of return ratios between species. Takes into account all the return species
        and return coefficients and creates a matrix R, where R.iat[i, j] (ith row and jth column) denotes number of
        i-th species created by each one j-th species stuck to the surface. Returned in the form of pd.DataFrame with
        columns and indexes keyed by species ids.

        :param disabled: (bool) if to include disabled species
        :param special: (bool) if to include special species - this is only for consistency, special species do not
                        define surface coefficients, those rows and columns will be filled with NaN.
        :return: (pd.DataFrame) with index and columns of species ids.
        """
        species_ret_coefs = self.get_species_ret_coefs(disabled=disabled, special=special)
        ret_mat = pd.DataFrame(index=species_ret_coefs.index, columns=species_ret_coefs.index)

        for sp_id in species_ret_coefs.index:
            ret_coefs = species_ret_coefs[sp_id]  # this is a dict of {return_sp.get_name(): ret_coef}
            if ret_coefs is not None:
                for return_sp_name in ret_coefs:
                    return_sp_id = self.get_rp(return_sp_name).id
                    ret_coef = ret_coefs[return_sp_name]
                    ret_mat.at[return_sp_id, sp_id] = ret_coef
        ret_mat = ret_mat.fillna(0.)

        if special:  # fill in the special species values back to NaN's...
            ret_mat.at[self._special_sp, :] = np.nan
            ret_mat.at[:, self._special_sp] = np.nan

        return ret_mat

    def get_stoichiomatrix(self, disabled=False, special=False, method='net'):
        """Constructs and returns a matrix of stoichiometric coefficients S. The S.iat[i, j] (i-th row and j-th column)
        is a stoichiometric coefficient of j-th species in i-th reaction.

        :param disabled: (bool) if to include disabled species
        :param special: (bool) if to include special species
        :param method: (str) of {'lhs', 'rhs', 'net'} where 'lhs' will be positive coefs for LHS's of reactions,
                       'rhs' positive coefs for RHS's of reactions and 'net' will be 'rhs' - 'lhs' (net production
                       of species)
        :return: (pd.DataFrame) with index of reaction ids and columns of species ids.
        """
        species = self.get_species(disabled=disabled, special=special, protected=True)
        reactions = self.get_reactions(disabled=disabled)
        stoichiomatrix = pd.DataFrame(columns=species.index, index=reactions.index)
        for r_id in reactions.index:
            reaction_stoich_coefs = reactions[r_id].get_stoich_coefs(method=method)
            for sp_id in reaction_stoich_coefs:
                if sp_id in stoichiomatrix.columns:
                    stoichiomatrix.at[r_id, sp_id] = reaction_stoich_coefs[sp_id]
        stoichiomatrix = stoichiomatrix.fillna(0).astype(int)
        return stoichiomatrix

    # ***************************** HIGHER ORDER GETTERS (BEYOND SIMPLE ATTRIBUTES) ********************************** #

    def get_reactions_stoich_coefs_electron(self, disabled=False, method='net'):
        """Constructs and returns a vector of electron stoichiometric coefficients s. The s.iat[i]
        is a stoichiometric coefficient of electron species in i-th reaction.

        :param disabled: (bool) if to include disabled species
        :param method: (str) of {'lhs', 'rhs', 'net'} where 'lhs' will be positive coefs for LHS's of reactions,
                       'rhs' positive coefs for RHS's of reactions and 'net' will be 'rhs' - 'lhs' (net production
                       of species)
        :return: (pd.Series) with index of reaction ids.
        """

        electron_id = self.get_rp('e').id

        reactions = self.get_reactions(disabled=disabled)
        reactions_stoich_coefs_electron = pd.Series(index=reactions.index, dtype='i')  # signed integer
        for r_id in reactions.index:
            r = reactions[r_id]
            r_stoich_coefs = r.get_stoich_coefs(method=method)
            if electron_id in r_stoich_coefs:
                reactions_stoich_coefs_electron.at[r_id] = r_stoich_coefs[electron_id]
        reactions_stoich_coefs_electron = reactions_stoich_coefs_electron.fillna(0)
        return reactions_stoich_coefs_electron

    def get_reactions_stoich_coefs_arbitrary(self, disabled=False, method='net'):
        """Constructs and returns a vector s of arbitrary species 'M' stoichiometric coefficients. The s.iat[i]
        is a stoichiometric coefficient of 'M' species in i-th reaction.

        :param disabled: (bool) if to include disabled species
        :param method: (str) of {'lhs', 'rhs', 'net'} where 'lhs' will be positive coefs for LHS's of reactions,
                       'rhs' positive coefs for RHS's of reactions and 'net' will be 'rhs' - 'lhs' (net production
                       of species)
        :return: (pd.Series) with index of reaction ids.
        """
        reactions = self.get_reactions(disabled=disabled)

        try:
            arbitrary_id = self.get_rp('M').id
        except ChemistrySpeciesNotPresentError:
            return pd.Series(0, index=reactions.index).astype(int)

        reactions_stoich_coefs_arbitrary = pd.Series(index=reactions.index, dtype='i')
        for r_id in reactions.index:
            r = reactions[r_id]
            r_stoich_coefs = r.get_stoich_coefs(method=method)
            if arbitrary_id in r_stoich_coefs:
                reactions_stoich_coefs_arbitrary.at[r_id] = r_stoich_coefs[arbitrary_id]
        reactions_stoich_coefs_arbitrary = reactions_stoich_coefs_arbitrary.fillna(0)
        return reactions_stoich_coefs_arbitrary

    # ******************************* PRINTOUTS OF SPECIES/REACTIONS ATTRIBUTES ************************************** #

    def species_attributes(self, disabled=True, special=True, printout=True,
                           pretty_print=True, sort=False, units=False,
                           ignored=('latex',)):
        """Returns a DataFrame with all attributes of all species. This dataframe is ONLY MEANT TO BE PRINTED into
        stdout. The dataframe is constructed in a way that it supports:
            - color-code in red species which are disabled from the chemistry
            - color-code in blue attributes, which can be defined, but were not defined and instead were returned
              by the RP methods as defaults. Also includes attributes which were overwritten by ad-hoc changes.
              In short, blue color-coding is for attributes, for which the chemistry-level
              getters return a something different than what is stored in RP._attribute.

        :param disabled: (bool) if True, also disabled species are present in the DF
        :param special: (bool) if True, also special species are present in the DF
        :param printout: (bool) if True, prints into stdout right before returning
        :param pretty_print: (bool) if True, will do the color-coding
        :param sort: (bool) if True, species are sorted according to the scheme defined in pygmol.classification
        :param units: (bool) if True, also attributes' units are included in the DF
        :param ignored: (iterable of str) these attributes will not be included in the dataframe of attributes.
                        Defaults to only 'latex'
        :return: (pd.DataFrame)
        """
        attributes = [attr for attr in RP.default_attributes.keys() if attr not in ignored]
        species = self.get_species(disabled=disabled, special=special, protected=True)

        species_attributes = pd.DataFrame()
        for attr in attributes:
            try:
                # check if a chemistry-level getter method is defined
                column = getattr(self, 'get_species_{}'.format(attr))(disabled=disabled, special=special)
            except AttributeError:
                # getter method is not defined, get it directly from species getter method
                column = pd.Series([getattr(sp, 'get_{}'.format(attr))() for sp in species], index=species.index)
            species_attributes[attr] = column
            # units:
            if units:
                try:
                    units_array = pd.Series([sp.get_unit(attr) for sp in species], index=species.index)
                    species_attributes['[{}]'.format(attr)] = units_array
                except RPAttributeError:
                    pass

        if sort:
            reactions_classifier = ReactionClassifier(species=species, reactions=[None, ])
            reactions_classifier.sort_species()
            sorted_index = [sp.id for sp in reactions_classifier.species]
            species_attributes = species_attributes.loc[sorted_index]

        if pretty_print:
            # convert all values to ColoValue type:
            pp = pd.DataFrame(index=species_attributes.index, columns=species_attributes.columns, dtype='O')
            for i in pp.index:
                for c in pp.columns:
                    pp.at[i, c] = ColorValue(species_attributes.at[i, c])
            # change colors to color-code:
            for sp_id in pp.index:
                for col in pp.columns:
                    if not self._enabled_sp[sp_id]:  # color-code red if disabled
                        pp.at[sp_id, col].set_color('RED')
                    else:  # check if to color-code blue:
                        if not str(col).startswith('['):  # dealing with attributes, not units.
                            saved_attrib = getattr(species.at[sp_id], '_{}'.format(col))
                            returned_attrib = pp.at[sp_id, col].value
                            equal = saved_attrib == returned_attrib
                            if not equal:
                                try:  # None are converted to nan in the chemistry level getters...
                                    equal = saved_attrib is None and np.isnan(returned_attrib)
                                except TypeError:
                                    pass
                            if not equal:
                                try:  # float attributes might be saved as int or str...
                                    equal = float(saved_attrib) == float(returned_attrib)
                                except TypeError:
                                    pass
                            if not equal and col == 'ret_coefs':
                                try:  # float return coefs might be saved as str on int...
                                    equal = {key: float(saved_attrib[key]) for key in saved_attrib} == returned_attrib
                                except TypeError:
                                    pass
                            if not equal:  # now I'm certain they're not equal => color code in blue
                                pp.at[sp_id, col].set_color('BLUE')
                        else:  # dealing with units
                            attr = str(col).strip('[]')
                            # noinspection PyProtectedMember
                            saved_unit = species.at[sp_id]._units[attr]
                            returned_unit = pp.at[sp_id, col].value
                            equal = saved_unit == returned_unit
                            if not equal:
                                pp.at[sp_id, col].set_color('BLUE')

                    # fill None or nan values with '...' substitute:
                    val = pp.at[sp_id, col].value
                    subs = '...'
                    if val is None:
                        pp.at[sp_id, col].set_value(subs)
                    else:
                        try:
                            if np.isnan(val):
                                pp.at[sp_id, col].set_value(subs)
                        except TypeError:
                            pass
            # convert also the columns to ColorValue type to keep alignment:
            pp.columns = [ColorValue(c) for c in pp.columns]
            species_attributes = pp

        if printout:
            print()
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print(species_attributes)
            print()

        return species_attributes

    def reactions_attributes(self, disabled=True, printout=True,
                             pretty_print=True, sort=False, classify=False, units=False,
                             ignored=('latex',)):
        """Returns a DataFrame with all attributes of all species. This dataframe is ONLY MEANT TO BE PRINTED into
        stdout. The dataframe is constructed in a way that it supports:
            - color-code in red species which are disabled from the chemistry
            - color-code in blue attributes, which can be defined, but were not defined and instead were returned
              by the RP methods as defaults. Also includes attributes which were overwritten by ad-hoc changes.
              In short, blue color-coding is for attributes, for which the chemistry-level
              getters return a something different than what is stored in RP._attribute.

        :param disabled: (bool) if True, also disabled species are present in the DF
        :param printout: (bool) if True, prints into stdout right before returning
        :param pretty_print: (bool) if True, will do the color-coding
        :param sort: (bool) if True, reactions will be sorted according to as scheme defined in plaschem.classification
        :param classify: (bool) if True, reactions will be classified into types and categories according to the
                         scheme defined in plaschem.classification. The reaction types and categories will be inserted
                         into the reaction_attributes dataframe and color-coded in yellow/green
        :param units: (bool) if True, also attributes' units are included in the DF
        :param ignored: (iterable of str) these attributes will not be included in the dataframe of attributes.
                        Defaults to only 'latex'
        :return: (pd.DataFrame)
        """
        attributes = ['string'] + [attr for attr in Reaction.default_attributes.keys() if attr not in ignored]
        reactions = self.get_reactions(disabled=disabled)

        reactions_attributes = pd.DataFrame()
        for attr in attributes:
            if attr == 'arrh_a':
                column = self.get_reactions_arrh_a(disabled=disabled, si_units=False)  # need in nominal units
            else:
                try:
                    # check if a chemistry-level getter method is defined
                    column = getattr(self, 'get_reactions_{}'.format(attr))(disabled=disabled)
                except AttributeError:
                    # getter method is not defined, get it directly from reactions getter method
                    column = pd.Series([getattr(r, 'get_{}'.format(attr))() for r in reactions], index=reactions.index)
            reactions_attributes[attr] = column
            # units:
            if units:
                try:
                    units_array = pd.Series([r.get_unit(attr) for r in reactions], index=reactions.index)
                    reactions_attributes['[{}]'.format(attr)] = units_array
                except ReactionAttributeError:
                    pass

        if sort or classify:
            species = self.get_species(disabled=disabled, special=True, protected=True)
            reactions_classifier = ReactionClassifier(species=species, reactions=reactions)
            if sort:
                reactions_classifier.sort_species()
                reactions_classifier.sort_reactions()
            if classify:
                classified_reactions = reactions_classifier.get_classified_reactions()
                first_col_width = 0
                # add the reaction types and categories into the reactions_attributes keyed by 'CX' and 'CX.Y'
                sorted_index = []
                for i, r_type in enumerate(classified_reactions):
                    columns = reactions_attributes.columns
                    sorted_index.append('(C{})'.format(i + 1))
                    reactions_attributes.loc['(C{})'.format(i + 1)] = \
                        pd.Series([r_type] + [''] * (len(columns) - 1), index=columns)
                    if len(r_type) > first_col_width:
                        first_col_width = len(r_type)
                    for j, r_category in enumerate(classified_reactions[r_type]):
                        if r_category != 'None':
                            sorted_index.append('(C{}.{})'.format(i + 1, j + 1))
                            reactions_attributes.loc['(C{}.{})'.format(i + 1, j + 1)] = \
                                pd.Series([r_category] + [''] * (len(columns) - 1), index=columns)
                            if len(r_category) > first_col_width:
                                first_col_width = len(r_category)
                        sorted_index.extend([r.id for r in classified_reactions[r_type][r_category]])
                # replace all the reaction strings for left-justified aligned strings:
                for r in reactions:
                    aligned_str = reactions_classifier.get_aligned_reaction_string(r)
                    reactions_attributes.loc[r.id, 'string'] = aligned_str.ljust(first_col_width)
                    if len(aligned_str) > first_col_width:
                        first_col_width = len(aligned_str)
                # left-justify all the r_categories:
                for i, r_type in enumerate(classified_reactions):
                    for j, r_category in enumerate(classified_reactions[r_type]):
                        reactions_attributes.loc['(C{}.{})'.format(i + 1, j + 1), 'string'] = \
                            r_category.ljust(first_col_width)
                # sort all rows:
                reactions_attributes = reactions_attributes.loc[sorted_index]
            elif sort:
                sorted_index = [r.id for r in reactions_classifier.reactions]  # array of indices of sorted reactions
                reactions_attributes = reactions_attributes.loc[sorted_index]

        if pretty_print:
            # convert all values to ColoValue type:
            pp = pd.DataFrame(index=reactions_attributes.index, columns=reactions_attributes.columns, dtype='O')
            for i in pp.index:
                for c in pp.columns:
                    pp.loc[i, c] = ColorValue(reactions_attributes.loc[i, c])
            # change colors to color-code for reactions:
            for r_id in reactions.index:
                for col in pp.columns:
                    if not self._enabled_r[r_id]:  # color-code red if disabled
                        pp.loc[r_id, col].set_color('RED')
                    elif col != 'string':  # check if to color-code blue, string is not proper attribute:
                        if not str(col).startswith('['):  # dealing with attributes, not units:
                            saved_attrib = getattr(reactions.at[r_id], '_{}'.format(col))
                            returned_attrib = pp.loc[r_id, col].value
                            equal = saved_attrib == returned_attrib
                            if not equal:
                                try:  # None are converted to nan in the chemistry level getters...
                                    equal = saved_attrib is None and np.isnan(returned_attrib)
                                except TypeError:
                                    pass
                            if not equal:
                                # if saved attribute is 'True' or 'False', need to catch it:
                                if saved_attrib in {'True', 'true', 'TRUE'}:
                                    equal = returned_attrib is True
                                elif saved_attrib in {'False', 'false', 'FALSE'}:
                                    equal = returned_attrib is False
                                else:  # if not boolean, it should be number...
                                    try:  # float attributes might be saved as int or str...
                                        equal = float(saved_attrib) == float(returned_attrib)
                                    except TypeError:
                                        pass
                            if not equal:  # now I'm certain they're not equal => color code in blue
                                pp.loc[r_id, col].set_color('BLUE')
                        else:  # dealing with units:
                            attr = str(col).strip('[]')
                            # noinspection PyProtectedMember
                            saved_unit = reactions.at[r_id]._units[attr]
                            returned_unit = pp.loc[r_id, col].value
                            equal = saved_unit == returned_unit
                            if not equal:
                                pp.loc[r_id, col].set_color('BLUE')

                    # fill None or nan values with '...' substitute:
                    val = pp.loc[r_id, col].value
                    subs = '...'
                    if val is None:
                        pp.loc[r_id, col].set_value(subs)
                    else:
                        try:
                            if np.isnan(val):
                                pp.loc[r_id, col].set_value(subs)
                        except TypeError:
                            pass
            # color-code the reaction types and categories:
            if classify:
                # noinspection PyUnboundLocalVariable
                for i, r_type in enumerate(classified_reactions):
                    pp.loc['(C{})'.format(i + 1), 'string'].set_color('YELLOW')
                    for j, r_category in enumerate(classified_reactions[r_type]):
                        if r_category != 'None':
                            pp.loc['(C{}.{})'.format(i + 1, j + 1), 'string'].set_color('MAGENTA')
            # convert also the columns to ColorValue type to keep alignment:
            pp.columns = [ColorValue(c) for c in pp.columns]
            reactions_attributes = pp

        if printout:
            print()
            with pd.option_context(
                    'display.max_rows', None, 'display.max_columns', None, 'display.width', None,
                    'display.max_colwidth', 999,
            ):
                print(reactions_attributes)
            print()

        return reactions_attributes
