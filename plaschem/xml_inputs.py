import os
import re
import sys
from collections import OrderedDict
from xml.dom import minidom
from xml.etree import ElementTree

# set the correct context, if executed from within the QDB django shell:
if __package__ == '':  # this is the QDB package name for some reason...
    plaschem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if plaschem_path not in sys.path:
        print("Appending '{}' to the system path variable... ".format(plaschem_path))
        sys.path.append(plaschem_path)

from plaschem.species import RP
from plaschem.reactions import Reaction
from plaschem.classification import ReactionClassifier
from plaschem.exceptions import XmlError, RPAttributeError, ReactionAttributeError


# noinspection DuplicatedCode
class XmlBuilder(object):
    """
    This class is dedicated to building chemistry xml input files. Once the XmlBuilder is instantiated, one can add
    species and reactions at will, either directly from attributes, from RP and Reaction instances or from QDB (for
    the last option, the XmlBuilder class needs to be imported in the QDB django shell.)
    Bundles of species and reactions can also be added from a Chemistry instance, QDB Chemistry object or another
    (previously built) xml input file.
    The xml can easily be created by calling the dump_xml method and later loaded again. The XmlBuilder also can
    be used to sort species and reactions in the resulting xml and cleanly assign RP and Reaction ids ascending from
    0 (or 1 for reactions) to N.
    There are no chemistry consistency checks implemented on the xml level. If the species and reactions in the xml
    are not consistent, all kinds of errors will be raised when parsing the .xml file for RP and Reaction instances
    or when building the Chemistry instance out of those.
    """

    rp_attributes = list(RP.default_attributes.keys())  # allowed attributes for the RP instances...
    r_attributes = list(Reaction.default_attributes.keys())  # allowed attributes for the Reaction instances...

    def __init__(self):
        """XmlBuilder initialiser. Pretty does nothing - only prepares lists and arrays - the same as reset_chemistry.
        """
        self.species_dict = {}  # dict of rp_name: rp_xml_element
        self.species = []  # list filled with the same xml element objects as the species_dict, but ordered!

        self.reactions = []  # list of reaction xml elements

    def reset_chemistry(self):
        """Method resetting the chemistry - removing all the species and reactions xml elements from their arrays.
        Currently done by re-calling the __init__ method.
        :return: None
        """
        self.__init__()

    # ********************************** RENAMING CONVENTIONS FROM QDB OBJECTS *************************************** #

    def _get_rp_name_from_qdb(self, qdb_rp):
        """Helper method creating the preferred name for an RP instance from it's QDB RP counterpart. Only relevant
        if building xml from inside the QDB shell.
        :param qdb_rp: (qdb.rp.RP instance)
        :return: (str) name of an RP which will be created from the QDB RP instance
        """
        stateless_str = self._get_rp_stateless_from_qdb(qdb_rp)  # name without the states
        states_str = self._get_rp_state_from_qdb(qdb_rp)  # string describing the states
        name = '{}{}'.format(stateless_str, states_str)  # simple concatenations
        return name

    @staticmethod
    def _get_rp_state_from_qdb(qdb_rp):
        """Method to get a custom state string representation from a QDB RP instance. Only relevant if building
        xml from inside the QDB shell.
        :param qdb_rp: (qdb.rp.RP instance)
        :return: (str) state string which will be used to create an RP instance from the QDB RP instance
        """
        state_strs = [state.state_str for state in qdb_rp.state.all()]
        states_str = ','.join(state_strs)
        # I want parenthesis around the states only if they are not */**/***...
        if states_str.strip('*'):
            states_str = '({})'.format(states_str)
        return states_str

    @staticmethod
    def _get_rp_stateless_from_qdb(qdb_rp):
        """Method to get a string representation (name) of a "stateless" counterpart of a QDB RP instance.
        Only relevant if building xml from inside the QDB shell. Will be used to create RP from it's QDB RP counterpart.
        :param qdb_rp: (qdb.rp.RP instance)
        :return: (str) stateless RP counterpart string which will be used to create an RP instance from the
                 QDB RP instance
        """
        stateless_str = qdb_rp.ordinary_formula
        # take care of any exceptions:
        if stateless_str == 'e-':  # electron renaming:
            stateless_str = 'e'
        for charge in ['+', '-']:  # multiple charges:
            for ionisation_degree in [2, 3]:
                qdb_charge_suffix = charge + str(ionisation_degree)
                if stateless_str.endswith(qdb_charge_suffix):
                    # replace 'Ar+2' with Ar++
                    stateless_str = stateless_str[:-len(qdb_charge_suffix)] + charge * ionisation_degree

        return stateless_str

    # ********************************************* ADDING SPECIES *************************************************** #

    def add_rp_from_attributes(self, rp_id=None, rp_attributes=None, rp_units=None, **additional_attributes):
        """Add a species xml element from it's attributes. All the passed parameters should be the same as for
        RP initialiser. If an rp xml element with the same name tag text is already present, XmlError is raised.
        The rp_attributes and additional attributes might also contain keys not supported by the RP constructor, in
        that case these are simply all ignored.
        :param rp_id: (int) optional. If not passed, 999 will be assigned
        :param rp_attributes: (dict) or RP attributes (look at RP.__init__)
        :param rp_units: (dict) or RP units (look at RP.__init__)
        :param additional_attributes: other RP attributes (look at RP.__init__)
        :return: None
        """
        if rp_id is None:
            rp_id = 999

        if rp_units is None:
            rp_units = {}
        if rp_attributes is None:
            all_attributes = {}
        else:
            all_attributes = rp_attributes.copy()
        all_attributes.update(additional_attributes)

        if all_attributes['name'] in self.species_dict:
            raise XmlError('Cannot add species {}, when it is already present!'.format(all_attributes['name']))

        rp_xml = ElementTree.Element('rp', attrib={'id': str(rp_id)})  # create the rp xml element
        skip_attributes = {'ret_coefs'}
        for attribute in self.rp_attributes:
            if attribute in all_attributes and all_attributes[attribute] not in [None, '']:
                if attribute not in skip_attributes:  # create the rp xml sub-elements
                    if attribute in rp_units:
                        attrib_xml = ElementTree.SubElement(rp_xml, attribute, attrib={'unit': rp_units[attribute]})
                    else:
                        attrib_xml = ElementTree.SubElement(rp_xml, attribute)
                    attrib_xml.text = str(all_attributes[attribute])
                else:  # deal with skipped attributes:
                    if attribute == 'ret_coefs':  # create the ret_coefs sub-element (or rp element)
                        ret_coefs_xml = ElementTree.SubElement(rp_xml, attribute)
                        for ret_sp in all_attributes[attribute]:
                            ret_sp_xml = ElementTree.SubElement(ret_coefs_xml, 'r', attrib={'rp': ret_sp})
                            ret_sp_xml.text = str(all_attributes[attribute][ret_sp])

        self.species_dict[all_attributes['name']] = rp_xml
        self.species.append(rp_xml)

    def add_rp_from_instance(self, rp):
        """Method to create and add an rp xml element from an RP instance. The ID will be preserved.
        If an rp xml element with the same name tag text is already present, XmlError is raised.
        :param rp: (RP instance)
        :return: None
        """
        rp_attributes = {}
        for attrib in self.rp_attributes:
            try:
                rp_attributes[attrib] = getattr(rp, '_{}'.format(attrib))
            except AttributeError:
                pass
        rp_units = {}
        for attrib in self.rp_attributes:
            try:
                rp_units[attrib] = rp.get_unit(attrib)  # this will save even default units explicitly!
            except RPAttributeError:
                pass
        self.add_rp_from_attributes(rp_id=rp.id, rp_attributes=rp_attributes, rp_units=rp_units)

    def add_rp_from_qdb(self, qdb_rp):
        """Method to create and add an rp xml element from a QDB RP instance. This needs to be called from within
        the QDB django shell. The rp xml element id will be the same as QDB RP primary key.
        If an rp xml element with the same name tag text is already present, XmlError is raised.
        :param qdb_rp: (qdb.rp.RP instance)
        :return: None
        """
        rp_attributes = {}
        rp_units = {}

        name = self._get_rp_name_from_qdb(qdb_rp)
        rp_attributes['name'] = name

        if len(qdb_rp.state.all()):
            rp_attributes['stateless'] = self._get_rp_stateless_from_qdb(qdb_rp)
            rp_attributes['state'] = self._get_rp_state_from_qdb(qdb_rp)

        rp_attributes['mass'] = qdb_rp.mass
        rp_units['mass'] = 'amu'

        if qdb_rp.charge != 0:
            rp_attributes['charge'] = qdb_rp.charge
            rp_units['charge'] = 'e'

        if qdb_rp.hform:
            rp_attributes['h_form'] = qdb_rp.hform
            rp_units['h_form'] = 'eV'

        if qdb_rp.lj_epsilon:
            rp_attributes['lj_epsilon'] = qdb_rp.lj_epsilon
            rp_units['lj_epsilon'] = 'K'

        if qdb_rp.lj_sigma:
            rp_attributes['lj_sigma'] = qdb_rp.lj_sigma
            rp_units['lj_sigma'] = 'A'

        if qdb_rp.hpem:
            rp_attributes['hpem_name'] = qdb_rp.hpem

        rp_attributes['qdb_id'] = qdb_rp.pk

        self.add_rp_from_attributes(rp_id=qdb_rp.pk, rp_attributes=rp_attributes, rp_units=rp_units)

    # ******************************************** ADDING REACTIONS ************************************************** #

    def add_reaction_from_attributes(self, reactants_names, products_names, r_id=None,
                                     r_attributes=None, r_units=None, **additional_attributes):
        """Add a reaction xml from it's attributes. All the passed parameters apart from reactants_names and
        products_names should be the same as for Reaction initialiser. If any of the reactants or products are not
        present in species already, XmlError is raised.
        The r_attributes and additional attributes might also contain keys not supported by the Reaction constructor, in
        that case these are simply all ignored.
        :param reactants_names: (tuple) of (str) of reactants names. All of those must already be present in species.
        :param products_names: (tuple) of (str) of products names. All of those must already be present in species.
        :param r_id: (int) optional. If not passed, 999 will be assigned
        :param r_attributes: (dict) or Reaction attributes (look at Reaction.__init__)
        :param r_units: (dict) or Reaction units (look at Reaction.__init__)
        :param additional_attributes: other Reaction attributes (look at Reaction.__init__)
        :return: None
        """
        if r_id is None:
            r_id = 999

        if r_units is None:
            r_units = {}
        if r_attributes is None:
            all_attributes = {}
        else:
            all_attributes = r_attributes.copy()
        all_attributes.update(additional_attributes)

        # check if all the species objects (xml elements) are present in self.species...
        for rp_name in set(reactants_names + products_names):
            if rp_name not in self.species_dict:
                raise XmlError('Some of the reactants or products are not present in species!')

        # the reaction root element:
        r_xml = ElementTree.Element('reaction', attrib={'id': str(r_id)})

        # reactants and products:
        ElementTree.SubElement(r_xml, 'reactants').text = ', '.join(reactants_names)
        ElementTree.SubElement(r_xml, 'products').text = ', '.join(products_names)
        # attributes:
        for attribute in self.r_attributes:
            if attribute in all_attributes and all_attributes[attribute] not in [None, '']:
                if attribute in r_units:
                    attrib_xml = ElementTree.SubElement(r_xml, attribute, attrib={'unit': r_units[attribute]})
                else:
                    attrib_xml = ElementTree.SubElement(r_xml, attribute)
                attrib_xml.text = str(all_attributes[attribute])

        self.reactions.append(r_xml)

    def add_reaction_from_instance(self, r):
        """Method creating and adding a reaction xml element from the passed Reaction instance. The ID is preserved.
        Species do not have to be present already, if reactants/products are not present (judging by species names),
        they will be added automatically.
        :param r: (Reaction instance)
        :return: None
        """
        # check if all reactants and products already have their xml objects, if no, add them
        for rp in r.get_reactants() + r.get_products():
            name = rp.get_name()
            if name not in self.species_dict:
                self.add_rp_from_instance(rp)

        # add the reaction, after all the species from reactants and products are already in place...
        reactants_names = tuple([rp.get_name() for rp in r.get_reactants()])
        products_names = tuple([rp.get_name() for rp in r.get_products()])
        r_attributes = {}
        for attrib in self.r_attributes:
            try:
                r_attributes[attrib] = getattr(r, '_{}'.format(attrib))
            except AttributeError:
                pass
        r_units = {}
        for attrib in self.r_attributes:
            try:
                r_units[attrib] = r.get_unit(attrib)  # this will save even default units explicitly!
            except ReactionAttributeError:
                pass
        self.add_reaction_from_attributes(
            reactants_names, products_names, r_id=r.id, r_attributes=r_attributes, r_units=r_units
        )

    def add_reaction_from_qdb(self, qdb_r):
        """Method to create and add a reaction xml element from the passed QDB Reaction instance.
        This needs to be called from within the QDB django shell. The reaction xml element id will be the same as
        QDB Reaction instance primary key. Species do not have to be present already, if reactants/products are not
        present (judging by species names), they will be added automatically.
        :param qdb_r: (qdb.rxn.Reaction instance)
        :return: None
        """
        # first, check if the qdb_r is a reaction lumping several reactions with resolved states:
        parent_reactions = qdb_r.parent_reaction.all()
        for resolved_reaction in parent_reactions:  # each parent reaction contains a resolved state species
            self.add_reaction_from_qdb(resolved_reaction)  # if there are any, add all of them instead and return...
        if len(parent_reactions):  # nothing else to do...
            return

        # now, if adding the "parent reaction" (a reaction with resolved states, which has a child reaction with
        # lumped states attached to it), I should take the dataset from the parent reaction, but species from the
        # child reaction...
        child_reactions = qdb_r.child_reactions.all()  # this should be of length either 0 or 1... Otherwise what to do?
        if len(child_reactions) > 1:
            raise XmlError('The reaction R{} has more than one child reaction!'.format(qdb_r.pk))
        master_reaction = child_reactions[0] if len(child_reactions) else qdb_r

        # check if all reactants and products already have their xml objects, if no, add them
        for qdb_rp in master_reaction.get_all_rps():
            name = self._get_rp_name_from_qdb(qdb_rp)
            if name not in self.species_dict:
                self.add_rp_from_qdb(qdb_rp)

        reactants_names = tuple([self._get_rp_name_from_qdb(qdb_rp) for qdb_rp in master_reaction.reactants.all()])
        products_names = tuple([self._get_rp_name_from_qdb(qdb_rp) for qdb_rp in master_reaction.products.all()])

        # get the qdb datasets:
        qdb_datasets = qdb_r.reactiondataset_set.all()
        if len(qdb_datasets) == 0:
            raise XmlError('The reaction R{} does not have any datasets attached'.format(qdb_r.pk))
        elif len(qdb_datasets) == 1:
            qdb_ds = qdb_datasets[0]
        else:  # if more than 1 dataset:
            # for electron collisions, prefer x-sec datasets, for heavy species collisions, prefer arrhenius datasets
            if 'e' in reactants_names:
                qdb_xsec_datasets = qdb_datasets.filter(dataset_type__pk=2)
                if qdb_xsec_datasets.count() != 0:
                    qdb_ds = qdb_xsec_datasets[0]
                else:
                    qdb_ds = qdb_datasets[0]
            else:
                qdb_nonxsec_datasets = qdb_datasets.filter(dataset_type__pk__in={1, 3})
                if qdb_nonxsec_datasets.count() != 0:
                    qdb_ds = qdb_nonxsec_datasets[0]
                else:
                    qdb_ds = qdb_datasets[0]

        r_attributes = {}
        r_units = {}

        # arrhenius data:
        if qdb_ds.is_xsec():
            (arrh_a, _), (arrh_b, _), (arrh_c, _) = qdb_ds.fit_arrhenius()
            # what is the unit? E unit is always eV (since the fitting method not defined for heavy sp. collisions)
            # and the A unit needs to be determined from the number of reactants of the reaction.
            arrh_c_unit = 'eV'
            num_reactants = len(reactants_names)
            if num_reactants == 2:
                arrh_a_unit = 'cm3.s-1'
            elif num_reactants == 3:
                arrh_a_unit = 'cm6.s-1'
            else:
                raise XmlError('Arrhenius formula can only be fitted for reaction with 2-3 reactants.')
        elif qdb_ds.is_arrhenius():
            params_a = qdb_ds.parameter_set.filter(meta__name__startswith='A')
            if not len(params_a):
                raise XmlError('Reaction R{} is missing arrhenius parameter A'.format(qdb_r.pk))
            params_b = qdb_ds.parameter_set.filter(meta__name='n')
            if not len(params_b):
                raise XmlError('Reaction R{} is missing arrhenius parameter n'.format(qdb_r.pk))
            params_c = qdb_ds.parameter_set.filter(meta__name__endswith='E')
            if not len(params_c):
                raise XmlError('Reaction R{} is missing arrhenius parameter E'.format(qdb_r.pk))
            # convert to float and to SI:
            arrh_a = float(params_a[0].value)
            arrh_a_unit = str(params_a[0].meta.units)
            arrh_b = float(params_b[0].value)
            arrh_c = float(params_c[0].value)
            arrh_c_unit = str(params_c[0].meta.units)
        else:
            raise XmlError('Reaction dataset D{} does not have xsec or arrh data!'.format(qdb_ds.pk))
        r_attributes['arrh_a'] = arrh_a
        r_units['arrh_a'] = arrh_a_unit
        if arrh_b:
            r_attributes['arrh_b'] = arrh_b
        if arrh_c:
            r_attributes['arrh_c'] = arrh_c
            r_units['arrh_c'] = arrh_c_unit

        # electron energy loss:
        el_en_loss = 0.
        electron_process = 'e' in reactants_names
        elastic_process = sorted(reactants_names) == sorted(products_names)
        if electron_process and not elastic_process and 'e' in products_names:
            # extract the electron energy loss approximate either from the cross section data or from the arrh. data
            if qdb_ds.is_arrhenius():  # approximate the threshold energy with the activation energy
                el_en_loss = max(0, arrh_c)
            elif qdb_ds.is_xsec():
                if qdb_ds.threshold is not None:
                    el_en_loss = float(qdb_ds.threshold)  # if it's explicitly saved
                else:
                    el_en_loss = qdb_ds.get_thresh()  # if not, read from xsec values via dedicated method in qdb model
            else:
                raise XmlError('DataSet D{} is neither arrhenius, nor xsec!'.format(qdb_ds.pk))

        if el_en_loss:
            r_attributes['el_en_loss'] = el_en_loss
            r_units['el_en_loss'] = 'eV'

        # Global_Kin special number:
        if qdb_ds.entry_id is not None and qdb_ds.entry_id < 0:  # special nums are negative in qdb and in global_kin...
            r_attributes['special_number'] = int(qdb_ds.entry_id)

        # QDB IDs and doi:
        r_attributes['qdb_r_id'] = int(qdb_r.pk)
        r_attributes['qdb_ds_id'] = int(qdb_ds.pk)
        for source in qdb_ds.citations.all():
            if source.doi not in {None, 'None'}:
                r_attributes['doi'] = str(source.doi)
                break

        # comments (if adding "lumped reaction", I want the "resolved reaction" string in the comments)
        if master_reaction.pk != qdb_r.pk:
            resolved_reaction_str = '{} > {}'.format(
                ' + '.join([self._get_rp_name_from_qdb(rp) for rp in qdb_r.reactants.all()]),
                ' + '.join([self._get_rp_name_from_qdb(rp) for rp in qdb_r.products.all()])
            )
            r_attributes['comments'] = resolved_reaction_str

        self.add_reaction_from_attributes(reactants_names, products_names, qdb_r.pk, r_attributes, r_units)

    # ******************************************** ADDING CHEMISTRY ************************************************** #

    def add_chemistry_from_instance(self, ch):
        """Method to create and add all species and reactions xml elements from a Chemistry instance. The resulting
        xml file will NOT reflect any of the adhoc changes implemented to the chemistry instance ch. All the RP and
        Reaction ids will be preserved.
        :param ch: (Chemistry instance)
        :return: None
        """
        # store the original species order (ids)
        sp_ids = list(ch.get_species_id(special=True))
        for r in ch.get_reactions():
            self.add_reaction_from_instance(r)
        # restore the species order as it was defined in the Chemistry instance:
        self.species.sort(key=lambda x: sp_ids.index(int(x.attrib['id'])))

    def add_chemistry_from_xml(self, xml_path):
        """Method to add all reaction and rp xml elements from a previously saved xml file. This will only add species
        and reactions, not the xml meta node or any other nodes.
        :param xml_path: (str) path to the xml, from which to import rp and reaction nodes.
        :return: None
        """
        root_xml = ElementTree.parse(xml_path).getroot()
        species_xml = root_xml.find('species')
        reactions_xml = root_xml.find('reactions')

        for rp_xml in species_xml.iter('rp'):
            rp_name = rp_xml.find('name').text
            if rp_name not in self.species_dict:
                self.species_dict[rp_name] = rp_xml
                self.species.append(rp_xml)

        for r_xml in reactions_xml.iter('reaction'):
            self.reactions.append(r_xml)

    def add_chemistry_from_qdb(self, qdb_ch):
        """Method to create and add all the reaction and rp xml elements from the passed QDB Chemistry instance.
        This needs to be called from within the QDB django shell. The reaction xml element ids will be the same as
        QDB Reaction instance primary keys and the same goes for the species.
        :param qdb_ch: (qdb.chemistry.Chemistry instance)
        :return: None
        """
        qdb_reactions = qdb_ch.get_reactions()
        for qdb_r in qdb_reactions:
            self.add_reaction_from_qdb(qdb_r)

    def add_chemistry_from_hpem(self, dat_path, skip_special_numbers=True):
        """Method to add all the species and reaction from the supplied HPEM input file (.dat file). All the data
        are simply parsed from the dat file and nodes for RPs and Reactions are simply added to the XML, without any
        consistency checks.
        TODO: WARNING - currently it will NOT handle more than one return species, also the ! needs to be present
              at the end of each line, otherwise the lines will not be parsed! Also this has been implemented rather
              ad-hoc, so not tested thoroughly... It also will not recognise states - those will stay as part
              of the species name.
        """

        # define the renaming rules for the species in the dat file:
        def rename(hpem_name):
            rename_map = {
                'E': 'e'
            }
            return rename_map.get(hpem_name, hpem_name).replace('^', '+')

        # read the dat file:
        with open(dat_path, 'r') as dat_file:
            dat_lines = [line.strip() for line in dat_file.readlines()]
        sep_index = dat_lines.index('*')
        species_lines = dat_lines[:sep_index]
        reactions_lines = dat_lines[sep_index + 1:]

        single_val = r'\s*(\S+)\s*'  # one or more non-white-space char sandwiched between optional whitespaces
        single_val_ws = r'(.*)'  # more relaxed - value can contain white spaces

        # parse species:
        species_dict = OrderedDict()  # filled with dicts of species attributes, keyed by their names.
        pattern_sp = \
            re.compile(r'^{sv}:{sv};{sv}&{sv}]{sv}\[{sv}@{sv}!{svs}$'.format(sv=single_val, svs=single_val_ws))
        sp_columns = ['name', 'charge', 'mass', 'stick_coef', '_', 'return_coefficient', 'return_species', 'comments']
        sp_types = [rename, int, float, float, int, float, rename, str.strip]
        for line in species_lines:
            sp_dict = {sp_col: sp_type(val)
                       for sp_col, sp_type, val in zip(sp_columns, sp_types, pattern_sp.findall(line)[0])}
            if sp_dict['stick_coef']:
                sp_dict['ret_coefs'] = {sp_dict['return_species']: sp_dict['return_coefficient']}
            species_dict[sp_dict['name']] = sp_dict

        # parse reactions:
        reactions = []  # filled with dicts of reaction attributes
        species_names_present = set()  # filled with species names in the reactions
        arrh_a_units = {1: 's-1', 2: 'cm3.s-1', 3: 'cm6.s-1'}
        arrh_c_units = {True: 'eV', False: 'K'}
        pattern_r = \
            re.compile(r'^{svs}>{svs}:{sv};{sv}&{sv}]{sv}\[{sv}!{svs}$'.format(sv=single_val, svs=single_val_ws))
        r_columns = ['reactants', 'products', 'arrh_a', 'arrh_b', 'arrh_c', 'gas_heat_contrib', 'special_number',
                     'comments']
        r_types = [str.strip, str.strip, float, float, float, float, int, str.strip]
        for line in reactions_lines:
            r_dict = {r_col: r_type(val)
                      for r_col, r_type, val in zip(r_columns, r_types, pattern_r.findall(line)[0])}
            r_dict['reactants'] = [rename(sp) for sp in r_dict['reactants'].split(' + ')]
            r_dict['products'] = [rename(sp) for sp in r_dict['products'].split(' + ')]
            if not skip_special_numbers or (skip_special_numbers and r_dict['special_number'] in {-1, 1}):
                # add species for later adding:
                for sp in r_dict['reactants'] + r_dict['products']:
                    species_names_present.add(sp)
                # add the reaction for later adding:
                reactions.append(r_dict)
            else:
                print('WARNING: Reaction {} has a special number defined and therefore is IGNORED (not added)!'.format(
                    ' + '.join(r_dict['reactants']) + ' > ' + ' + '.join(r_dict['products'])
                ))

        # finally add all the species and reactions into the builder:
        sp_units = {'charge': 'e', 'mass': 'amu'}
        for sp, sp_dict in species_dict.items():
            if sp in species_names_present:
                self.add_rp_from_attributes(rp_attributes=sp_dict, rp_units=sp_units)
            else:
                print('WARNING: Species {} is not present in the reactions and is IGNORED (not added)'.format(
                    sp
                ))
        r_units = {'gas_heat_contrib': 'eV', 'arrh_a': 'N/A', 'arrh_c': 'N/A'}
        for r_dict in reactions:
            r_units.update(
                {'arrh_a': arrh_a_units[len(r_dict['reactants'])], 'arrh_c': arrh_c_units['e' in r_dict['reactants']]}
            )
            self.add_reaction_from_attributes(
                reactants_names=r_dict['reactants'],
                products_names=r_dict['products'],
                r_attributes=r_dict,
                r_units=r_units
            )

    # ********************************************** SORT AND SAVE *************************************************** #

    def sort_elements(self):
        """Method to sort all the species and reactions according to the soring scheme defined in the
        classification.ReactionClassifier class. Since the sorting method requires unique ids for all rp and reaction
        xml elements, if those are not unique, the ids will be silently reassigned (which will change ids of ALL
        rp/reaction xml elements, not only of those which are not unique.)
        :return: None
        """
        # need to have distinct ids prior to sorting:
        rp_ids = [sp.attrib['id'] for sp in self.species]
        r_ids = [r.attrib['id'] for r in self.reactions]
        species_ids_reassigned, reactions_ids_reassigned = False, False
        if len(rp_ids) != len(set(rp_ids)):
            self.assign_species_ids()
            species_ids_reassigned = True
        if len(r_ids) != len(set(r_ids)):
            self.assign_reactions_ids()
            reactions_ids_reassigned = True

        # need RP and Reaction instances to instantiate the ReactionClassifier with:
        species, reactions = XmlParser().get_species_and_reactions_from_xml_elements(self.species, self.reactions)
        sorting_class = ReactionClassifier(species, reactions)

        # sort the species:
        sorting_class.sort_species()  # sorted species according to the scheme in ReactionClassifier
        sorted_species = sorting_class.species
        sorted_species_ids = [str(sp.id) for sp in sorted_species]

        # sort the reactions:
        sorting_class.sort_reactions()
        # the reactions (in ReactionsClassifier namespace) are sorted, but not according to their categories - fix:
        classified_reactions = sorting_class.get_classified_reactions()  # dict of classes and categories - flatten out:
        sorted_reactions = []
        for r_type in classified_reactions:
            for r_category in classified_reactions[r_type]:
                sorted_reactions.extend(classified_reactions[r_type][r_category])
        sorted_reactions_ids = [str(r.id) for r in sorted_reactions]

        self.species.sort(key=lambda x: sorted_species_ids.index(x.attrib['id']))
        self.reactions.sort(key=lambda x: sorted_reactions_ids.index(x.attrib['id']))

        # and if I reassigned ids before sorting, might as well reassign them again, so they are all ascending:
        if species_ids_reassigned:
            self.assign_species_ids()
        if reactions_ids_reassigned:
            self.assign_reactions_ids()

    def assign_species_ids(self):
        """This method will change all species ids so they are all ascending in the order of rp xml elements in
        self.species, starting with 1, with two additional rules: electron gets 0 and 'M' gets 999.
        :return: None
        """
        # it's a good practice to assign electron e.id = 0 and M.id = 999
        special_ids = {'e': 0, 'M': 999}
        next_id = 1
        for rp_xml in self.species:
            rp_name = rp_xml.find('name').text
            if rp_name in special_ids:
                new_id = special_ids[rp_name]
            else:
                new_id = next_id
                next_id += 1
            rp_xml.attrib['id'] = str(new_id)

    def assign_reactions_ids(self):
        """This method will change all reactions ids so they are all ascending in the order of reactions xml elements
        in self.reactions, starting with 1.
        :return: None
        """
        for i, r_xml in enumerate(self.reactions):
            r_id = i + 1  # starting with 1
            r_xml.attrib['id'] = str(r_id)

    def assign_ids(self):
        """This method will cleanly reassign ids for all rps and reactions present here. It might be advantageous to
        first call the sort method before reassigning the ids. The rules of id reassignment are described in
        assign_species_ids and assign_reactions_ids methods' docstrings.
        :return: None
        """
        self.assign_species_ids()
        self.assign_reactions_ids()

    def dump_xml(self, path, overwrite=False):
        """This method takes all the rp and reaction xml elements present in self.species and self.reactions and
        creates the chemistry input .xml file which can then be parsed by XmlParser or as an input for initialising
        the Chemistry class.
        :param path: (str) path of the dumped xml file. If the file already exists, will raise the XmlError, unless
                     the overwrite parameter is set to True.
        :param overwrite: (bool) if to overwrite the existing path.
        :return: None
        """
        if os.path.isfile(path=path) and not overwrite:
            raise XmlError(
                'The xml file {} already exists! You may wish to specify the overwrite parameter!'.format(path)
            )
        root_xml = ElementTree.Element('chemistry')
        meta_xml = ElementTree.SubElement(root_xml, 'meta')  # node for some metadata for the chemistry
        ElementTree.SubElement(meta_xml, 'comments').text = ''  # comments node in the metadata - empty

        species_xml = ElementTree.SubElement(root_xml, 'species')
        reactions_xml = ElementTree.SubElement(root_xml, 'reactions')

        for rp_element in self.species:
            species_xml.append(rp_element)
        for r_element in self.reactions:
            reactions_xml.append(r_element)

        # dump to a file, but first format it with minidom:
        xmlstr = minidom.parseString(ElementTree.tostring(root_xml, encoding='utf-8')).toprettyxml(indent=4 * ' ')
        with open(path, 'w') as f:
            f.write(xmlstr)


class XmlParser(object):
    """
    This class handles parsing the chemistry .xml input files. Includes method to build RP and Reaction instances
    from the xml file. There is no error handling implemented and no chemistry consistency checks here. If the
    rp or reaction xml elements in the chemistry xml input file are not consistent (do not contain all the
    necessary sub-elements describing the attributes etc...), this will result in some errors raised when the
    RP and/or Reaction classes are instantiated...
    """

    def __init__(self):
        """Initialises the XmlBuilder instance.
        """
        pass

    @staticmethod
    def get_species_and_reactions_from_xml_elements(rp_xml_elements, reaction_xml_elements):
        """Method to construct arrays of species and reactions instances from their respective xml root elements.

        :param rp_xml_elements: iterable of ElementTree.Element tagged 'rp', with all relevant sub-elements
        :param reaction_xml_elements: iterable of ElementTree.Element tagged 'reaction', with all relevant sub-elements
        :return: list of RP, list of Reaction
        """

        # ***************************************** DEAL WITH SPECIES ************************************************ #
        species = []  # list of RP instances
        species_dict = dict()  # mapping from species_dict names to the RP object instances (used in Reaction__init__())

        for rp_xml in rp_xml_elements:

            rp_id = int(rp_xml.attrib['id'])

            skip_tags = {
                'ret_coefs'
            }  # set of all rp attributes, which will not be simply assigned values of their texts

            attributes = {element.tag: element.text for element in list(rp_xml) if element.tag not in skip_tags}
            # deal with the skipped tags:
            ret_coefs_xml = rp_xml.find('ret_coefs')
            if ret_coefs_xml is not None:
                ret_coefs = {r_xml.attrib['rp']: r_xml.text for r_xml in ret_coefs_xml.iter('r')}
                attributes['ret_coefs'] = ret_coefs

            # build units dict:
            units = {element.tag: element.attrib['unit'] for element in list(rp_xml) if 'unit' in element.attrib}

            rp = RP(rp_id=rp_id, attributes=attributes, units=units)
            species_dict[rp.get_name()] = rp
            species.append(rp)

        # ***************************************** DEAL WITH REACTIONS ********************************************** #
        reactions = []

        for reaction_xml in reaction_xml_elements:
            r_id = int(reaction_xml.attrib['id'])

            skip_tags = {
                'reactants', 'products'
            }  # set of all reaction attributes, which will not be simply assigned values of their texts

            attributes = {element.tag: element.text for element in list(reaction_xml) if element.tag not in skip_tags}
            # deal with the skipped tags:
            reactants_names = [sp_name.strip() for sp_name in reaction_xml.find('reactants').text.split(',')]
            products_names = [sp_name.strip() for sp_name in reaction_xml.find('products').text.split(',')]
            reactants = [species_dict[sp_name] for sp_name in reactants_names]
            products = [species_dict[sp_name] for sp_name in products_names]

            # build units dict:
            units = {element.tag: element.attrib['unit'] for element in list(reaction_xml) if 'unit' in element.attrib}

            reaction = Reaction(r_id=r_id, reactants=reactants, products=products, attributes=attributes, units=units)
            reactions.append(reaction)

        return species, reactions

    def get_species_and_reactions(self, xml_path):
        """Reads the xml file and builds a list of species (RP) and list of reactions (Reaction). Those
        can then be supplied to Chemistry.__init__. Will raise XmlAttributeError if the supplied path to xml file is
        not leading to any file.
        Being really lazy here, since it assumes correct xml structure, no error handling is implemented here.
        :param xml_path: (str) path to xml chemistry input file.
        :return: list of RP, list of Reaction
        """
        if not os.path.isfile(xml_path):
            raise XmlError('Invalid XML file path: {}!'.format(xml_path))

        tree = ElementTree.parse(xml_path)
        root = tree.getroot()

        species_xml = root.find('species')
        rp_xml_elements = species_xml.iter('rp')
        reactions_xml = root.find('reactions')
        reaction_xml_elements = reactions_xml.iter('reaction')

        species, reactions = self.get_species_and_reactions_from_xml_elements(rp_xml_elements, reaction_xml_elements)

        return species, reactions
