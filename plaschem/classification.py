
from collections import OrderedDict

import numpy as np


class ReactionClassifier(object):
    """
    This Class handles classification of a reactions and sorting of reactions and species from a chemistry instance.
    """
    classification_scheme = [
        ['ELECTRON IMPACT COLLISIONS', [
            'elastic and momentum transfer',
            'excitation',
            'dissociative excitation',
            'deexcitation',
            'ionization',
            'dissociative ionization',
            'attachment',
            'dissociative attachment',
            'three body attachment',
            'detachment',
            'recombination',
            'dissociative recombination',
            'three body recombination',
            'dissociation',
            'other electron energy loss channels',
            'unclassified electron impact collisions'
        ]],
        ['NEUTRAL - NEUTRAL COLLISIONS', [
            'elastic scattering',
            'association',
            'dissociation',
            'excitation',
            'dissociative excitation',
            'collisional quenching',
            'dissociative quenching',
            'excitation transfer',
            'ionization',
            'dissociative ionization',
            'associative ionization',
            'penning ionization',
            'interchange',
            'stripping',
            'unclassified neutral - neutral collisions'
        ]],
        ['POSITIVE ION - NEUTRAL COLLISIONS', [
            'resonant charge transfer',
            'charge transfer',
            'dissociative charge transfer',
            'unclassified positive ion - neutral collisions'
        ]],
        ['NEGATIVE ION - NEUTRAL COLLISIONS', [
            'resonant charge transfer',
            'charge transfer',
            'detachment',
            'associative detachment',
            'unclassified negative ion - neutral collisions'
        ]],
        ['ION - ION COLLISIONS', [
            'neutralization',
            'dissociative neutralization',
            'unclassified ion - ion collisions'
        ]],
        ['THREE BODY COLLISIONS', [
            'None',
        ]],
        ['RADIATIVE DECAY', [
            'None',
        ]],
        ['UNCLASSIFIED COLLISIONS', [
            'None',
        ]],
    ]

    elements_order = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'SC', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce',
        'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
        'Pt', 'Au', 'Hg', 'Ti', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'M'
    ]

    def __init__(self, species, reactions):
        """Initialises the reactions classifier. The species and reactions passed need to be self-consistent,
        the consistency of both iterables is not asserted here, but should be checked before instantiating the
        ReactionClassifier (for example by instantiating plaschem.chemistry.Chemistry from the same species and
        reactions.
        :param species: (iterable of RP instances)
        :param reactions: (iterable) of Reaction instances
        """
        self.species = list(species)
        self.reactions = list(reactions)

        # prepare the nested ordered dict of reactions types and categories to be populated.
        self._classified_reactions = OrderedDict()

        # nested dict storing number maximum lengths of species names for each category from classified_reactions:
        self._alignments = dict()

        self._memo_species_index = dict()  # this is just a lookup table to speed up the self.species indices
        self._memo_reactions_classifications = dict()  # type and category of each reaction keyed by r.id
        self._memo_reactions_sorted_species_names = dict()  # sorted reacts. and prods names in ((), ()) keyed by r.id

    def set_new_chemistry(self, species, reactions):
        """Method to set a new chemistry to the instance, need to go through this method, since all the memos need to
        be reset.
        :param species: (iterable of RP instances)
        :param reactions: (iterable) of Reaction instances
        :return: None
        """
        self.species = list(species)
        self.reactions = list(reactions)
        self._classified_reactions = OrderedDict()
        self._alignments = dict()
        self._memo_species_index = dict()
        self._memo_reactions_classifications = dict()
        self._memo_reactions_sorted_species_names = dict()

    def _get_species_sort_key(self, sp):
        """This method returns a sorting key for species. Less constituent elements first, lighter elements first,
        ground state neutral before positive ion, before negative ion, before excited species. Works with elements up
        to atomic number 99, maximum allowed number of any particular atoms is 9 and maximum allowed number of
        distinct elements is 4.

        :param sp: (plaschem.species.RP instance)
        :return: float() sorting key for species
        """
        score = 0.

        # artificially push 'M' to the last place (lowest score):
        if sp.get_name() == 'M':
            score = 0.1
            return 1/score

        charge = sp.get_charge()

        # positive charge bit (least significant bit):
        bit_val = 1e1 - 1
        step = 1e0
        if charge > 0:
            bit_val -= step * charge
        score += bit_val

        # negative charge bit:
        bit_val = 1e2 - 1
        step = 1e1
        if charge < 0:
            bit_val += step * charge
        score += bit_val

        # excitation bit:
        bit_val = 1e3 - 1
        step = 1e2
        if sp.get_state() is not None:
            bit_val -= step
        score += bit_val

        elements = list(sp.get_atoms().keys())
        if len(elements) > 3:
            raise NotImplementedError('Only species with les than 4 elements are implemented for sorting!')

        if len(elements) == 3:
            # 3rd element count bit:
            bit_val = 1e4 - 1
            step = 1e3
            bit_val -= step*sp.get_atoms()[elements[2]]
            score += bit_val

            # 3rd element atomic number bit:
            bit_val = 1e6 - 1
            step = 1e4
            bit_val -= step*(self.elements_order.index(elements[2]) + 1)
            score += bit_val

        if len(elements) >= 2:
            # 2nd element count bit:
            bit_val = 1e7 - 1
            step = 1e6
            bit_val -= step*sp.get_atoms()[elements[1]]
            score += bit_val

            # 2nd element atomic number bit:
            bit_val = 1e9 - 1
            step = 1e7
            bit_val -= step*(self.elements_order.index(elements[1]) + 1)
            score += bit_val

        if len(elements) >= 1:
            # 1st element count bit:
            bit_val = 1e10 - 1
            step = 1e9
            bit_val -= step*sp.get_atoms()[elements[0]]
            score += bit_val

            # 1st element atomic number bit:
            bit_val = 1e12 - 1
            step = 1e10
            bit_val -= step*(self.elements_order.index(elements[0]) + 1)
            score += bit_val

        # number of elements bit (most significant bit):
        bit_val = 1e13 - 1
        step = 1e12
        bit_val -= step*len(elements)
        score += bit_val

        return 1/score

    def sort_species(self):
        """Sorts species according to the sorting scheme defined by the self._get_species_sort_key method. Directly
        modifies the self.species list
        :return: None
        """
        self.species.sort(key=self._get_species_sort_key)
        # reset dependencies:
        self._classified_reactions = OrderedDict()
        self._memo_species_index = dict()

    def _species_index(self, sp):
        """This is just getting index of a species inside the self.species list. Only here to optimise the lookup
        time and utilise memoization.
        :param sp: (RP instance)
        :return: (int) index (position) of the sp in self.species.
        """
        try:
            return self._memo_species_index[sp.id]
        except KeyError:
            self._memo_species_index[sp.id] = self.species.index(sp)
            return self._memo_species_index[sp.id]

    def _get_reaction_sort_key(self, r):
        """This methods builds a reactions sorting key building up on sorting scheme of tuples of integers. The sorting
        of the reactions will be influenced by the order of the species in self.species. Might want to call
        self.sort_species() first.
        :param r: (Reaction instance)
        :return: (tuple) nested tuples of sorted indices of species in reactants and products
                 ((i_r1, i_r2, i_r3, ...), (i_p1, i_p2, i_p3, ...)), where i_sp is index of the species sp inside the
                 self.species list and all reactants and products indices are ascending (more like non-descending).
        """
        sorted_reactants = sorted(r.get_reactants(), key=self._species_index)
        sorted_products = sorted(r.get_products(), key=self._species_index)
        self._memo_reactions_sorted_species_names[r.id] = \
            (tuple([sp.get_name() for sp in sorted_reactants]), tuple([sp.get_name() for sp in sorted_products]))
        return (tuple([self._species_index(sp) for sp in sorted_reactants]),
                tuple([self._species_index(sp) for sp in sorted_products]))

    def sort_reactions(self):
        """Sorts reactions according to the sorting scheme defined by the self._get_reaction_sort_key method. Directly
        modifies the self.reactions list
        :return: None
        """
        self.reactions.sort(key=self._get_reaction_sort_key)
        # reset dependencies:
        self._classified_reactions = OrderedDict()

    def _init_classified_reactions_dict(self):
        """Prepares an empty ordered dict of types and categories to be populated with reactions.
        :return: None
        """
        self._classified_reactions = OrderedDict()
        for r_type, r_categories in self.classification_scheme:
            self._classified_reactions[r_type] = OrderedDict()
            for r_category in r_categories:
                self._classified_reactions[r_type][r_category] = list()

    def _classify(self, reaction):
        """This method will classify the reaction according to a custom scheme defined above in cls attribute and assign
        a "type" and "category" to the reaction. Both are strings.

        :return: (tuple): (type, category) of strings.
        """
        # memoization:
        try:
            return self._memo_reactions_classifications[reaction.id]
        except KeyError:
            pass

        r_type = 'UNCLASSIFIED COLLISIONS'
        r_category = 'None'

        reactants = reaction.get_reactants()
        products = reaction.get_products()

        # M of course does not define charge, for the purpose of the classification, treat 'M' as neutral:
        charges_r = np.array([sp.get_charge() if sp.get_charge() is not None else 0 for sp in reactants])
        charges_p = np.array([sp.get_charge() if sp.get_charge() is not None else 0 for sp in products])

        stoich_coefficients = reaction.get_stoich_coefs(method='net')
        el_stoich_coef = 0.
        if 'e' in reaction.get_rp_names_set(method='all'):
            el_id = [sp.id for sp in reactants+products if sp.get_name() == 'e'][0]
            el_stoich_coef = stoich_coefficients[el_id]

        if reaction.is_electron_process():
            r_type = 'ELECTRON IMPACT COLLISIONS'
            r_category = 'unclassified electron impact collisions'
            col_partners = [sp for sp in reactants if sp.get_name() != 'e']
            heavy_products = [sp for sp in products if sp.get_name() != 'e']
            col_partner = col_partners[0]  # used only for the case of only one collision partner
            if len(reactants) < 2:  # this is a screw case, will go to unclassified
                pass
            elif len(reactants) > 3:  # this will lend in unclassified as well
                pass
            elif len(reactants) == 3:  # three body electron collisions
                col_partners_charges = [sp.get_charge() for sp in col_partners if sp.get_charge() is not None]
                heavy_products_charges = [sp.get_charge() for sp in heavy_products if sp.get_charge() is not None]
                # if the neutral ground states are equal on both sides:
                if not len(set([sp.strip_charge(sp.get_stateless()) for sp in col_partners]) -
                           set([sp.strip_charge(sp.get_stateless()) for sp in heavy_products])):
                    if sum(col_partners_charges) > sum(heavy_products_charges):
                        if sum(col_partners_charges) > 0:
                            r_category = 'three body recombination'
                        elif sum(heavy_products_charges) < 0:
                            r_category = 'three body attachment'

            # the rest are definitely 2-body electron collisions:
            elif el_stoich_coef > 0:  # more electrons on the RHS
                if len(col_partners) == len(heavy_products):
                    if col_partner.get_charge() == 0:
                        r_category = 'ionization'  # turning neutral to positive ion
                    elif col_partner.get_charge() < 0:
                        r_category = 'detachment'  # turning negative ion into neutral
                elif len(col_partners) < len(heavy_products):
                    if col_partner.get_charge() == 0:
                        r_category = 'dissociative ionization'
            elif el_stoich_coef == 0:  # electrons are preserved
                if reaction.is_elastic_process():
                    r_category = 'elastic and momentum transfer'
                elif len(col_partners) == len(heavy_products):
                    heavy_product = heavy_products[0]
                    if col_partner.is_stateless() and not heavy_product.is_stateless():
                        r_category = 'excitation'
                    elif not col_partner.is_stateless() and heavy_product.is_stateless():
                        r_category = 'deexcitation'
                    elif not col_partner.is_stateless() and not heavy_product.is_stateless():
                        if reaction.get_el_en_loss() > 0:
                            r_category = 'excitation'
                        else:
                            r_category = 'deexcitation'
                    elif col_partner == heavy_product:
                        r_category = 'other electron energy loss channels'
                elif len(col_partners) < len(heavy_products):
                    products_states = [not sp.is_stateless() for sp in heavy_products]
                    if True in products_states:
                        r_category = 'dissociative excitation'
                    else:
                        r_category = 'dissociation'
            elif el_stoich_coef < 0:  # less electrons on the RHS
                if len(col_partners) == len(heavy_products):
                    if col_partner.get_charge() == 0:
                        r_category = 'attachment'
                    elif col_partner.get_charge() > 0:
                        r_category = 'recombination'
                elif len(col_partners) < len(heavy_products):
                    if col_partner.get_charge() == 0:
                        r_category = 'dissociative attachment'
                    elif col_partner.get_charge() > 0:
                        r_category = 'dissociative recombination'

        elif len(reactants) == 3:
            r_type = 'THREE BODY COLLISIONS'

        elif len(reactants) == 1:
            if len(products) == 1:
                r_type = 'RADIATIVE DECAY'

        elif len(reactants) == 2:
            if (charges_r == 0).all():
                r_type = 'NEUTRAL - NEUTRAL COLLISIONS'
                r_category = 'unclassified neutral - neutral collisions'
                reactants_states = [not sp.is_stateless() for sp in reactants]
                products_states = [not sp.is_stateless() for sp in products]
                if reaction.is_elastic_process():
                    r_category = 'elastic scattering'
                elif len(reactants) > len(products):
                    r_category = 'association'
                elif el_stoich_coef > 0:
                    if len(reactants) == len(products) - el_stoich_coef:
                        # penning ionization:
                        if sum(reactants_states) > 0 and sum(products_states) == 0:
                            r_category = 'penning ionization'
                        elif sum(reactants_states) == sum(products_states) == 0:
                            r_category = 'ionization'
                    elif len(reactants) < len(products) - el_stoich_coef:
                        r_category = 'dissociative ionization'
                    elif len(reactants) > len(products) - el_stoich_coef:
                        r_category = 'associative ionization'
                elif el_stoich_coef == 0 and len(reactants) < len(products):
                    # the projectile needs to be on the LHS also!
                    if len(set(sp.get_name() for sp in reactants) &
                           set(sp.get_name() for sp in products)):
                        if not charges_p.any():
                            if sum(reactants_states) == sum(products_states):
                                r_category = 'dissociation'
                            elif sum(reactants_states) < sum(products_states):
                                r_category = 'dissociative excitation'
                            elif sum(reactants_states) > sum(products_states):
                                r_category = 'dissociative quenching'
                elif len(reactants) == len(products):
                    if sum(reactants_states) < sum(products_states):
                        r_category = 'excitation'
                    elif sum(reactants_states) > sum(products_states):
                        r_category = 'collisional quenching'
                    elif sum(reactants_states) == sum(products_states) and sum(reactants_states) > 0:
                        if set([sp.get_stateless() for sp in reactants]) == \
                                set([sp.get_stateless() for sp in products]):
                            r_category = 'excitation transfer'
                    elif not len(
                            set([sp.get_stateless() for sp in reactants]) -
                            set([sp.get_stateless() for sp in reactants])
                    ):
                        r_category = 'interchange'

            elif ((charges_r > 0) + (charges_r == 0)).all():
                r_type = 'POSITIVE ION - NEUTRAL COLLISIONS'
                r_category = 'unclassified positive ion - neutral collisions'
                r_ion = reactants[np.where(charges_r > 0)[0][0]]
                r_neutral = reactants[np.where(charges_r == 0)[0][0]]
                if len(products) == len(reactants):
                    p_ion = products[np.where(charges_p > 0)[0][0]]
                    p_neutral = products[np.where(charges_p == 0)[0][0]]
                    if r_ion.get_name().strip('+') == p_neutral.get_name() and \
                            r_neutral.get_name() == p_ion.get_name().strip('+'):
                        if r_neutral.get_name() == p_neutral.get_name() and r_ion.get_name() == p_ion.get_name():
                            r_category = 'resonant charge transfer'
                        else:
                            r_category = 'charge transfer'
                elif len(products) > len(reactants):
                    if r_neutral.get_name() + '+' in [sp.get_name() for sp in products]:
                        r_category = 'dissociative charge transfer'

            elif ((charges_r < 0) + (charges_r == 0)).all():
                r_type = 'NEGATIVE ION - NEUTRAL COLLISIONS'
                r_category = 'unclassified negative ion - neutral collisions'
                r_ion = reactants[np.where(charges_r < 0)[0][0]]
                r_neutral = reactants[np.where(charges_r == 0)[0][0]]
                if el_stoich_coef > 0:
                    if len(reactants) == len(products) - el_stoich_coef:
                        r_category = 'detachment'
                    elif len(reactants) > len(products) - el_stoich_coef:
                        r_category = 'associative detachment'
                elif len(products) == len(reactants):
                    p_ion = products[np.where(charges_p < 0)[0][0]]
                    p_neutral = products[np.where(charges_p == 0)[0][0]]
                    if r_ion.get_name().strip('-') == p_neutral.get_name() and \
                            r_neutral.get_name() == p_ion.get_name().strip('-'):
                        if r_neutral.get_name() == p_neutral.get_name() and r_ion.get_name() == p_ion.get_name():
                            r_category = 'resonant charge transfer'
                        else:
                            r_category = 'charge transfer'

            elif ((charges_r < 0) + (charges_r > 0)).all():
                if (charges_p == 0).all():
                    r_type = 'ION - ION COLLISIONS'
                    r_category = 'unclassified ion - ion collisions'
                    if (charges_p == 0).all():
                        if len(reactants) == len(products):
                            r_category = 'neutralization'
                        elif len(reactants) < len(products):
                            r_category = 'dissociative neutralization'

        # save into the memo table:
        self._memo_reactions_classifications[reaction.id] = (r_type, r_category)
        return r_type, r_category

    def get_classified_reactions(self):
        """This method build a nested ordered dict with keys of reactions types and categories as defined by the
        classification_scheme class attribute populated by lists of reactions in each category.
        The order of the reactions in each category is determined by the order of the reactions in self.reactions.
        These can be sorted beforehand by self.sort_reactions, which will in turn be affected by the order of the
        species in self.species, which itself can be sorted before calling self.sort_reactions by calling
        self.sort_species.

        :return: (OrderedDict od OrderedDicts of lists of Reaction instances) with first key of reaction type and
                 second key of reaction category as defined in self.classification_scheme and self._classify(reaction)
        """
        # prepare an empty nested ordered dict to be populated with reactions from self.reactions
        self._init_classified_reactions_dict()

        # populate the dict
        for r in self.reactions:
            r_type, r_category = self._classify(r)
            self._classified_reactions[r_type][r_category].append(r)

        # delete empty categories:
        keys_to_del = []
        for r_type in self._classified_reactions:
            for r_category in self._classified_reactions[r_type]:
                if not len(self._classified_reactions[r_type][r_category]):
                    keys_to_del.append((r_type, r_category))
        for r_type, r_category in keys_to_del:
            del self._classified_reactions[r_type][r_category]
        # delete empty types:
        keys_to_del = []
        for r_type in self._classified_reactions:
            if not len(self._classified_reactions[r_type]):
                keys_to_del.append(r_type)
        for r_type in keys_to_del:
            del self._classified_reactions[r_type]
        # return
        return self._classified_reactions

    def get_aligned_reaction_string(self, r):
        """This method returns a string for reaction r in a way that all reactions in one category
        (in self._classified_reactions) are nicely aligned below each other with all reactants and products starting
        at the same line position. WARNING: self.get_classified_reactions needs to be run before this one!

        :param r: (Reaction instance)
        :return: (str) reactions string useful for pretty printing reactions in aligned lines
        """
        r_type, r_category = self._classify(r)
        try:
            max_reactants_names_lengths, max_products_names_lengths = self._alignments[r_type][r_category]
        except KeyError:
            max_reactants_names_lengths = []
            max_products_names_lengths = []
            for reaction in self._classified_reactions[r_type][r_category]:
                try:
                    sorted_reactants_names, sorted_products_names = \
                        self._memo_reactions_sorted_species_names[reaction.id]
                except KeyError:
                    sorted_reactants = sorted(reaction.get_reactants(), key=self._species_index)
                    sorted_products = sorted(reaction.get_products(), key=self._species_index)
                    sorted_reactants_names = [sp.get_name() for sp in sorted_reactants]
                    sorted_products_names = [sp.get_name() for sp in sorted_products]
                    self._memo_reactions_sorted_species_names[reaction.id] = \
                        (sorted_reactants_names, sorted_products_names)
                for i, reactant_name in enumerate(sorted_reactants_names):
                    try:
                        if max_reactants_names_lengths[i] < len(reactant_name):
                            max_reactants_names_lengths[i] = len(reactant_name)
                    except IndexError:
                        max_reactants_names_lengths.append(len(reactant_name))
                for i, product_name in enumerate(sorted_products_names):
                    try:
                        if max_products_names_lengths[i] < len(product_name):
                            max_products_names_lengths[i] = len(product_name)
                    except IndexError:
                        max_products_names_lengths.append(len(product_name))
            try:
                self._alignments[r_type][r_category] = (max_reactants_names_lengths, max_products_names_lengths)
            except KeyError:
                self._alignments[r_type] = dict()
                self._alignments[r_type][r_category] = (max_reactants_names_lengths, max_products_names_lengths)

        # now I have max_reactants/products_names_lengths arrays which can be used to align reactants and products:
        sorted_reactants_names, sorted_products_names = self._memo_reactions_sorted_species_names[r.id]

        lhs_str = ' + '.join([
            '{:<{}}'.format(sorted_reactants_names[i], max_len)
            if i < len(sorted_reactants_names) else '{:<{}}'.format('', max_len)
            for i, max_len in enumerate(max_reactants_names_lengths)
        ])
        rhs_str = ' + '.join([
            '{:<{}}'.format(sorted_products_names[i], max_len)
            if i < len(sorted_products_names) else '{:<{}}'.format('', max_len)
            for i, max_len in enumerate(max_products_names_lengths)
        ])
        r_str = ' > '.join([lhs_str, rhs_str])

        return r_str
