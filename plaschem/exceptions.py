
class RPInitError(Exception):
    pass


class RPAttributeError(Exception):
    pass


class ReactionInitError(Exception):
    pass


class ReactionAttributeError(Exception):
    pass


class ReactionValueError(Exception):
    pass


class ChemistryInitError(Exception):
    pass


class ChemistryAttributeError(Exception):
    pass


class ChemistryConsistencyError(Exception):
    pass


class ChemistrySpeciesNotPresentError(Exception):
    pass


class ChemistryReactionNotPresentError(Exception):
    pass


class ChemistryDisableError(Exception):
    pass


class XmlError(Exception):
    pass
