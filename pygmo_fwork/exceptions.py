
class GlobalModelInitialGuessError(Exception):
    pass


class GlobalModelAttributeError(Exception):
    pass


class BackendNoRunError(Exception):
    pass


class BackendRunUnsuccessfulError(Exception):
    pass


class GlobalModelLogError(Exception):
    pass


class ConfigError(Exception):
    def __init__(self, msg):
        ammended_message = '\n'.join([
            msg,
            100*'.',
            'Configuration file is expected at "~/.pygmofwork".',
            'The file needs to be in a .yaml format with the following structure: ',
            '\tworkspace: absolute_path/to/workspace_directory',
            '\tglobalkin_bin: absolute_path/to/global_kin/binary_file  # this is only needed for PyGKin backend...',
            'The user needs to have write rights for the workspace directory, or for the parent directory!',
            100*'.'
        ])
        super().__init__(ammended_message)


class ResultsAttributeError(Exception):
    pass


class ResultsNotFoundError(Exception):
    pass
