
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class SolutionsDiff(object):
    """
    This class is specifically implemented for metrology of error between model results with full and reduced
    kinetic mechanisms. Methods are implemented to return mixed (combination of relative and absolute) error for a
    single output at a single time, or across multiple times and outputs.
    """
    def __init__(self, sol_full):
        """Instantiate the class with the full mechanism model solution with a full kinetic mechanism.
        :param sol_full: (pd.DataFrame) time-dependent solution of a global model.
        """
        self._validate_solution(sol_full)
        self.sol_full = sol_full

    def reset(self, sol_full):
        """Method to reload the base solution (of the full model) and to reset all the memo tables.
        :param sol_full: (pd.DataFrame) time-dependent solution of a global model.
        :return: None
        """
        self.__init__(sol_full)

    @staticmethod
    def _array_rms(array):
        return np.sqrt(np.mean(np.square(array)))

    @staticmethod
    def _validate_solution(solution):
        """Validates a passed solution dataframe.
        :param solution: (pd.DataFrame) time-dependent solution of a global model. Needs to have 't' column with times
        :return: None
        """
        assert isinstance(solution, pd.DataFrame), 'Solution needs to be pandas.DataFrame'
        assert 't' in solution.columns, 'Solution needs to containt time column "t"'

    @staticmethod
    def _get_result_in_time(solution, model_output, time):
        """Gets a model result for a certain model_output at an arbitrary time from the solution DataFrame.
        :param solution: (pd.DataFrame) validated solution
        :param model_output: (str) label of a specific solution column (species name, 'p', 'e', 'Te' etc...)
        :param time: (float) time inside the solution time ranges
        :return: (float) value of the model output in the solution at the time.
        """
        assert model_output in solution.columns, 'Output {} not in the passed solution!'.format(model_output)
        assert solution['t'].iloc[0] <= time <= solution['t'].iloc[-1], \
            'Time t={}s outside of the solution time frame!'.format(time)
        # interpolate the solution[model_output] to get the value at passed time:
        return float(interp1d(solution['t'], solution[model_output])(time))

    def get_output_errors(self, sol_red, model_output, times):
        """Returns mixed relative error for a single model output for each time from passed times array. The error is
        between the passed reduced solution sol_red and the self.sol_full base solution.
        :param sol_red: (pd.DataFrame) time-dependent solution of a global model with a reduced kinetic mechanism.
        :param model_output: (str) label of a specific solution column (species name, 'p', 'e', 'Te' etc...)
        :param times: (iterable) of float times. All need to be in time range of both solutions
        :return: (array) of (flaot) a mixed relative error between the full and reduced models for a single model
                 output and for each time from passed times array.
        """
        # full model solution values of the model_output at times:
        vals_full = np.array([self._get_result_in_time(self.sol_full, model_output, time) for time in times])
        # reduced model solution values of the model_output at times:
        vals_red = np.array([self._get_result_in_time(sol_red, model_output, time) for time in times])
        # mixed relative error function:
        deltas_output = 2*(vals_red - vals_full) / (vals_full + max(vals_full))
        return deltas_output

    def get_output_error_max(self, sol_red, model_output, times):
        """Returns the maximal mixed relative error for a single model output across the times array.
        :param sol_red: (pd.DataFrame) time-dependent solution of a global model with a reduced kinetic mechanism.
        :param model_output: (str) label of a specific solution column (species name, 'p', 'e', 'Te' etc...)
        :param times: (iterable) of float times. All need to be in time range of both solutions
        :return: (flaot) maximal mixed output error.
        """
        return max(np.absolute(self.get_output_errors(sol_red, model_output, times)))

    def get_output_error_rms(self, sol_red, model_output, times):
        """Returns the RMS of mixed relative errors for a single model output across the times array.
        :param sol_red: (pd.DataFrame) time-dependent solution of a global model with a reduced kinetic mechanism.
        :param model_output: (str) label of a specific solution column (species name, 'p', 'e', 'Te' etc...)
        :param times: (iterable) of float times. All need to be in time range of both solutions
        :return: (flaot) RMS of mixed output errors.
        """
        return self._array_rms(self.get_output_errors(sol_red, model_output, times))

    def get_error_max(self, sol_red, important_outputs, times):
        """Returns the maximal mixed relative error across multiple important model outputs and times.
        :param sol_red: (pd.DataFrame) time-dependent solution of a global model with a reduced kinetic mechanism.
        :param important_outputs: (array) of (str) of several model outputs
        :param times: (iterable) of float times. All need to be in time range of both solutions
        :return: (flaot) maximal mixed relative error across all times and important outputs.
        """
        return max([self.get_output_error_max(sol_red, model_output, times) for model_output in important_outputs])

    def get_error_rms(self, sol_red, important_outputs, times):
        """Returns the RMS of mixed relative errors across multiple important model outputs and times.
        :param sol_red: (pd.DataFrame) time-dependent solution of a global model with a reduced kinetic mechanism.
        :param important_outputs: (array) of (str) of several model outputs
        :param times: (iterable) of float times. All need to be in time range of both solutions
        :return: (flaot) RMS of mixed relative errors across all times and important outputs.
        """
        return self._array_rms(
            np.array([self.get_output_error_rms(sol_red, model_output, times) for model_output in important_outputs])
        )
