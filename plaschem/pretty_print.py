
import colorama
import pandas as pd


class ColorValue:
    """Custom class for wrapping DataFrame values between color escape characters - for
    color printing of pd.DataFrames...
    """

    def __init__(self, value, color='RESET'):
        self.value = value
        self.color = getattr(colorama.Fore, color)

    def set_value(self, value):
        self.value = value

    def set_color(self, color):
        self.color = getattr(colorama.Fore, color)

    def __str__(self):
        return self.color + self.value.__str__() + colorama.Fore.RESET

    def __repr__(self):
        return self.color + self.value.__repr__() + colorama.Fore.RESET

    def __eq__(self, other):
        return self.value == other

    def __hash__(self):
        return self.value.__hash__()


class PrettyPrint(object):
    """This is a wrapper providing a print() method for arrays returned by the Chemistry class instance. All
    the arrays have indices and columns of reactions/species ids and this class provides annatating the dataframes
    with species names and reaction strings as well as colorcoding rows and columns belonging to disabled species and
    reactions in RED.
    """
    def __init__(self, chemistry):
        """PrettyPrint initialiser
        :param chemistry: (Chemistry instance)
        """
        self.chemistry = chemistry  # this needs to be stored to access the info about disabled sp/r and names/strings

    def _get_pretty_dataframe(self, array, annotation):
        """This is a method to pretty-print a pd.DataFrame with indices/columns of reactions/species ids.
        Inserts row and column with labels (strings for reactions and names for species) and also color-codes rows and
        columns belonging to disabled species and reactions in RED.
        :param array: (pd.DataFrame) array to prettyprint
        :param annotation: (bool) if to insert annotations (strings for reactions and names for species)
        :return: pd.DataFrame - colorcoded and annotated array passed
        """

        # create a top level copy of the array where I'll wrap the objects in color and insert the annotations:
        to_print = array.copy()

        # colorfy the objects (values)
        for i in to_print.index:
            for c in to_print.columns:
                to_print.at[i, c] = ColorValue(to_print.at[i, c])

        # determine if index or columns belong to species or reactions
        structure = {}
        for keys in ['index', 'columns']:
            for candidates in ['species', 'reactions']:
                if set(getattr(to_print, keys)).issubset(set(getattr(self.chemistry, '_{}'.format(candidates)).index)):
                    structure[keys] = candidates
                    break
        # model structure after the loop: {'index': 'reactions', 'columns': 'species'}

        if annotation:
            # insert annotations rows and columns:
            index_labels_objects = getattr(self.chemistry, '_{}'.format(structure['index']))
            index_labels = pd.Series([ColorValue(str(a)) for a in index_labels_objects],
                                     index=index_labels_objects.index)
            columns_labels_objects = getattr(self.chemistry, '_{}'.format(structure['columns']))
            columns_labels = pd.Series([ColorValue(str(a)) for a in columns_labels_objects],
                                       index=columns_labels_objects.index)
            to_print.insert(0, -1, index_labels[to_print.index])
            columns_labels = pd.Series([ColorValue(''), ], index=[-1]).append(columns_labels)
            to_print = pd.DataFrame(columns_labels[to_print.columns]).T.append(to_print)

        # colorcode the disabled values:
        for i in getattr(self.chemistry, 'get_disabled_{}'.format(structure['index']))().index:
            # ids of disabled species/reactions in the to_print's index
            if i in to_print.index:
                for c in to_print.columns:
                    to_print.at[i, c].set_color('RED')
        for c in getattr(self.chemistry, 'get_disabled_{}'.format(structure['columns']))().index:
            # ids of disabled species/reactions in the to_print's columns
            if c in to_print.columns:
                for i in to_print.index:
                    to_print.at[i, c].set_color('RED')

        # remove the index and columns values for the label row and column:
        to_print.index = ['', ] + list(to_print.index)[1:]
        to_print.columns = ['', ] + list(to_print.columns)[1:]

        # colorify columns. Index cannot be colorfied, since it will mess up the alignment.
        to_print.columns = [ColorValue(c) for c in to_print.columns]

        return to_print

    def print(self, array, annotation=True):
        """This is a method to pretty-print an pandas array with indices/columns of reactions/species ids.
        Inserts row and column with labels (strings for reactions and names for species) and also color-codes rows and
        columns belonging to disabled species and reactions in RED.
        :param array: (pd.DataFrame) array to prettyprint
        :param annotation: (bool) if to insert annotations (strings for reactions and names for species)
        :return: pd.DataFrame or Series (as array)
        """
        if type(array) == pd.DataFrame:
            pretty_df = self._get_pretty_dataframe(array=array, annotation=annotation)
            print()
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
                print(pretty_df)
            print()
            return pretty_df
        else:
            raise NotImplementedError('PrettyPrint.print onlys defined for pd.DataFrame.')
