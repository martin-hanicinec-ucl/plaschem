
from collections import OrderedDict

import pandas as pd


class Texify:
    """
    A simple namespace which handles building LaTeX representations of species from their plain names.
    """
    element_symbols = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
        'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
        'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
        'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
        'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
        'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
        'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts',
        'Og'
    ]

    @staticmethod
    def _texify_ground_state(ordinary_formula):
        """Returns LaTeX representation of an ordinary formula (which does not have any brackets included!)
        :param ordinary_formula: (str)
        :return (str)
        """
        # general subscripts and superscripts:
        # ok, this is really ugly!!
        of = ordinary_formula
        # + and - signs and * sign are superscribed, if found after number, or '}' char (end of \mathrm{...})
        for end in Texify.element_symbols + list(range(10)):
            for s in ['****', '***', '**', '+++', '++', '---', '--']:
                of = of.replace('{}{}'.format(end, s), '{}^{{{}}}'.format(end, s))
            for s in ['+', '-', '*']:
                of = of.replace('{}{}'.format(end, s), '{}^{}'.format(end, s))
        # numbers behind elements are subscribed
        for a in Texify.element_symbols:
            for n in range(10, 20):
                of = of.replace('{}{}'.format(a, n), '{}_{{{}}}'.format(a, n))
            for n in range(1, 10):
                of = of.replace('{}{}'.format(a, n), '{}_{}'.format(a, n))
        # fix the multiple ionisation:
        for a in ['^{++}', '^{+++}', '^{--}', '^{---}']:
            of = of.replace(a, '^{{{}{}}}'.format(len(a[2:-1]), a[2]))
        # everything except numbers, '+', '-', '*' or '{', '}' is wrapped in \mathrm...
        alphas = [char.isalpha() for char in of]
        in_env = False
        for i in reversed(range(len(alphas))):
            if alphas[i]:
                if not in_env:
                    # insert the end of the mathrm env:
                    of = of[:i+1] + '}' + of[i+1:]
                    in_env = True
                if i == 0 and in_env:
                    of = r'\mathrm{' + of
            else:
                if in_env:
                    # insert the beginning of the mathrm env:
                    of = of[:i+1] + r' \mathrm{' + of[i+1:]
                    in_env = False

        return of

    @staticmethod
    def texify_species(rp_name):
        """Returns LaTeX representation of the RP name. This method is horribly coded and not general at all! It is
        bound to fail in many cases! Use with care!
        Returns string which should live in a math environment, but without '$' symbols!
        If the rp_name is a name of a stateful species, the state is expected to either be from
        {'*', '**', '***', '****'} or enclosed in a single pair of round brackets. The closing bracket needs to be
        the last character of the name. The insides of the closing brackets are not manipulated at all.
        Outside the possible brackets, every alpha character is put inside mathrm environment, while each number
        following an element will be subscribed and each +, -, or '*' following an element or number will be
        superscribed.
        Any other case will fail (likely produce a bullshit string, not likely raising an exception).
        :param rp_name: (str) RP name (RP.get_name())
        :return (str)
        """
        # first find brackets and preserve everything inside brackets:
        lbs = [i for i, ltr in enumerate(rp_name) if ltr == '(']
        rbs = [i for i, ltr in enumerate(rp_name) if ltr == ')']
        assert len(lbs) == len(rbs), 'TeXify error! Brackets not closed in species name!'
        assert len(lbs) in {0, 1}, 'TeXify error! More than one bracket!'
        if len(lbs) == 1:
            assert lbs[0] < rbs[0], 'TeXify error! Inconsistent bracket!'
            assert rbs[0] == len(rp_name) - 1, 'TeXify error! Bracket not at the end of species name!'

        # texify the ordinary formula outside the bracket:
        if len(lbs) == 0:
            return Texify._texify_ground_state(rp_name)
        elif len(lbs) == 1:
            return '{} {}'.format(Texify._texify_ground_state(rp_name[:lbs[0]]), rp_name[lbs[0]:])

        raise AssertionError('TeXify error!')

    @staticmethod
    def texify_rate_coefficient(reaction, si_units=True, fractype='/'):
        """Returns the LaTeX equation of the Arrhenius formula with the Reaction kinetic data.
        :param reaction: (Reaction instance)
        :param si_units: (bool)
        :return (str) LaTeX representation for math environment without the $ symbols
        """
        # pre-exponential factor:
        a_exp = '{:.2E}'.format(reaction.get_arrh_a(si_units=si_units))
        a, exp = a_exp.split('E')
        a = a.rstrip('.0')
        if int(exp) != 0:
            tex_a = r'{} \times 10^{{{}}}'.format(a, int(exp))
        else:
            tex_a = a

        # temperature power factor:
        if reaction.get_arrh_b() == 0:
            tex_b = ''
        else:
            exp = str(round(reaction.get_arrh_b(), 2)).rstrip('.0')
            if reaction.is_electron_process():
                tex_b = r' T_\mathrm{{e}}^{{{}}}'.format(exp)
            else:
                if fractype == 'dfrac':
                    tex_b = r' \left( \dfrac{{\mathrm{{T_g}}}}{{300}} \right)^{{{}}}'.format(exp)
                else:
                    tex_b = r' \left( \mathrm{{T_g}}/300 \right)^{{{}}}'.format(exp)

        # exponential factor:
        if reaction.get_arrh_c() == 0:
            tex_c = ''
        else:
            act_en = str(round(abs(reaction.get_arrh_c()), 2)).rstrip('.0')
            sign = '-' if reaction.get_arrh_c() > 0 else ''
            if reaction.is_electron_process():
                if fractype == 'dfrac':
                    tex_c = r' \exp \left( {}\dfrac{{{}}}{{T_\mathrm{{e}}}} \right)'.format(sign, act_en)
                else:
                    tex_c = r' \exp \left( {}{}/T_\mathrm{{e}} \right)'.format(sign, act_en)
            else:
                if fractype == 'dfrac':
                    tex_c = r' \exp \left( {}\dfrac{{{}}}{{\mathrm{{T_g}}}} \right)'.format(sign, act_en)
                else:
                    tex_c = r' \exp \left( {}{}/\mathrm{{T_g}} \right)'.format(sign, act_en)

        return tex_a + tex_b + tex_c


class TexTable(object):
    """
    Simple re-definition of the OrderedDict structure, only with a custom __str__ method, returning a string
    representation of a single latex table line.
    """

    def __init__(self, table_type, spacing=1.5, fontsize='tiny'):
        self.dataframe = pd.DataFrame(dtype='str')
        self.decorators = pd.DataFrame(dtype='object')
        self.notes = OrderedDict()
        self.caption = ''
        self.table_type = table_type  # from {'longtable', 'table'}
        self.label = ''
        self.spacing = spacing  # only for table
        self.fontsize = fontsize  # only for the longtable, tabular is scaled to page width

    def load_dataframe(self, dataframe):
        self.dataframe = dataframe
        self.decorators = pd.DataFrame(lambda x: x, index=self.dataframe.index, columns=self.dataframe.columns)

    def add_label(self, label):
        self.label = label

    def add_note(self, superscript, note):
        self.notes[superscript] = note

    def add_caption(self, caption_str):
        self.caption = caption_str

    def add_decorator(self, index, column, decorator_func):
        self.decorators.at[index, column] = decorator_func

    # build the text:
    def _get_caption_lines(self):
        lines = list()
        lines.append(self.caption + r' \\')
        for superscript, note in self.notes.items():
            lines.append(r'$^\mathrm{{{}}}$ {} \\'.format(superscript, note))
        return lines

    def _get_header_line(self, table_type):
        if table_type == 'table':
            line = ' & '.join([r'\textbf{{{}}}'.format(col) for col in self.dataframe.columns]) + r' \\'
        elif table_type == 'longtable':
            line = ' & '.join(
                [r'\multicolumn{{1}}{{l}}{{\textbf{{{}}}}}'.format(col) for col in self.dataframe.columns]
            ) + r' \\'
        else:
            raise ValueError('Unrecognised table type!')
        return line

    def _get_body_lines(self):
        lines = list()
        for index in self.dataframe.index:
            lines.append(
                ' & '.join(
                    [self.decorators.at[index, col](self.dataframe.at[index, col]) for col in self.dataframe.columns]
                ) + r' \\'
            )
        return lines

    @staticmethod
    def _get_sep_lines():
        lines = list()
        lines.append(r'\noalign{\vskip 1mm}')
        lines.append(r'\hline')
        lines.append(r'\noalign{\vskip 1mm}')
        return lines

    def _str_table(self, tab='    '):
        lines = list()

        indent = 1

        lines.append(r'\begin{table}[!htb]')
        lines.append(indent * tab + r'\centering')

        # caption:
        lines.append(indent * tab + r'\caption{')
        for line in self._get_caption_lines():
            lines.append(2 * tab + line)
        lines.append(indent * tab + '}')

        # label:
        lines.append(indent * tab + r'\label{{{}}}'.format(self.label))

        # group for the gap stretch:
        lines.append(r'\begingroup')
        lines.append(r'\renewcommand{{\arraystretch}}{{{}}}'.format(self.spacing))

        # resizebox and start the tabular
        lines.append(indent * tab + r'\resizebox{\textwidth}{!}{%')
        lines.append(indent * tab + r'\begin{{tabular}}{{{}}}'.format(len(self.dataframe.columns) * 'l'))
        indent = 2

        # header:
        lines.append(indent * tab + self._get_header_line(self.table_type))

        # separator:
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)

        # body:
        for line in self._get_body_lines():
            lines.append(indent * tab + line)

        # separator:
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)

        indent = 1
        lines.append(indent * tab + r'\end{tabular}')
        lines.append(indent * tab + '}')
        lines.append(indent * tab + r'\endgroup')

        lines.append(r'\end{table}')

        return '\n'.join(lines)

    def _str_longtable(self, tab='    '):
        lines = list()

        indent = 1
        cols = len(self.dataframe.columns)

        # fontsize:
        lines.append(r'\{}'.format(self.fontsize))

        lines.append(r'\begin{{longtable}}{{{}}}'.format(cols*'l'))

        # caption:
        lines.append(indent * tab + r'\caption{')
        for line in self._get_caption_lines():
            lines.append((indent + 1) * tab + line)
        lines.append(indent * tab + '}')

        # label:
        lines.append(indent * tab + r'\label{{{}}} \\'.format(self.label))

        # headers:
        lines.append(indent * tab + self._get_header_line(table_type=self.table_type))
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)
        lines.append(indent * tab + r'\endfirsthead')
        lines.append('')
        lines.append(
            indent * tab +
            r'\multicolumn{{{}}}{{c}}{{{{\tablename\ \thetable{{}} (\textit{{Continued}})}}}} \\'.format(cols)
        )
        lines.append(indent * tab + self._get_header_line(table_type=self.table_type))
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)
        lines.append(indent * tab + r'\endhead')
        lines.append('')

        # footers:
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)
        lines.append(indent * tab + r'\multicolumn{{{}}}{{r}}{{\textit{{Continued on next page}}}} \\'.format(cols))
        lines.append(indent * tab + r'\endfoot')
        lines.append('')
        for line in self._get_sep_lines():
            lines.append(indent * tab + line)
        lines.append(indent * tab + r'\endlastfoot')
        lines.append('')

        # body:
        for line in self._get_body_lines():
            lines.append(indent * tab + line)

        lines.append(r'\end{longtable}')
        lines.append(r'\normalsize')

        return '\n'.join(lines)

    def __str__(self, tab='    '):
        if self.table_type == 'table':
            return self._str_table(tab=tab)
        elif self.table_type == 'longtable':
            return self._str_longtable(tab=tab)
        else:
            raise ValueError('Table type not recognised!')

    # pre-coded decorator functions:
    @staticmethod
    def get_note_decorator(superscripts):

        def note_decorator(cell_text):
            return r'{} $^\mathrm{{{}}}$'.format(cell_text, ','.join(superscripts))

        return note_decorator


class ReactionsTable(TexTable):

    def __init__(self, chemistry, columns=('ID', 'Reaction', '$k$', r'$\Delta E_\mathrm{e}$', 'Source')):
        super().__init__('longtable')
        self.chemistry = chemistry
        # load dataframe:
        self.reactions = self.chemistry.get_reactions(disabled=False)
        df = pd.DataFrame(index=self.reactions.index, columns=columns, dtype='str')
        for col in columns:
            if col == 'ID':
                df[col] = pd.Series([str(r.id) for r in self.reactions], index=self.reactions.index)
            elif col == 'Reaction':
                df[col] = pd.Series(['${}$'.format(r.get_latex()) for r in self.reactions], index=self.reactions.index)
            elif col == '$k$':
                df[col] = pd.Series(
                    ['${}$'.format(Texify.texify_rate_coefficient(r, si_units=True)) for r in self.reactions],
                    index=self.reactions.index
                )
            elif col == r'$\Delta E_\mathrm{e}$':
                df[col] = pd.Series(
                    [str(round(r.get_el_en_loss(), 2)).rstrip('.0') if r.get_el_en_loss() != 0 else ''
                     for r in self.reactions],
                    index=self.reactions.index
                )
            elif col == 'Source':
                df[col] = pd.Series([str(r.get_doi()) for r in self.reactions], index=self.reactions.index)
            else:
                raise ValueError('Unrecognised ReactionsTable column!')

        self.load_dataframe(df)
