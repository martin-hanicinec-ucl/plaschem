import glob
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

doc_name = 'pygmol_doc'

commands = [
    f'pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex',
    f'bibtex {doc_name}.aux',
    f'pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex',
    f'pdflatex -synctex=1 -interaction=nonstopmode {doc_name}.tex',
]
for cmd in commands:
    os.system(cmd)

# remove the byproducts:
to_keep = set([f'{doc_name}.{ext}' for ext in ['tex', 'pdf', 'bib']])
to_keep.add(f'{doc_name}_content.tex')
for file in glob.glob(f'{doc_name}*.*'):
    if file not in to_keep:
        os.system('rm {}'.format(file))
