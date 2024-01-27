# Timothy Geiger, acse-tfg22

import sys
import os

# used from an MPM excersice:
# We're working in the ./docs directory, but need the package root in the path
# This command appends the directory one level up, in a cross-platform way.
sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

# General Informations
project = 'Lung Cancer Prediction using Quantum Machine Learning'
copyright = '2023, Timothy Geiger'
author = 'Timothy Geiger'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']
source_suffix = '.rst'
master_doc = 'index'
autoclass_content = "both"
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# PDF / Latex
latex_documents = [
    ('index',
     'lung_cancer.tex',
     'Documentation for lung cancer drug reponse prediction using ' +
     'quantum machine learning',
     'Timothy Geiger',
     'manual'),
]
