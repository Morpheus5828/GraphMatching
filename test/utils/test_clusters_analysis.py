"""This module test cluster_analysis.py

..moduleauthor:: Marius Thorre
"""

from unittest import TestCase
import sys, os
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.append(project_path)

class Test(TestCase):
    pass
