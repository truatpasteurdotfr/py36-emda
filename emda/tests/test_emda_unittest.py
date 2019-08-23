"""
Author: "Rangana Warshamanage, Garib N. Murshudov"
MRC Laboratory of Molecular Biology
    
This software is released under the
Mozilla Public License, version 2.0; see LICENSE.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from emda import iotools
import numpy as np

class TestEmda(unittest.TestCase):

    def test_iotools(self):
        iotools.test()

    def test_fcodes(self):
        import fcodes
        fcodes.test()

if __name__ == '__main__':
    unittest.main()
