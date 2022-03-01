# The MIT License (MIT)
# Copyright (c) 2022 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import unittest

from xcube.util.types import normalize_number_scalar_or_pair


class NormalizeNumberScalarOrPairTest(unittest.TestCase):

    def test_scalar_float(self):
        self.assertEqual((1.1, 1.1),
                         normalize_number_scalar_or_pair(1.1, float))
        self.assertEqual((2.0, 2.0),
                         normalize_number_scalar_or_pair(2.0, float))

    def test_scalar_int(self):
        self.assertEqual((1, 1),
                         normalize_number_scalar_or_pair(1.1, int))
        self.assertEqual((1, 1),
                         normalize_number_scalar_or_pair(1.9, int))
        self.assertEqual((2, 2),
                         normalize_number_scalar_or_pair(2.0, int))

    def test_pair_float(self):
        self.assertEqual((1.1, 1.9),
                         normalize_number_scalar_or_pair((1.1, 1.9), float))
        self.assertEqual((2.0, 1.9),
                         normalize_number_scalar_or_pair((2.0, 1.9), float))

    def test_pair_int(self):
        self.assertEqual((1, 1),
                         normalize_number_scalar_or_pair((1.1, 1.9), int))
        self.assertEqual((2, 1),
                         normalize_number_scalar_or_pair((2.0, 1.9), int))

    def test_wrong_type(self):
        with self.assertRaises(ValueError) as ve:
            normalize_number_scalar_or_pair('xyz', int)
        self.assertEqual(
            'Value "xyz" must be a number or a segment of two numbers',
            f'{ve.exception}'
        )

    def test_wrong_length(self):
        with self.assertRaises(ValueError) as ve:
            normalize_number_scalar_or_pair((0, 1, 2))
        self.assertEqual(
            'Value "(0, 1, 2)" must be a number or a segment of two numbers',
            f'{ve.exception}'
        )