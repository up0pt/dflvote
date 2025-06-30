import unittest
import math
from generate_division import generate_divisions

class TestGenerateDivisions(unittest.TestCase):
    def test_n8_explicit(self):
        expected = [
            [[0],[1],[2],[3],[4],[5],[6],[7]],
            [[0,1],[2,3],[4,5],[6,7]],
            [[0,4],[1,5],[2,6],[3,7]],
            [[0,1,2,3],[4,5,6,7]],
            [[0,1,4,5],[2,3,6,7]],
            [[0,1,2,3,4,5,6,7]]
        ]
        result = generate_divisions(8)
        self.assertEqual(result, expected)

    def test_n16_explicit(self):
        expected = [
            [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]],
            [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]],
            [[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]],
            [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],
            [[0,1,8,9],[2,3,10,11],[4,5,12,13],[6,7,14,15]],
            [[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]],
            [[0,1,2,3,8,9,10,11],[4,5,6,7,12,13,14,15]],
            [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
        ]

        result = generate_divisions(16)
        print(result)
        self.assertEqual(result, expected)

    def test_n32_explicit(self):
        splits = generate_divisions(32)
        # first: 32 singletons
        self.assertEqual(splits[0], [[i] for i in range(32)])
        # last: one group of 0..31
        self.assertEqual(splits[-1], [list(range(32))])
        # correct count
        self.assertEqual(len(splits), 2 * int(math.log2(32)))
