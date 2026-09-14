import unittest
from experiments.analyze_truncation import stratum


class StratumTests(unittest.TestCase):
    def test_boundary_includes_special_tokens(self):
        self.assertEqual(stratum([256,10],256),'all_within_limit')
        self.assertEqual(stratum([257,300],256),'all_over_limit')
        self.assertEqual(stratum([256,257],256),'mixed')
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):stratum([],256)


if __name__=='__main__':unittest.main()
