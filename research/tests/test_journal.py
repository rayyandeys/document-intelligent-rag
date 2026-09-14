import math
import unittest
from src.evaluation.journal_metrics import metrics, interval
from src.retrieval.bm25 import BM25


class EvaluationTests(unittest.TestCase):
    def test_graded_ndcg(self):
        m=metrics(['b','x','a'],{'a':2,'b':1})
        self.assertAlmostEqual(m['ndcg_at_10'],(1+2/math.log2(4))/(2+1/math.log2(3)))
        self.assertEqual(m['recall_at_10'],1)
    def test_cutoff(self):
        m=metrics([str(i) for i in range(11)],{'10':1})
        self.assertEqual(m['mrr_at_10'],0)
        self.assertEqual(m['ndcg_at_10'],0)
    def test_duplicates(self):
        with self.assertRaises(ValueError): metrics(['a','a'],{'a':1})
    def test_bootstrap(self):
        self.assertEqual(interval([0,0,0]),dict(mean=0,ci95_low=0,ci95_high=0))
        self.assertEqual(interval([1,0,1]),interval([1,0,1]))
    def test_bm25_formula(self):
        b=BM25({'a':'cat cat','b':'dog dog'})
        result=b.retrieve('cat')
        self.assertEqual(result[0][0],'a')
        self.assertAlmostEqual(result[0][1],math.log(2)*2*2.2/(2+1.2))
        self.assertEqual(result[1][1],0)
    def test_empty_query(self):
        self.assertEqual(BM25({'z':'','a':'test'}).retrieve(''),[('a',0),('z',0)])


if __name__=='__main__': unittest.main()
