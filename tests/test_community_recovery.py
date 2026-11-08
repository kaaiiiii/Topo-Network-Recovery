import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from community_recovery import community_system, edge_strata


class CommunityTests(unittest.TestCase):
    def test_fixed_counts_and_sis_step_bound(self):
        for seed in [5420]+list(range(1001,1011)):
            b,truth,labels = community_system(seed)
            self.assertEqual(int(truth.sum()),300)
            self.assertFalse(np.diag(truth).any())
            for group in range(4):
                self.assertEqual(int(truth[np.ix_(labels==group,labels==group)].sum()),60)
            self.assertEqual(int(truth[labels[:,None]!=labels[None,:]].sum()),60)
            self.assertAlmostEqual(float(max(abs(np.linalg.eigvals(b)))),1.,places=10)
            self.assertLessEqual(.05*3*b.sum(axis=1).max(),1.)
        first = community_system(1001)
        second = community_system(1001)
        for x,y in zip(first,second): np.testing.assert_array_equal(x,y)
        self.assertFalse(np.array_equal(first[1],community_system(1002)[1]))

    def test_stratified_recovery_and_global_edge_budget(self):
        b,truth,labels = community_system(1001)
        scores = b.copy()
        np.fill_diagonal(scores,1000.)
        perfect = edge_strata(truth,scores,labels)
        for key in ['within_ap','bridge_ap','within_recall_at_300','bridge_recall_at_300']:
            self.assertAlmostEqual(perfect[key],1.)
        self.assertEqual(perfect['selected_edges'],300)
        self.assertEqual(perfect['within_candidate_edges'],2400)
        self.assertEqual(perfect['bridge_candidate_edges'],7500)
        scores[labels[:,None]!=labels[None,:]]=0.
        internal_only = edge_strata(truth,scores,labels)
        self.assertEqual(internal_only['within_recall_at_300'],1.)
        self.assertEqual(internal_only['bridge_recall_at_300'],0.)
        self.assertEqual(internal_only['selected_edges'],240)


if __name__=='__main__':
    unittest.main()
