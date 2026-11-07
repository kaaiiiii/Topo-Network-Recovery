import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from sis_aware import node_design,positive_path,fit_path
from sis_recovery import sis_step


class SISAwareTests(unittest.TestCase):
    def test_exact_sis_features_and_direction(self):
        b=np.array([[0.,0.,0.],[.7,0.,0.],[0.,.4,0.]])
        x=np.random.default_rng(17).uniform(.02,.8,size=(100,3))
        y=sis_step(x,b,1.5)
        for i in range(3):
            design,target,sources=node_design(x,y,i,1.5)
            np.testing.assert_allclose(design@b[i,sources],target,atol=1e-15)
        path,info=fit_path(x,y,1.5,[0.,.01])
        np.testing.assert_allclose(path[0.],b,atol=1e-10)
        self.assertTrue(np.all(np.diag(path[.01])==0))
        self.assertGreaterEqual(path[.01].min(),0)

    def test_nonnegative_lasso_closed_form_on_orthogonal_design(self):
        x=np.eye(4)*2
        y=np.array([1.,-.4,2.,0.])
        path,info=positive_path(x,y,[0.,.1,1.])
        sx=np.sqrt(np.mean(x*x,axis=0))
        sy=np.sqrt(np.mean(y*y))
        xs,ys=x/sx,y/sy
        corr=xs.T@ys/len(x)
        amax=max(0.,corr.max())
        for fraction,coef in path.items():
            expected=np.maximum(corr-fraction*amax,0.)*sy/sx
            np.testing.assert_allclose(coef,expected,atol=1e-12)

    def test_no_intercept_and_zero_signal(self):
        x=np.random.default_rng(8).uniform(0,1,size=(40,5))
        path,info=positive_path(x,np.zeros(40),[0.,.01,1.])
        for coef in path.values():
            np.testing.assert_array_equal(coef,np.zeros(5))


if __name__=='__main__':
    unittest.main()
