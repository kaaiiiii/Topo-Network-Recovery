import sys
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'third_party/neural_gc'))
from neural_granger import BatchedCMLP
from neural_granger_ista_pilot import train_ista
from models.cmlp import cMLP,train_model_ista


class ISTATests(unittest.TestCase):
    def test_training_matches_author_ista(self):
        torch.set_num_threads(1);torch.manual_seed(7)
        series=torch.randn(2,12,3,dtype=torch.float64)
        batched=BatchedCMLP(3,4,[12],dtype=torch.float64)
        author=cMLP(3,1,[4]).double()
        with torch.no_grad():
            for i,net in enumerate(author.networks):
                net.layers[0].weight.copy_(batched.w1[0,i,:,:,None])
                net.layers[0].bias.copy_(batched.b1[0,i])
                net.layers[1].weight.copy_(batched.w2[0,i][None,:,None])
                net.layers[1].bias.copy_(batched.b2[0,i:i+1])
        expected=train_model_ista(author,series,lr=.05,max_iter=200,lam=.03,
                                 lam_ridge=.01,penalty='H',check_every=10,verbose=0)
        x=series[:,:-1].reshape(1,-1,3);y=series[:,1:].reshape(1,-1,3)
        _,trace=train_ista(batched,x,y,.03,max_iter=200,check_every=10,verbose=False)
        np.testing.assert_allclose([float(v) for v in expected],[r['objective'] for r in trace],rtol=1e-12,atol=1e-12)
        with torch.no_grad():
            torch.testing.assert_close(batched(x)[0],author(series[:,:-1]).reshape(-1,3),rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(batched.scores()[0],author.GC(threshold=False),rtol=1e-11,atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_capture_matches_eager(self):
        torch.manual_seed(15)
        x=torch.randn(1,20,3,device='cuda');y=torch.randn_like(x)
        models=[BatchedCMLP(3,4,[15],device='cuda') for _ in range(2)]
        for m,capture in zip(models,[False,True]):
            train_ista(m,x,y,.03,max_iter=100,check_every=10,capture=capture,verbose=False)
        for a,b in zip(models[0].parameters(),models[1].parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-7)


if __name__=='__main__':unittest.main()
