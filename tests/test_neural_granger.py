import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'third_party/neural_gc'))
from neural_granger import BatchedCMLP,CONFIG,objective,predict,train_batch
from models.cmlp import cMLP,regularize,ridge_regularize


class NeuralGrangerTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is unavailable')
    def test_captured_refit_matches_eager_updates(self):
        rng=np.random.default_rng(13)
        x=rng.normal(size=(1,30,3));y=rng.normal(size=(1,30,3))
        with patch.dict(CONFIG,hidden=4,batch_size=8):
            eager,_=train_batch(x,y,[8],[.01],[100],'cuda',lambda step,model:None,cuda_graph=False)
            captured,_=train_batch(x,y,[8],[.01],[100],'cuda',lambda step,model:None,cuda_graph=True)
        for a,b in zip(eager.parameters(),captured.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=1e-7)

    def test_author_forward_objective_gradient_and_adam_step(self):
        torch.manual_seed(100)
        batch,nodes,hidden,samples=2,3,4,7
        model=BatchedCMLP(nodes,hidden,[11,12],dtype=torch.float64)
        x=torch.randn(batch,samples,nodes,dtype=torch.float64)
        y=torch.randn_like(x)
        penalties=torch.tensor([.001,.03],dtype=torch.float64)
        originals=[]
        for k in range(batch):
            original=cMLP(nodes,lag=1,hidden=[hidden]).double()
            with torch.no_grad():
                for i,net in enumerate(original.networks):
                    net.layers[0].weight.copy_(model.w1[k,i,:,:,None])
                    net.layers[0].bias.copy_(model.b1[k,i])
                    net.layers[1].weight.copy_(model.w2[k,i][None,:,None])
                    net.layers[1].bias.copy_(model.b2[k,i:i+1])
            originals.append(original)
        expected=torch.cat([m(x[k:k+1]) for k,m in enumerate(originals)],dim=0)
        torch.testing.assert_close(model(x),expected,rtol=1e-12,atol=1e-12)
        loss,_=objective(model,x,y,penalties)
        reference=sum(((m(x[k:k+1])-y[k:k+1])**2).mean(dim=(0,1)).sum()+
                      sum(regularize(net,penalties[k],'GL')+ridge_regularize(net,CONFIG['output_ridge'])
                          for net in m.networks) for k,m in enumerate(originals))
        torch.testing.assert_close(loss,reference,rtol=1e-12,atol=1e-12)
        optimizer=torch.optim.Adam(model.parameters(),lr=CONFIG['learning_rate'])
        other=[torch.optim.Adam(m.parameters(),lr=CONFIG['learning_rate']) for m in originals]
        loss.backward();reference.backward()
        for k,m in enumerate(originals):
            for i,net in enumerate(m.networks):
                for grad,ref in [(model.w1.grad[k,i],net.layers[0].weight.grad[:,:,0]),
                                 (model.b1.grad[k,i],net.layers[0].bias.grad),
                                 (model.w2.grad[k,i],net.layers[1].weight.grad[0,:,0]),
                                 (model.b2.grad[k,i:i+1],net.layers[1].bias.grad)]:
                    torch.testing.assert_close(grad,ref,rtol=1e-11,atol=1e-12)
        optimizer.step()
        for opt in other:opt.step()
        expected=torch.cat([m(x[k:k+1]) for k,m in enumerate(originals)],dim=0)
        torch.testing.assert_close(model(x),expected,rtol=1e-11,atol=1e-12)
        for k,m in enumerate(originals):torch.testing.assert_close(model.scores()[k],m.GC(threshold=False))

    def test_input_group_direction_and_independent_saved_prediction(self):
        model=BatchedCMLP(3,4,[42],dtype=torch.float64)
        with torch.no_grad():model.w1[0,1,:,2]=0.
        x=torch.randn(1,9,3,dtype=torch.float64)
        altered=x.clone();altered[:,:,2]+=5
        torch.testing.assert_close(model(x)[0,:,1],model(altered)[0,:,1])
        self.assertEqual(model.scores()[0,1,2].item(),0.)
        arrays={name:p.detach().numpy()[0] for name,p in model.named_parameters()}
        arrays.update(x_mean=np.array([1.,2.,3.]),x_scale=np.array([2.,3.,4.]),
                      y_mean=np.array([4.,3.,2.]),y_scale=np.array([3.,4.,5.]))
        raw=x.numpy()[0]*arrays['x_scale']+arrays['x_mean']
        expected=model(x).detach().numpy()[0]*arrays['y_scale']+arrays['y_mean']
        np.testing.assert_allclose(predict(arrays,raw),expected,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()
