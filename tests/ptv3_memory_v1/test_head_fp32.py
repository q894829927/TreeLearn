"""State compatibility, exact FP32 behavior and protection before BN mutation."""
import copy
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from tree_learn.experiments.ptv3_head_fp32_v1 import enable_head_fp32, variant_config, POLICY
from tree_learn.experiments.ptv3_memory_v1.formal import formal_config


class HeadFP32Tests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(**{name:nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4),nn.ReLU(),nn.Linear(4,2))
                                  for name in ["semantic_linear","offset_linear"]})

    def test_fp32_outputs_gradients_bn_rng_and_state_keys(self):
        torch.manual_seed(42)
        old=self.model();new=copy.deepcopy(old);keys=list(new.semantic_linear.state_dict())
        parameters=[id(x) for x in new.semantic_linear.parameters()]
        enable_head_fp32(new);method=new.semantic_linear.forward;enable_head_fp32(new)
        self.assertEqual(keys,list(new.semantic_linear.state_dict()))
        self.assertEqual(parameters,[id(x) for x in new.semantic_linear.parameters()])
        self.assertIs(method.__func__,new.semantic_linear.forward.__func__)
        x=torch.randn(11,4);a=x.clone().requires_grad_();b=x.clone().requires_grad_();rng=torch.get_rng_state()
        y=old.semantic_linear(a);z=new.semantic_linear(b)
        self.assertTrue(torch.equal(y,z));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        y.square().sum().backward();z.square().sum().backward()
        self.assertTrue(torch.equal(a.grad,b.grad))
        for p,q in zip(old.semantic_linear.parameters(),new.semantic_linear.parameters()):self.assertTrue(torch.equal(p.grad,q.grad))
        for k,v in old.semantic_linear.state_dict().items():self.assertTrue(torch.equal(v,new.semantic_linear.state_dict()[k]))
        self.assertEqual(int(new.semantic_linear[1].num_batches_tracked),1)

    def test_overflow_guard_precedes_bn_mutation(self):
        m=enable_head_fp32(self.model())
        with torch.no_grad():m.semantic_linear[0].weight.fill_(torch.finfo(torch.float32).max)
        bn=copy.deepcopy(m.semantic_linear[1].state_dict())
        with self.assertRaisesRegex(FloatingPointError,"before BatchNorm"):m.semantic_linear(torch.ones(3,4))
        for k,v in bn.items():self.assertTrue(torch.equal(v,m.semantic_linear[1].state_dict()[k]))

    def test_only_numerical_policy_metadata_changes(self):
        old,new=formal_config(),variant_config()
        for k in old:
            if k!="experiment":self.assertEqual(old[k],new[k])
        expected=dict(old["experiment"],variant="full-crop-micro1-accum2-block-nested-head-fp32-v1",head_precision=POLICY)
        self.assertEqual(expected,new["experiment"])


if __name__=="__main__":unittest.main()
