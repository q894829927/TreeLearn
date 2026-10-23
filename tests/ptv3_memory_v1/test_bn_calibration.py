import unittest
import torch
from torch import nn
from tree_learn.experiments.ptv3_bn_calibration_v1 import Moments, verify_state, validate_trace


class CalibrationTests(unittest.TestCase):
    def test_population_variance_includes_between_crop_mean(self):
        a = torch.tensor([[1., 3.], [3., 7.]], dtype=torch.float64)
        b = torch.tensor([[20., 30.]], dtype=torch.float64)
        moments = Moments(); moments.add(a, 1); moments.add(b)
        full = torch.cat([a, b])
        torch.testing.assert_close(moments.mean, full.mean(0), rtol=0, atol=1e-14)
        torch.testing.assert_close(moments.m2 / 2, full.var(0), rtol=0, atol=1e-12)
        self.assertEqual(moments.count, 3)

    def test_large_common_mean_and_small_variance(self):
        torch.manual_seed(3)
        data = 1e9 + torch.randn(101, 4, dtype=torch.float64) * .01
        moments = Moments(); moments.add(data, 13)
        torch.testing.assert_close(moments.mean, data.mean(0), rtol=0, atol=5e-7)
        torch.testing.assert_close(moments.m2 / 100, data.var(0), rtol=1e-5, atol=1e-9)

    def test_install_changes_only_two_statistics(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)).eval()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        moments = Moments(); moments.add(torch.tensor([[1., 2.], [4., 8.], [8., 9.]]))
        moments.install(model[1])
        report = verify_state(before, model.state_dict(), ['1'])
        self.assertEqual(set(report['actual_changed_keys']), {'1.running_mean', '1.running_var'})
        torch.testing.assert_close(model[1].running_var, torch.tensor([12.333333, 14.333333]))
        self.assertEqual(int(model[1].num_batches_tracked), 0)

    def test_parameter_mutation_rejected(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad(): model[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'non-calibration'):
            verify_state(before, model.state_dict(), ['0'])

    def test_nonfinite_and_empty_statistics_rejected(self):
        with self.assertRaises(FloatingPointError): Moments().add(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError): Moments().snapshot()

    def test_topological_group_order_and_coverage(self):
        validate_trace([['stem'], ['up', 'skip'], ['semantic', 'offset']],
                       ['stem', 'skip', 'up', 'semantic', 'offset'])
        with self.assertRaises(ValueError): validate_trace([['up'], ['stem']], ['stem', 'up'])
        with self.assertRaises(ValueError): validate_trace([['stem']], ['stem', 'head'])


if __name__ == '__main__': unittest.main()
