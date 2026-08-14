import unittest
from unittest import mock

from tools.diagnostics import smoke_sparse_unet_pipeline as smoke


class SparseUNetPipelineSmokeTests(unittest.TestCase):

    def test_pointwise_tuple_is_reordered_for_coordinate_first_ensemble(self):
        pointwise = tuple(object() for _ in range(14))
        logger = object()
        sentinel = object()
        with mock.patch.object(smoke, 'ensemble', return_value=sentinel) as call:
            result = smoke.ensemble_pointwise_predictions(
                pointwise, logger=logger)

        self.assertIs(result, sentinel)
        call.assert_called_once_with(
            pointwise[6],
            pointwise[0],
            pointwise[1],
            pointwise[2],
            pointwise[3],
            pointwise[4],
            pointwise[5],
            pointwise[7],
            pointwise[8],
            pointwise[9],
            pointwise[10],
            pointwise[11],
            pointwise[12],
            pointwise[13],
            logger=logger)


if __name__ == '__main__':
    unittest.main()
