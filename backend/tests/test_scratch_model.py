"""Shape, range, and gradient smoke tests for EffectLab Scratch."""
import unittest
import torch

from scratch_model import EffectLabScratchVideoAI


class ScratchModelTests(unittest.TestCase):
    def test_output_shape_and_range(self):
        model = EffectLabScratchVideoAI()
        reference = torch.rand(1, 3, 64, 64)
        driving = torch.rand(1, 3, 2, 64, 64)
        output = model(reference, driving)
        self.assertEqual(tuple(output.shape), (1, 3, 2, 64, 64))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_gradients_reach_inputs_and_weights(self):
        model = EffectLabScratchVideoAI()
        reference = torch.rand(1, 3, 64, 64, requires_grad=True)
        driving = torch.rand(1, 3, 2, 64, 64, requires_grad=True)
        output = model(reference, driving)
        output.mean().backward()
        self.assertIsNotNone(reference.grad)
        self.assertIsNotNone(driving.grad)
        self.assertIsNotNone(model.video_stem.weight.grad)


if __name__ == "__main__":
    unittest.main()
