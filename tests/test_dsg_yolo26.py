"""Standard-library tests; no pytest installation is required."""

from __future__ import annotations

import unittest

import torch
from ultralytics.nn.modules import Detect

from dsg_yolo26 import (
    DPSE,
    SGCF,
    DSGYOLO26,
    ScaleSensitiveRegression,
    build_dsg_yolo26,
)


class TestCustomModules(unittest.TestCase):
    def test_dpse_preserves_resolution_and_backpropagates(self):
        module = DPSE(8, 8, n=2)
        sample = torch.randn(2, 8, 8, 8, requires_grad=True)
        output = module(sample)
        self.assertEqual(output.shape, sample.shape)
        self.assertEqual(module.blocks[0].depthwise.conv.groups, 8)
        output.square().mean().backward()
        self.assertIsNotNone(sample.grad)
        self.assertGreater(sample.grad.abs().sum().item(), 0)

    def test_sgcf_gates_both_scales_and_backpropagates(self):
        module = SGCF([16, 8], c2=8)
        semantics = torch.randn(2, 16, 8, 8, requires_grad=True)
        details = torch.randn(2, 8, 8, 8, requires_grad=True)
        projected = module.project([semantics, details])
        weights = module.scale_weights(projected)
        self.assertEqual(weights.shape, (2, 2, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2, 8), atol=1e-6))
        # The zero-initialized output gate starts with neutral 50/50 scale admission.
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 0.5)))

        output = module([semantics, details])
        self.assertEqual(output.shape, (2, 8, 8, 8))
        output.mean().backward()
        self.assertGreater(semantics.grad.abs().sum().item(), 0)
        self.assertGreater(details.grad.abs().sum().item(), 0)

    def test_sgcf_rejects_mismatched_spatial_dimensions(self):
        module = SGCF([16, 8], c2=8)
        with self.assertRaisesRegex(ValueError, "identical batch and spatial"):
            module([torch.randn(1, 16, 8, 8), torch.randn(1, 8, 4, 4)])

    def test_ssr_is_training_only_and_normalizes_valid_weights(self):
        ssr = ScaleSensitiveRegression(power=-0.5)
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
        weights = ssr(boxes)
        self.assertEqual(len(list(ssr.parameters())), 0)
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertAlmostEqual(weights[:2].mean().item(), 1.0, places=5)
        self.assertEqual(weights[2].item(), 0.0)


class TestDSGModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = DSGYOLO26(scale="n", nc=10, ch=3, verbose=False)
        cls.model = cls.wrapper.model

    def test_architecture_and_yolo26_head(self):
        head = self.model.model[-1]
        self.assertIs(type(head), Detect)  # SSR adds no inference-time head or cost.
        self.assertEqual(head.nl, 4)
        self.assertEqual(head.reg_max, 1)
        self.assertTrue(hasattr(head, "one2one_cv2"))
        self.assertTrue(self.model.yaml["end2end"])
        self.assertEqual(head.stride.tolist(), [4.0, 8.0, 16.0, 32.0])
        self.assertEqual(sum(isinstance(module, DPSE) for module in self.model.model), 1)
        self.assertEqual(sum(isinstance(module, SGCF) for module in self.model.model), 1)
        self.assertIsInstance(self.model.model[18], SGCF)
        self.assertIsInstance(self.model.model[19], DPSE)

    def test_high_level_wrapper_metadata(self):
        self.assertEqual(self.wrapper.task, "detect")
        self.assertEqual(self.wrapper.model.task, "detect")
        self.assertEqual(self.wrapper.model.yaml["nc"], 10)
        self.assertEqual(self.wrapper.model.args["task"], "detect")

    def test_training_forward_at_multiple_input_sizes(self):
        self.model.train()
        for image_size in (64, 96):
            with self.subTest(image_size=image_size):
                sample = torch.randn(1, 3, image_size, image_size)
                with torch.no_grad():
                    output = self.model(sample)
                self.assertIn("one2many", output)
                self.assertIn("one2one", output)
                features = output["one2many"]["feats"]
                expected_channels = (32, 64, 128, 256)
                expected_strides = (4, 8, 16, 32)
                self.assertEqual(len(features), 4)
                for feature, channels, stride in zip(features, expected_channels, expected_strides):
                    self.assertEqual(
                        feature.shape,
                        (1, channels, image_size // stride, image_size // stride),
                    )

    def test_end_to_end_eval_forward(self):
        self.model.eval()
        head = self.model.model[-1]
        head.end2end = True
        head.max_det = 20
        with torch.no_grad():
            predictions, raw = self.model(torch.randn(1, 3, 64, 64))
        self.assertEqual(predictions.shape, (1, 20, 6))
        self.assertIn("one2one", raw)
        head.end2end = False

    def test_invalid_scale_is_rejected_before_construction(self):
        with self.assertRaises(ValueError):
            build_dsg_yolo26(scale="invalid", verbose=False)


if __name__ == "__main__":
    unittest.main()
