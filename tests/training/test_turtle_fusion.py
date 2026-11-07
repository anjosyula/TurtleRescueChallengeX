import unittest
import torch
from torch import nn
from train_challenge import local_runtime
local_runtime()
from ultralytics.nn.tasks import DetectionModel
from turtle_fusion import ConfidenceGatedFusion,RGBDepthP2


class FusionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_quality_blocks_nonzero_learned_biases(self):
        fusion=ConfidenceGatedFusion(8,4)
        nn.init.normal_(fusion.projection.weight);nn.init.constant_(fusion.projection.bias,3)
        image=torch.randn(2,8,4,4);depth=torch.randn(2,4,4,4)
        actual=fusion(image,depth,torch.zeros(2,1,16,16))
        self.assertTrue(torch.equal(actual,image))

    def test_quality_downsampling_retains_support_fraction(self):
        fusion=ConfidenceGatedFusion(8,4)
        nn.init.constant_(fusion.projection.bias,1)
        for p in fusion.gate.parameters():nn.init.zeros_(p)
        quality=torch.zeros(1,1,16,16);quality[0,0,0,0]=1
        actual=fusion(torch.zeros(1,8,4,4),torch.zeros(1,4,4,4),quality)
        self.assertAlmostEqual(float(actual[0,0,0,0].detach()),.5/16)
        self.assertEqual(int(torch.count_nonzero(actual)),8)

    def test_four_scales_and_exact_rgb_fallback(self):
        detector=DetectionModel('yolo26n-p2.yaml',nc=1,verbose=False).eval()
        wrapper=RGBDepthP2(detector).eval()
        for fusion in wrapper.fusions.values():
            nn.init.normal_(fusion.projection.weight);nn.init.constant_(fusion.projection.bias,2)
        image=torch.rand(1,3,128,128);zero=torch.zeros_like(image)
        with torch.no_grad():
            baseline=detector(image)[0];fallback=wrapper(image,zero)[0]
        self.assertTrue(torch.equal(baseline,fallback))
        self.assertEqual(tuple(fallback.shape),(1,5,1360))
        wrapper.train()
        geometry=torch.ones(2,3,128,128)
        output=wrapper(image.expand(2,-1,-1,-1),geometry)
        self.assertEqual([f.shape[-1] for f in output['one2many']['feats']],[32,16,8,4])
        loss=output['one2many']['scores'].mean()+output['one2many']['boxes'].mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.parameters()))


if __name__=='__main__':unittest.main()
