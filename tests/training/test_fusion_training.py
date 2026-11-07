import unittest
import numpy as np
import torch
from train_fusion import flip_pair, shift_geometry, collate_pairs, set_fusion_train
from turtle_fusion import RGBDepthP2
from ultralytics.nn.tasks import DetectionModel


class FusionTrainingTests(unittest.TestCase):
    def test_flip_preserves_rgb_depth_and_box_correspondence(self):
        image=np.zeros((3,32,32),np.float32);geometry=image.copy()
        image[:,8:16,4:12]=1;geometry[:,8:16,4:12]=.75
        labels=np.array([[0,.25,.375,.25,.25]],np.float32)
        rgb,geo,boxes=flip_pair(image,geometry,labels)
        self.assertTrue(np.array_equal(rgb>0,geo>0))
        self.assertEqual(boxes[0,1],.75)
        np.testing.assert_array_equal(flip_pair(rgb,geo,boxes)[2],labels)
        self.assertEqual(labels[0,1],.25)

    def test_shift_preserves_unknown_and_never_wraps(self):
        geometry=np.ones((3,8,8),np.float32)
        geometry[:,4,4]=0
        shifted=shift_geometry(geometry,2,-1)
        self.assertTrue((shifted[:,:,:2]==0).all())
        self.assertTrue((shifted[:,-1,:]==0).all())
        self.assertTrue((shifted[:,3,6]==0).all())

    def test_collate_background_labels(self):
        x=torch.zeros(3,32,32)
        result=collate_pairs([(x,x,torch.empty(0,5)),(x,x,torch.tensor([[0,.5,.5,.2,.2]]))])
        self.assertEqual(result['batch_idx'].tolist(),[1.])
        self.assertEqual(result['bboxes'].shape,(1,4))

    def test_frozen_batchnorm_and_learned_depth_gradients(self):
        torch.set_num_threads(2)
        detector=DetectionModel('yolo26n-p2.yaml',nc=1,verbose=False)
        for p in detector.parameters():p.requires_grad_(False)
        model=RGBDepthP2(detector)
        for fusion in model.fusions.values():
            torch.nn.init.normal_(fusion.projection.weight,std=.01)
        set_fusion_train(model)
        self.assertTrue(detector.model[-1].training)
        self.assertFalse(any(m.training for m in detector.modules() if isinstance(m,torch.nn.BatchNorm2d)))
        before={k:v.clone() for k,v in detector.state_dict().items()}
        predictions=model(torch.rand(2,3,128,128),torch.ones(2,3,128,128))
        loss=predictions['one2many']['scores'].mean()+predictions['one2many']['boxes'].mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in detector.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.encoder.parameters()),0)
        self.assertTrue(all(torch.equal(v,detector.state_dict()[k]) for k,v in before.items()))


if __name__=='__main__':unittest.main()
