import unittest
import numpy as np
from turtle_geometry import prepare_rgb_geometry,reproject_depth


class GeometryContractTests(unittest.TestCase):
    def test_missing_geometry_and_padding_are_zero(self):
        rgb=np.zeros((24,48,3),np.uint8)
        image,geometry,t=prepare_rgb_geometry(rgb,target_wh=(64,64))
        self.assertEqual(image.shape,(3,64,64));self.assertFalse(geometry.any())
        self.assertEqual(t.top,16)
        boxes=np.array([[2,3,20,18]],np.float32)
        np.testing.assert_allclose(t.boxes(t.boxes(boxes),inverse=True),boxes,atol=1e-5)

    def test_nonfinite_depth_and_holes_stay_invalid(self):
        rgb=np.zeros((2,4,3),np.uint8)
        depth=np.array([[0,np.nan,np.inf,-1],[.1,1,3,20]],np.float32)
        _,geometry,t=prepare_rgb_geometry(rgb,depth,target_wh=(32,32))
        self.assertTrue(np.isfinite(geometry).all())
        np.testing.assert_array_equal(geometry[1]>0,geometry[2]>0)
        self.assertFalse(geometry[:,:t.top].any())
        self.assertTrue((geometry[0][geometry[1]==0]==0).all())
        self.assertEqual(int(geometry[1].sum()),128)

    def test_raw_stereo_confidence_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'normalized confidence'):
            prepare_rgb_geometry(np.zeros((2,2,3),np.uint8),np.ones((2,2)),
                                 aligned_confidence=np.full((2,2),200))

    def test_projection_identity_and_target_axis_z(self):
        depth=np.array([[0,1],[2,3]],np.float32);k=np.eye(3)
        np.testing.assert_array_equal(reproject_depth(depth,k,k,np.eye(4),(2,2)),depth)
        transform=np.eye(4);transform[2,3]=1
        out=reproject_depth(np.array([[2]],np.float32),k,k,transform,(1,1))
        self.assertEqual(out[0,0],3)

    def test_projection_nearest_surface_wins(self):
        k=np.eye(3);target=np.diag([.1,.1,1.])
        out=reproject_depth(np.array([[2,1]],np.float32),k,target,np.eye(4),(1,1))
        self.assertEqual(out[0,0],1)


if __name__=='__main__':unittest.main()
