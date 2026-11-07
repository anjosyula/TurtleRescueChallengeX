import unittest
import numpy as np
import torch
from gaussian_dataset import seed_cloud,render_cloud,camera_delta,visible_boxes,select_sources


class GaussianGeometryTests(unittest.TestCase):
    def test_seed_uses_pixel_centers_and_never_completes_depth(self):
        depth=np.full((4,4),2.,np.float32);depth[0,0]=0;depth[1,1]=np.nan
        K=np.array([[4,0,2],[0,4,2],[0,0,1]],np.float32)
        cloud=seed_cloud(np.zeros((4,4,3),np.uint8),depth,np.ones((4,4),np.uint8),K,1)
        self.assertEqual(len(cloud['means']),14)
        self.assertTrue(np.isfinite(cloud['means']).all())
        projected=cloud['means']@K.T;uv=projected[:,:2]/projected[:,2:]
        np.testing.assert_allclose(uv%1,.5)
        self.assertTrue((cloud['scales']>0).all())

    def test_pose_and_visible_exclusive_bounds(self):
        for i in range(4):
            pose=camera_delta(i,2)
            np.testing.assert_allclose(pose[:3,:3]@pose[:3,:3].T,np.eye(3),atol=1e-6)
            self.assertAlmostEqual(float(np.linalg.det(pose[:3,:3])),1.,places=5)
        mask=np.zeros((20,20),np.uint8);mask[2:6,4:8]=1
        self.assertEqual(visible_boxes(mask),[[4,2,8,6]])

    def test_source_sampling_includes_true_negatives_without_duplicates(self):
        records=[{'sample_id':str(i),'accepted':True,'negative_scene':i%8==7} for i in range(96)]
        chosen=select_sources(records,24)
        self.assertEqual(len({r['sample_id'] for r in chosen}),24)
        self.assertEqual(sum(r['negative_scene'] for r in chosen),3)


@unittest.skipUnless(torch.backends.mps.is_available(),'Metal rasterizer requires Apple Silicon')
class GaussianRasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.K=np.array([[64,0,32],[0,64,32],[0,0,1]],np.float32)

    def cloud(self,depths,opacities,classes):
        n=len(depths)
        return {'means':np.array([[0,0,z] for z in depths],np.float32),
            'scales':np.array([[.3,.3,.01]]*n,np.float32),
            'quats':np.array([[1,0,0,0]]*n,np.float32),
            'opacities':np.array(opacities,np.float32),
            'colors':np.ones((n,3),np.float32)*.6,'shell':np.array(classes,np.float32)}

    def test_alpha_normalization_returns_axis_z_not_radial_or_accumulated_depth(self):
        r=render_cloud(self.cloud([2],[.8],[1]),self.K,np.eye(4),64,64,alpha_min=.5)
        valid=r['validity']>0
        self.assertGreater(int(valid.sum()),10)
        np.testing.assert_allclose(r['depth'][valid],2,atol=2e-6)
        self.assertEqual(r['depth'][0,0],0)

    def test_near_background_occludes_far_shell(self):
        r=render_cloud(self.cloud([1,2],[.99,.99],[0,1]),self.K,np.eye(4),64,64,alpha_min=.5)
        self.assertEqual(r['mask'][32,32],0)

    def test_mixed_depth_invalid_does_not_remove_visible_rgb_label(self):
        r=render_cloud(self.cloud([1,3],[.7,.99],[1,0]),self.K,np.eye(4),64,64,alpha_min=.5)
        self.assertEqual(r['validity'][32,32],0)
        self.assertEqual(r['depth'][32,32],0)
        self.assertEqual(r['mask'][32,32],1)


if __name__=='__main__':unittest.main()
