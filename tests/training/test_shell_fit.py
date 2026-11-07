import unittest
import numpy as np
from shell_fit import canonical,deform,project,rotation,resample_polygon

class ShellFitGeometryTests(unittest.TestCase):
    def test_size_and_proportions_are_explicit(self):
        raw=np.array([[-2,-1,-3],[2,1,3],[0,0,0]],float)
        vertices=deform(canonical(raw),.2,.75,.4)
        np.testing.assert_allclose(np.ptp(vertices,axis=0),[.2,.15,.08])
        self.assertEqual(vertices[:,2].min(),0)

    def test_projection_matches_pinhole_and_off_center_intrinsics(self):
        K=np.array([[800,0,300],[0,900,250],[0,0,1]],float)
        uv,z=project(np.array([[0,0,0],[.1,-.2,0]]),K,np.eye(3),np.array([0,0,2]))
        np.testing.assert_allclose(uv,[[300,250],[340,160]])
        np.testing.assert_allclose(z,[2,2])

    def test_rgb_scale_ambiguity_is_preserved(self):
        rng=np.random.default_rng(3);vertices=rng.uniform(-.1,.1,(100,3))
        K=np.array([[900,0,320],[0,900,320],[0,0,1]],float)
        R=rotation(23,35);t=np.array([.1,.05,1.])
        a,za=project(vertices,K,R,t);b,zb=project(vertices*2,K,R,t*2)
        np.testing.assert_allclose(a,b);np.testing.assert_allclose(zb,za*2)

    def test_camera_rotation_and_behind_camera_rejection(self):
        R=rotation(-123,46);np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R),1)
        with self.assertRaises(ValueError):project(np.zeros((1,3)),np.eye(3),R,np.array([0,0,-1]))

    def test_contour_sampling_is_uniform_and_closed_without_duplicate_endpoint(self):
        p=resample_polygon([[0,0],[1,0],[1,1],[0,1]],8)
        self.assertEqual(p.shape,(8,2))
        np.testing.assert_allclose(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1),.5)

if __name__=='__main__':unittest.main()
