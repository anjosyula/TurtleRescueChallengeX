import copy
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from synthetic_dataset import simulated_depth,validate_render


class SyntheticDatasetTests(unittest.TestCase):
    def test_depth_corruption_preserves_unknown_and_is_reproducible(self):
        z=np.full((64,64),2.,np.float32);z[20:32,20:32]=0
        for profile in ('oak','iphone'):
            a,meta=simulated_depth(z,profile,42);b,_=simulated_depth(z,profile,42)
            np.testing.assert_array_equal(a,b)
            self.assertTrue(np.isfinite(a).all());self.assertTrue((a>=0).all())
            self.assertTrue((a[z==0]==0).all());self.assertGreater((a>0).sum(),0)
            self.assertFalse(meta['calibrated_sensor_model'])
        self.assertTrue((z[0]==2).all())

    def test_missing_geometry_stays_entirely_unknown(self):
        for profile in ('oak','iphone'):
            depth,_=simulated_depth(np.zeros((64,64),np.float32),profile,42)
            self.assertFalse(depth.any())

    def test_mask_drives_visible_boxes_and_rejects_false_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);mask=np.zeros((32,32),np.uint8);mask[10:15,12:20]=1
            cv2.imwrite(str(root/'rgb.png'),np.zeros((32,32,3),np.uint8))
            np.save(root/'depth.npy',np.full((32,32),2.,np.float32));np.save(root/'mask.npy',mask)
            np.save(root/'valid.npy',np.ones((32,32),np.uint8))
            record={'files':{'rgb':'rgb.png','depth':'depth.npy','mask':'mask.npy','validity':'valid.npy'},
                    'image_size_wh':[32,32],'boxes_xyxy':[[12,10,20,15]],'negative_scene':False,
                    'accepted':True,'visible_mask_pixels':40,'K':np.eye(3).tolist(),
                    'world_to_camera_cv':np.eye(4).tolist(),'split':'train','synthetic':True,'scale_measured':False}
            validate_render(root,record)
            for changed in [{'boxes_xyxy':[[0,0,20,15]]},{'negative_scene':True},{'scale_measured':True},{'split':'val'}]:
                bad=copy.deepcopy(record);bad.update(changed)
                with self.assertRaises(ValueError):validate_render(root,bad)


if __name__=='__main__':unittest.main()
