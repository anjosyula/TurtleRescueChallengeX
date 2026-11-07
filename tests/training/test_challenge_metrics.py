import unittest
from evaluate_challenge import match_predictions, operating_points, breakdown
from predict_challenge import associate_depth
import numpy as np


class ChallengeMetricsTests(unittest.TestCase):
    def test_duplicate_detection_cannot_match_twice(self):
        outcomes, used = match_predictions([[0,0,10,10,.9,0],[0,0,10,10,.8,0]], [[0,0,10,10]])
        self.assertEqual([tp for _,tp in outcomes],[1,0])
        self.assertEqual(used,{0})

    def test_empty_background_and_no_predictions(self):
        self.assertEqual(match_predictions([[0,0,10,10,.8,0]],[])[0],[(.8,0)])
        result=operating_points([{'truth':[[0,0,10,10]],'predictions':[]}])
        self.assertEqual(result['best_F2']['fn'],1)
        self.assertEqual(result['best_F2']['recall'],0)

    def test_equal_scores_are_included_as_a_group(self):
        result=operating_points([{'truth':[[0,0,10,10]],
            'predictions':[[0,0,10,10,.8,0],[20,20,30,30,.8,0]]}])
        self.assertEqual(result['best_F2']['tp'],1)
        self.assertEqual(result['best_F2']['fp'],1)
        self.assertEqual(result['best_F2']['precision'],.5)

    def test_size_recall_uses_letterboxed_pixels(self):
        result=breakdown([{'width':1280,'height':960,'truth':[[0,0,40,40]],
                          'predictions':[[0,0,40,40,.8,0]]}],.5)
        self.assertEqual(result['recall_by_size_at_640']['small'],{'objects':1,'matched':1,'recall':1.})
        self.assertIsNone(result['recall_by_size_at_640']['large']['recall'])

    def test_missing_depth_never_fabricates_distance(self):
        self.assertIsNone(associate_depth(np.zeros((40,40)),[0,0,40,40])['distance_meters'])
        depth=np.full((40,40),np.nan);depth[20,20]=2
        self.assertEqual(associate_depth(depth,[0,0,40,40])['distance_quality'],'unknown')

    def test_depth_association_reports_uncalibrated_estimate(self):
        result=associate_depth(np.full((40,40),2.),[0,0,40,40])
        self.assertEqual(result['distance_meters'],2.)
        self.assertEqual(result['distance_quality'],'low')


if __name__=='__main__':unittest.main()
