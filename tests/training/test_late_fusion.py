import unittest
from turtle_late_fusion import late_or

class LateFusionTests(unittest.TestCase):
    a=[10,20,30,40,.8,0]
    b=[50,60,70,80,.6,0]
    def test_unavailable_depth_never_vetoes_rgb(self):
        out=late_or([self.a],None,.5,.5,False)
        self.assertTrue(out['possible_turtle']);self.assertIsNone(out['depth_hit'])
        self.assertEqual(out['detections'][0]['box'],self.a[:4])
    def test_depth_recovers_unmatched_and_retains_provenance(self):
        out=late_or([], [self.b],.5,.5,True)
        self.assertTrue(out['possible_turtle']);self.assertFalse(out['rgb_hit'])
        self.assertEqual(out['detections'][0]['branches'],{'depth':.6})
    def test_duplicates_do_not_average_scores_or_move_rgb(self):
        out=late_or([self.a],[self.a[:4]+[.9,0],self.b],.5,.5,True)
        self.assertEqual(len(out['detections']),2)
        self.assertEqual(out['detections'][0]['branches'],{'rgb':.8,'depth':.9})
        self.assertEqual(out['detections'][0]['box'],self.a[:4])
    def test_below_threshold_and_unavailable_are_distinct(self):
        self.assertIs(late_or([],[],.5,.5,True)['depth_hit'],False)
        self.assertIsNone(late_or([],[],.5,.5,False)['depth_hit'])
        self.assertFalse(late_or([], [self.b],.5,.7,True)['possible_turtle'])
    def test_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):late_or([[0,0,10,10,float('nan'),0]],[],.5,.5,True)
        with self.assertRaises(ValueError):late_or([],None,.5,.5,True)
    def test_or_truth_table(self):
        for rgb_hit in (False,True):
            for depth_hit in (False,True):
                out=late_or([self.a] if rgb_hit else [],[self.b] if depth_hit else [],.5,.5,True)
                self.assertEqual(out['possible_turtle'],rgb_hit or depth_hit)
    def test_fixed_fp_budget_matches_explicit_threshold_counts(self):
        from evaluate_dual_pilot import at_fp_budget,counts
        samples=[dict(truth=[[0,0,10,10]],predictions=[
            [0,0,10,10,.8,0],[20,20,30,30,.7,0],[0,0,10,10,.6,0]]),
            dict(truth=[[0,0,10,10]],predictions=[[0,0,10,10,.65,0]])]
        for budget in range(3):
            point=at_fp_budget(samples,budget)
            actual=counts(samples,point['threshold'])
            self.assertEqual(actual['tp'],point['tp'])
            self.assertEqual(actual['fp'],point['fp'])
            self.assertLessEqual(actual['fp'],budget)

if __name__=='__main__':unittest.main()
