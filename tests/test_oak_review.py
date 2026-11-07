import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from oak_review import digest, load_proposals


class ProposalReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cv2.imwrite(str(self.root/'rgb.png'),np.zeros((20,30,3),dtype=np.uint8))
        self.records = [{'sample_id':'000000','session_id':'session-A','files':{'rgb':'rgb.png'}}]
        self.data = {'annotation_status':'unreviewed_proposals','session_id':'session-A','view':'rgb',
                     'selected_sample_ids':['000000'],'samples':{'000000':{'image':'rgb.png',
                      'image_sha256':digest(self.root/'rgb.png'),'size_wh':[30,20],
                      'boxes_xyxy':[[2.,3.,15.,17.]],'scores':[.5]}}}

    def load(self, data):
        path=self.root/'proposals.json';path.write_text(json.dumps(data))
        return load_proposals(path,self.root,self.records,'rgb')

    def test_valid_proposals_stay_unreviewed_and_do_not_create_labels(self):
        self.assertEqual(self.load(self.data)['annotation_status'],'unreviewed_proposals')
        self.assertFalse((self.root/'annotations').exists())

    def test_empty_predictions_are_not_reviewed_background(self):
        self.data['samples']['000000'].update(boxes_xyxy=[],scores=[])
        self.assertEqual(self.load(self.data)['annotation_status'],'unreviewed_proposals')

    def test_wrong_session_view_changed_image_and_invalid_box_rejected(self):
        for field,value in [('session_id','other'),('view','rectified_left')]:
            data=copy.deepcopy(self.data);data[field]=value
            with self.assertRaises(ValueError):self.load(data)
        for field,value in [('image_sha256','incorrect'),('boxes_xyxy',[[0,0,40,20]])]:
            data=copy.deepcopy(self.data);data['samples']['000000'][field]=value
            with self.assertRaises(ValueError):self.load(data)

    def test_duplicate_and_unknown_selection_rejected(self):
        for ids in [['000000','000000'],['000001']]:
            data=copy.deepcopy(self.data);data['selected_sample_ids']=ids
            with self.assertRaises(ValueError):self.load(data)

    def run_review(self, keys):
        import annotate_oak
        (self.root/'manifest.jsonl').write_text(json.dumps(self.records[0])+'\n')
        self.load(self.data)
        argv=['annotate_oak.py',str(self.root),'--proposals',str(self.root/'proposals.json')]
        with patch('sys.argv',argv), patch.object(annotate_oak,'validate'), \
             patch.object(cv2,'imshow'), patch.object(cv2,'destroyAllWindows'), \
             patch.object(cv2,'waitKey',side_effect=keys):
            annotate_oak.main()
        return self.root/'annotations/rgb/000000.json'

    def test_accept_key_does_not_turn_empty_proposals_into_background(self):
        self.data['samples']['000000'].update(boxes_xyxy=[],scores=[])
        self.assertFalse(self.run_review([ord('a'),ord('q')]).exists())

    def test_accept_key_records_explicit_review_with_model_provenance(self):
        self.data['checkpoint_sha256']='test-checkpoint'
        label=json.loads(self.run_review([ord('a')]).read_text())
        self.assertEqual(label['annotation_status'],'reviewed_turtle')
        self.assertEqual(label['review_method'],'explicit_proposal_acceptance')
        self.assertEqual(label['proposal_checkpoint_sha256'],'test-checkpoint')


if __name__=='__main__':unittest.main()
