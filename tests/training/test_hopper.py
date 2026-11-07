import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from train_fusion import save_training_state, restore_training_state

spec=importlib.util.spec_from_file_location('hopper',Path(__file__).resolve().parents[2]/'cluster/hopper/hopper.py')
hopper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(hopper)


class HopperTests(unittest.TestCase):
    def fixture(self,root):
        rows=[]
        for split,color in [('train','green'),('val','brown')]:
            image=root/f'datasets/example/images/{split}/one.png'
            label=root/f'datasets/example/labels/{split}/one.txt'
            image.parent.mkdir(parents=True);label.parent.mkdir(parents=True)
            Image.new('RGB',(16,12),color).save(image)
            label.write_text('0 0.5 0.5 0.25 0.5\n')
            rows.append(dict(image=str(image.relative_to(root)),label=str(label.relative_to(root)),
                sha256=hopper.digest(image),label_sha256=hopper.digest(label),width=16,height=12,
                boxes=1,split=split,session_id=split+'-session'))
        manifest=root/'manifest.json';manifest.write_text(json.dumps(rows))
        return manifest,rows

    def test_prepare_rebuilds_paths_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest,rows=self.fixture(root)
            original=manifest.read_bytes()
            result=hopper.prepare(root,manifest,root/'prepared')
            self.assertEqual(result['counts'],dict(train=1,val=1,test=0))
            self.assertEqual((root/'prepared/train.txt').read_text().strip(),str((root/rows[0]['image']).resolve()))
            self.assertEqual(manifest.read_bytes(),original)

    def test_rejects_leakage_and_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest,rows=self.fixture(root)
            rows[1]['session_id']=rows[0]['session_id'];manifest.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'leakage'):hopper.audit(root,manifest)
            rows[1]['session_id']='new-session';manifest.write_text(json.dumps(rows))
            (root/rows[0]['label']).write_text('')
            with self.assertRaisesRegex(ValueError,'Checksum'):hopper.audit(root,manifest)

    def test_rejects_review_only_and_escaping_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);manifest,rows=self.fixture(root)
            rows[0]['review_only']=True;manifest.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'Review-only'):hopper.audit(root,manifest)
            for value in ('../secret','/etc/passwd','bad\nfile'):
                with self.assertRaises(ValueError):hopper.local_path(root,value)

    def test_scene_inventory_checks_frames_and_excludes_raw_media(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();scene=root/'scene';scene.mkdir()
            frame=scene/'frame.jpg';frame.write_bytes(b'scene-fixture')
            raw=root/'original.mov';raw.write_bytes(b'original not selected')
            manifest=scene/'scene.json'
            manifest.write_text(json.dumps(dict(format='turtlehero-scene-candidate-v1',
                frames=[dict(file_path='frame.jpg',sha256=hopper.digest(frame))])))
            output=root/'transfer.txt'
            report=hopper.scene_inventory(root,[scene],output)
            self.assertEqual(report['files'],2)
            self.assertNotIn('original.mov',output.read_text())
            frame.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                hopper.scene_inventory(root,[scene],output)

    def test_epoch_restart_preserves_optimizer_schedule_shuffle_and_rng(self):
        torch.manual_seed(42)
        def setup():
            model=torch.nn.Linear(3,1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,4)
            generator=torch.Generator().manual_seed(42)
            return model,optimizer,scheduler,generator
        def step(model,optimizer,scheduler,generator):
            indices=torch.randperm(8,generator=generator)
            x=torch.randn(8,3)[indices]
            optimizer.zero_grad();model(x).square().mean().backward();optimizer.step();scheduler.step()
        model,opt,sched,gen=setup();step(model,opt,sched,gen)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training_state.pt'
            save_training_state(path,model,opt,sched,gen,1,[{'epoch':1}],{}, {})
            step(model,opt,sched,gen)
            expected={k:v.clone() for k,v in model.state_dict().items()}
            model,opt,sched,gen=setup()
            state=torch.load(path,weights_only=True,map_location='cpu')
            restore_training_state(state,model,opt,sched,gen)
            step(model,opt,sched,gen)
            for key,value in expected.items():self.assertTrue(torch.equal(value,model.state_dict()[key]))
            self.assertEqual(sched.last_epoch,2)
            self.assertFalse(path.with_suffix('.tmp').exists())


if __name__=='__main__':unittest.main()
