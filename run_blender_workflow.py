"""One-command local Blender -> validated dataset -> controlled training -> exports."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from train_challenge import ROOT,sha256,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default='blender_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--scan',type=Path,default=Path('assets/Scan.obj'))
    p.add_argument('--assumed-length-m',type=float,default=.2)
    p.add_argument('--count',type=int,default=96);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--blender',type=Path,default=Path('/Applications/Blender.app/Contents/MacOS/Blender'))
    args=p.parse_args()
    if args.count<1 or args.epochs<1 or args.assumed_length_m<=0:
        p.error('Count, epoch limit, and assumed shell length must be positive')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        p.error('--tag must contain only letters, digits, underscores or hyphens')
    base=ROOT/'runs/blender-workflow'/args.tag;package=ROOT/'models'/args.tag
    run_names=[args.tag+'_control',args.tag+'_synthetic']
    if any(path.exists() for path in [base,package,*[ROOT/'runs/challenge'/n for n in run_names]]):
        raise FileExistsError('Choose a new tag; existing outputs are preserved')
    py=ROOT/'.venv/bin/python';coreml=ROOT/'.venv-coreml/bin/python'
    for path in [args.scan,args.blender,py,coreml]:
        if not path.is_file():raise FileNotFoundError(path)
    base.mkdir(parents=True);logdir=base/'logs';logdir.mkdir()
    sources=base/'source';sources.mkdir()
    for name in ['blender_synthetic.py','finalize_blender_asset.py','synthetic_dataset.py',
                 'train_challenge.py','evaluate_challenge.py','report_challenge.py','select_challenge.py','export_challenge.py',
                 'run_blender_workflow.py','requirements-training.txt']:
        shutil.copy2(ROOT/name,sources/name)
    commands=[]
    def run(stage,command):
        print(stage,flush=True)
        commands.append({'stage':stage,'argv':list(map(str,command))})
        write_json(base/'commands.json',commands)
        with (logdir/(stage+'.log')).open('w') as log:
            subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        write_json(base/'progress.json',{'completed_stage':stage})
    def blender(script,*arguments):
        return [args.blender,'--background','--factory-startup','--python-exit-code','1','--python',script,'--',*arguments]
    run('geometry-fixture',blender('blender_synthetic.py','--fixture','--output',base/'fixture'))
    run('renders',blender('blender_synthetic.py','--input',args.scan,'--output',base/'renders',
                         '--count',args.count,'--assumed-length-m',args.assumed_length_m))
    run('editable-asset',blender('finalize_blender_asset.py','--scene',base/'renders/scene.blend','--output',base/'shell_workbench.blend'))
    run('dataset',[py,'synthetic_dataset.py','--renders',base/'renders','--output',base/'dataset'])
    common=[py,'-u','train_challenge.py','--variant','p2','--init-checkpoint',ROOT/'models/challenge-20261019/turtle.pt',
            '--learning-rate','.00005','--epochs',args.epochs,'--patience','8']
    run('train-control',common+['--name',run_names[0]])
    run('train-synthetic',common+['--name',run_names[1],'--data',base/'dataset/data.yaml'])
    comparison=base/'comparison';comparison.mkdir()
    old=ROOT/'runs/challenge/comparison_20261019';original=json.loads((old/'comparison.json').read_text())
    write_json(comparison/'comparison.json',{'p2_20261019':original['p2_20261019']})
    shutil.copytree(old/'p2_20261019',comparison/'p2_20261019')
    checkpoints=[ROOT/'runs/challenge'/name/'weights/best.pt' for name in run_names]
    run('evaluate',[py,'evaluate_challenge.py',*checkpoints,'--output',comparison])
    run('report',[py,'report_challenge.py','--comparison',comparison/'comparison.json'])
    run('select',[py,'select_challenge.py','--comparison',comparison/'comparison.json','--output',package])
    card=json.loads((package/'model.json').read_text());prior=ROOT/'models/challenge-20261019'
    if card['checkpoint_sha256']==sha256(prior/'turtle.pt'):
        for name in ['turtle.onnx','turtle.mlpackage','export_environment.json']:
            source=prior/name
            if source.is_dir():shutil.copytree(source,package/name)
            else:shutil.copy2(source,package/name)
        report=json.loads((prior/'export_report.json').read_text())
        report['checkpoint']=str(package/'turtle.pt');report['reused_checks_from']=str(prior/'export_report.json')
        for kind,entry in report['exports'].items():entry['artifact']=str(package/Path(entry['artifact']).name)
        write_json(package/'export_report.json',report)
    else:
        run('export',[coreml,'-u','export_challenge.py','--model-card',package/'model.json'])
    exports=json.loads((package/'export_report.json').read_text())
    if any(e['status']!='passed' for e in exports['exports'].values()):raise RuntimeError('Export parity failed; inspect export_report.json')
    write_json(base/'completed.json',{'model_card':str(package/'model.json'),'selected_experiment':card['selected_experiment'],
        'synthetic_candidate_selected':card['selected_experiment']==run_names[1],
        'scan_sha256':sha256(args.scan),'editable_scene':str(base/'shell_workbench.blend'),
        'limitations':['Synthetic shell proxy and assumed scale','RGB training only; generated depth retained for later fusion work',
                       'Validation selection, not independent field testing','No physical OAK/iPhone deployment verified'],
        'uploads':False})
    print(base/'completed.json',flush=True)


if __name__=='__main__':main()
