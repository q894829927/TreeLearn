"""CPU-only collection of short stability records, including negative controls."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def collect(roots, output, tests):
    output.mkdir(parents=True,exist_ok=False)
    inventory=[];cases={};comparisons={}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '/sources/' in str(path) or path.suffix=='.tmp':continue
            relative=path.relative_to(root)
            if path.suffix=='.pth':continue  # hashes of checkpoints/tensor dumps are recorded separately
            inventory.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest(path)))
            if path.suffix!='.npz':
                target=output/root.name/relative;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(path,target)
            if path.name=='stability_result.json':
                key=str(Path(root.name)/relative.parent)
                rows=[json.loads(line) for line in path.with_name('stability_steps.jsonl').read_text().splitlines()] if path.with_name('stability_steps.jsonl').exists() else []
                cases[key]=dict(result=read(path),steps=rows)
            if path.name.startswith('comparison_') and path.suffix=='.json':
                comparisons[path.stem]=read(path)
        if (root/'plan.json').exists():
            plan=read(root/'plan.json')
            for case in [k for k in cases if k.startswith(root.name+'/')]:
                seed=cases[case]['result']['seed'];samples=plan['plans'][str(seed)]['samples']
                target=output/(case.replace('/','_')+'_steps.csv')
                with target.open('w') as stream:
                    writer=csv.writer(stream,lineterminator='\n')
                    writer.writerow(['update','points_micro1','points_micro2','voxels_micro1','voxels_micro2','allocated_bytes','reserved_bytes','peak_allocated_bytes','peak_reserved_bytes','seconds','loss','lr','bn_once','point_alive','amp_skipped'])
                    for row in cases[case]['steps']:
                        i=(row['update']-1)*2;m=row['memory']
                        writer.writerow([row['update'],samples[i]['points'],samples[i+1]['points'],*row['voxels'],m['allocated'],m['reserved'],m['peak_allocated'],m['peak_reserved'],row['seconds'],row['loss'],row['lr'],row['bn_once_per_micro'],row['point_alive'],row['optimizer_step_skipped']])
    (output/'runs_summary.json').write_text(json.dumps(cases,indent=2)+'\n')
    (output/'comparisons_summary.json').write_text(json.dumps({k:{name:value for name,value in row.items() if name not in ('gradient_errors','buffer_errors')} for k,row in comparisons.items()},indent=2)+'\n')
    (output/'artifact_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    shutil.copyfile(tests,output/'unit_tests.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tests',type=Path,required=True)
    args=p.parse_args();collect(args.root,args.output,args.tests)
