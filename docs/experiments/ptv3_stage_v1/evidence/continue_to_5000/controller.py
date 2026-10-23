import csv, hashlib, json, os, pathlib, subprocess, sys, time, traceback
HERE=pathlib.Path(__file__).resolve().parent
CFG=json.loads((HERE/'commands.json').read_text())
ROOT=pathlib.Path(CFG['stage_root'])
def write(path,value):
 temporary=path.with_name(path.name+'.tmp')
 temporary.write_text(json.dumps(value,indent=2)+'\n')
 os.replace(temporary,path)
def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for data in iter(lambda:f.read(8388608),b''):h.update(data)
 return h.hexdigest()
try:
 for phase in ('training','evaluation'):
  with (HERE/(phase+'.log')).open('x') as log:
   child=subprocess.Popen(CFG[phase],cwd=CFG['cwd'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
   write(HERE/'controller_status.json',dict(status=phase,controller_pid=os.getpid(),child_pid=child.pid,command=CFG[phase],updated_unix=time.time()))
   code=child.wait()
  if code:raise RuntimeError(phase+' exited with code '+str(code))
  if phase=='training':
   provenance=json.loads((ROOT/'training/to_5000/checkpoint_provenance.json').read_text())
   assert provenance['global_update']==5000 and not provenance['is_formal_final']
 folder=HERE/'evaluation'
 manifest=json.loads((folder/'evaluation_manifest.json').read_text())
 original=json.loads((ROOT/'attempts/v2/evaluations/e1_u2900/evaluation_manifest.json').read_text())
 assert manifest['code_sha256']==original['code_sha256']
 assert manifest['protocol_sha256']==original['protocol_sha256']==digest(ROOT/'protocol.json')
 assert manifest['config_sha256']==original['config_sha256']
 assert manifest['checkpoint_update']==5000 and not manifest['is_formal_final']
 metrics=json.loads((folder/'metrics.json').read_text())
 assert metrics['gt_trees']==24
 comparison=json.loads((ROOT/'comparison.json').read_text())
 comparison['results'].append(dict(**metrics,total_seconds=manifest['total_seconds'],pointwise_seconds=manifest['pointwise_seconds'],checkpoint_sha256=manifest['checkpoint_sha256']))
 write(ROOT/'comparison_at_5000.json',comparison)
 columns=['tag','checkpoint_update','TP','FP','FN','detection_precision','detection_recall','detection_f1','coverage','semantic_tree_recall','offset_xy_error_m','offset_xyz_error_m','pointwise_seconds','total_seconds']
 with (ROOT/'comparison_at_5000.csv').open('x',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(comparison['results'])
 write(HERE/'controller_status.json',dict(status='complete',stop_update=5000,comparison=str(ROOT/'comparison_at_5000.json'),automatic_later_training=False,updated_unix=time.time()))
 write(ROOT/'stage_status.json',dict(status='awaiting_stage_decision',checkpoint_update=5000,comparison=str(ROOT/'comparison_at_5000.json'),automatic_resume=False,updated_unix=time.time()))
 print(json.dumps(metrics,indent=2),flush=True)
except BaseException as error:
 (HERE/'controller_exception.txt').write_text(traceback.format_exc())
 failure=dict(status='continuation_failed',error=str(error),controller_pid=os.getpid(),attempt=str(HERE),updated_unix=time.time(),automatic_restart=False)
 write(HERE/'controller_status.json',failure);write(ROOT/'stage_status.json',failure)
 raise
