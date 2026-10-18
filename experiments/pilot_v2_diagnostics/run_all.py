import argparse, time, traceback
from pathlib import Path
import diagnose as d

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=d.OUT)
args=parser.parse_args()
d.OUT=args.out.resolve()
stages=[('inventory','inventory.json'),('smoke','smoke.json'),('freeze','FROZEN_PROTOCOL.json'),('train','TRAINING_COMPLETE.json'),('select','SELECTIONS_FROZEN.json'),('test','TEST_COMPLETE.json')]
d.OUT.mkdir(parents=True,exist_ok=True)
for stage,marker in stages:
    if (d.OUT/marker).exists():
        print('Skip completed:',stage,flush=True);continue
    started=time.monotonic();print('Start:',stage,flush=True)
    try:getattr(d,stage)()
    except Exception:
        d.write(d.OUT/f'FAILURE_{stage}_{int(time.time())}.json',dict(stage=stage,traceback=traceback.format_exc()));raise
    d.write(d.OUT/f'timing_{stage}.json',dict(stage=stage,wall_seconds=time.monotonic()-started))
    print('Finished:',stage,'seconds=',time.monotonic()-started,flush=True)
d.preservation_check()
print('V1 byte preservation passed.',flush=True)
