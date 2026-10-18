"""Sequential execution in one interpreter; log stages without overwriting data."""
import argparse, json, time, traceback
from pathlib import Path
import pilot

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=pilot.HERE/'runs'/'pilot_v1');a=p.parse_args();root=a.out.resolve();root.mkdir(parents=True,exist_ok=True)
stages=[('prepare','data_audit.json'),('smoke','smoke_audit.json'),('develop','development_result.json'),('freeze','FROZEN_PROTOCOL.json'),('formal','TRAINING_COMPLETE.json'),('results','COMPLETED.json')]
for stage,marker in stages:
    if (root/marker).exists():
        print('Skipping completed stage:',stage,flush=True);continue
    started=time.time();print('Starting:',stage,flush=True)
    try: getattr(pilot,stage)(root)
    except Exception:
        pilot.write(root/'LAST_FAILURE.json',dict(stage=stage,unix=time.time(),traceback=traceback.format_exc()));raise
    print('Completed:',stage,'seconds=',time.time()-started,flush=True)
