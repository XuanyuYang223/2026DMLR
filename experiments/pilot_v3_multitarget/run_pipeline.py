"""Resume the finite pilot, locking development evidence before formal training."""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path
import pilot_v3 as p

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=p.OUT)
    args=parser.parse_args()
    p.OUT=args.out.resolve()
    for stage,marker in p.STAGES:
        if (p.OUT/marker).exists():
            print('Skip completed:',stage,flush=True)
        else:
            start=time.monotonic()
            print('Start:',stage,flush=True)
            try:
                getattr(p,stage)()
            except Exception:
                p.write(p.OUT/f'FAILURE_{stage}_{int(time.time())}.json',dict(stage=stage,traceback=traceback.format_exc()))
                raise
            p.write(p.OUT/f'timing_{stage}.json',dict(stage=stage,seconds=time.monotonic()-start))
            print('Finished:',stage,'seconds',time.monotonic()-start,flush=True)
        if stage=='freeze':
            start=time.monotonic()
            subprocess.run([sys.executable,str(p.HERE/'freeze_evidence.py'),'--out',str(p.OUT)],check=True)
            if not (p.OUT/'timing_freeze_evidence.json').exists():
                p.write(p.OUT/'timing_freeze_evidence.json',dict(stage='freeze_evidence',seconds=time.monotonic()-start))
    p.preservation_check()
    print('Finite v3 complete; old files byte preserved; no expansion scheduled.',flush=True)

if __name__=='__main__':main()
