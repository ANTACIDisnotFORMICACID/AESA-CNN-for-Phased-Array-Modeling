"""Resume the authorized campaign and finish its evaluation and report."""
import argparse
import subprocess
import sys
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args()
scripts=Path(__file__).parent
for action in ['train','evaluate']:
    subprocess.run([sys.executable,'-B',str(scripts/'beam_campaign.py'),action,'--root',str(args.root)],check=True)
subprocess.run([sys.executable,'-B',str(scripts/'beam_diagnostics.py'),'--root',str(args.root)],check=True)
print('CAMPAIGN_COMPLETE',flush=True)
