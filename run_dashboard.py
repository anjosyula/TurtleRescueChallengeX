"""Start the camera dashboard using this checkout's environment and bundled turtle weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROFILES = {'baseline': 'rtdetr-baseline', 'usb': 'rtdetr-usb-background-20261106'}


def model_config(profile):
    folder = ROOT / 'models' / PROFILES[profile]
    card = json.loads((folder / 'model.json').read_text())
    weights = folder / card['weights']
    if not weights.is_file():
        raise RuntimeError('The trained model is missing. Fetch this repository again.')
    with weights.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checksum != card['sha256']:
        raise RuntimeError('The trained model is damaged. Fetch this repository again.')
    return weights, card['confidence']


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--model', choices=PROFILES, default='baseline')
    parser.add_argument('--python', type=Path, help='Use an existing configured environment instead of .venv')
    args, extra = parser.parse_known_args()
    if '--help' in extra or '-h' in extra:
        print('TurtleHero launcher: --model baseline|usb, --python PATH')
        print('Other options (such as --source oak, --port 8765) are passed to the dashboard.\n')
        return subprocess.call([sys.executable, str(ROOT / 'turtlehero_dashboard.py'), '--help'])
    python = args.python or ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        parser.error('Run python3.12 setup_dashboard.py first (Windows: py -3.12 setup_dashboard.py).')
    try:
        weights, confidence = model_config(args.model)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    command = [str(python.absolute()), '-u', str(ROOT / 'turtlehero_dashboard.py'),
               '--weights', str(weights), '--confidence', str(confidence),
               '--oak-python', str(python.absolute()), *extra]
    # Replace the launcher so Ctrl+C reaches the server directly and its camera
    # child receives the server's normal cleanup instead of being orphaned.
    os.chdir(ROOT)
    os.execv(command[0], command)


if __name__ == '__main__':
    raise SystemExit(main())
