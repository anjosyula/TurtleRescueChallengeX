"""Create a project-local camera/dashboard environment. Run with Python 3.12."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def environment_python(folder):
    return folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=ROOT / '.venv')
    parser.add_argument('--cuda', action='store_true', help='Use the standard PyPI GPU build on Windows/Linux instead of CPU wheels')
    parser.add_argument('--check-only', action='store_true', help='Check an existing environment without installing')
    args = parser.parse_args()
    if sys.version_info[:2] not in ((3, 12), (3, 13)):
        parser.error('Run with Python 3.12 (recommended) or 3.13; on Windows use py -3.12 setup_dashboard.py')
    folder = args.env.resolve()
    python = environment_python(folder)
    if not args.check_only:
        if not python.exists():
            print('Creating the project Python environment...', flush=True)
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(folder)
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        if sys.platform != 'darwin' and not args.cuda:
            subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=:all:',
                '--index-url', 'https://download.pytorch.org/whl/cpu',
                'torch==2.14.0', 'torchvision==0.29.0'], check=True)
        subprocess.run([str(python), '-m', 'pip', 'install',
            '--only-binary=depthai,torch,torchvision,numpy,opencv-python',
            '-r', str(ROOT / 'requirements-dashboard.txt')], check=True)
    if not python.is_file():
        parser.error('The project environment is missing. Run setup without --check-only.')
    subprocess.run([str(python), str(ROOT / 'check_setup.py')], check=True, cwd=ROOT)
    print('\nReady. Connect the OAK with a USB data cable, then run:')
    print('  ' + ('py -3.12' if os.name == 'nt' else 'python3.12') + ' run_dashboard.py')


if __name__ == '__main__':
    main()
