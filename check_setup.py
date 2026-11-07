"""Check packages and trained-model integrity. --camera discovers OAKs without opening them."""
import argparse
import importlib.metadata
import json
from pathlib import Path
from run_dashboard import model_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', action='store_true')
    args = parser.parse_args()
    from dashboard.runtime import local_runtime
    local_runtime()
    import cv2
    import depthai as dai
    import torch
    import ultralytics
    for name in ('ultralytics', 'torch', 'torchvision', 'depthai', 'numpy', 'opencv-python'):
        print(name, importlib.metadata.version(name))
    if dai.__version__.split('.')[0] != '3':
        parser.error('DepthAI v3 is required. Reinstall requirements-dashboard.txt.')
    for profile in ('baseline', 'usb'):
        weights, confidence = model_config(profile)
        print(profile, 'checkpoint SHA-256 verified, threshold', confidence)
    print('Compute:', 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if args.camera:
        devices = dai.Device.getAllAvailableDevices()
        print('Available OAK cameras:', json.dumps([{'id': d.deviceId, 'state': str(d.state)} for d in devices]))
        if not devices:
            parser.exit(2, 'No available OAK found. Check the data cable/power and close other camera programs.\n')
        if len(devices) > 1:
            print('Choose one camera with run_dashboard.py --oak-device ID.')
        print('Discovery is not a live-frame check. Start the dashboard and confirm LIVE OAK with a fresh image.')


if __name__ == '__main__':
    main()
