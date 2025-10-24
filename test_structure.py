#!/usr/bin/env python3
"""
Test project structure without requiring dependencies
"""
import os

def check_file(filepath):
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists

print("=" * 60)
print("VSLIM Project Structure Check")
print("=" * 60)

all_ok = True

print("\n1. Core Python modules:")
files = [
    "main.py",
    "trainer.py",
    "data_loader.py",
    "utils.py"
]
all_ok &= all(check_file(f) for f in files)

print("\n2. VSLIM package:")
vslim_files = [
    "vslim/__init__.py",
    "vslim/models/__init__.py",
    "vslim/models/slim.py",
    "vslim/models/layers.py",
    "vslim/processors/__init__.py",
    "vslim/processors/label_loader.py",
    "vslim/metrics/__init__.py",
    "vslim/metrics/metrics.py"
]
all_ok &= all(check_file(f) for f in vslim_files)

print("\n3. Configuration files:")
config_files = [
    "requirements.txt",
    "run_vslim_train.sh",
    "VSLIM_USAGE.md"
]
all_ok &= all(check_file(f) for f in config_files)

print("\n4. Data files:")
data_files = [
    "data/vped/intent_label.txt",
    "data/vped/slot_label.txt"
]
all_ok &= all(check_file(f) for f in data_files)

print("\n" + "=" * 60)
if all_ok:
    print("✓ SUCCESS: All files present!")
    print("\nProject is ready. To install dependencies:")
    print("  pip install -r requirements.txt")
    print("\nTo train:")
    print("  bash run_vslim_train.sh")
else:
    print("✗ FAILED: Some files missing!")

