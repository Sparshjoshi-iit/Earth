"""
Test EarthGPT Installation

Checks if all required packages are installed.
"""

import sys

def check_imports():
    """Check if all required packages can be imported."""

    required_packages = {
        'torch': 'torch>=2.1.0',
        'transformers': 'transformers>=4.38.0',
        'PIL': 'Pillow>=10.0.0',
        'numpy': 'numpy>=1.24.0',
        'cv2': 'opencv-python>=4.8.0',
        'datasets': 'datasets>=2.16.0',
        'peft': 'peft>=0.8.0',
        'accelerate': 'accelerate>=0.27.0',
        'bitsandbytes': 'bitsandbytes>=0.42.0',
    }

    print("=" * 80)
    print("EarthGPT Installation Test")
    print("=" * 80)
    print()

    missing = []
    installed = []

    for package, requirement in required_packages.items():
        try:
            __import__(package)
            installed.append(package)
            print(f"✅ {package:20s} - Installed")
        except ImportError:
            missing.append(requirement)
            print(f"❌ {package:20s} - Missing")

    print()
    print("=" * 80)

    if missing:
        print(f"Missing {len(missing)} packages:")
        print()
        print("Install with:")
        print("  pip install -r requirements.txt")
        print()
        print("Or individually:")
        for req in missing:
            print(f"  pip install {req}")
        print()
        return False
    else:
        print("✅ All packages installed successfully!")
        print()
        print("You're ready to use EarthGPT!")
        print()
        print("Next steps:")
        print("  1. Generate synthetic data: python scripts/generate_sample_data.py")
        print("  2. Or download real datasets: python scripts/download_datasets.py --all")
        print("  3. Train model: python training/train.py")
        print()
        return True


if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
