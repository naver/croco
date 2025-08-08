import os
from setuptools import setup, find_packages


ROOT = os.path.abspath(os.path.dirname(__file__))
LOCAL_CUROPE = os.path.isdir(os.path.join(ROOT, "curope"))

if LOCAL_CUROPE:
    # Use a file:// URL to install the curope folder directly
    curope_dep = f"curope @ file://{os.path.join(ROOT, 'curope')}"
else:
    # Fallback to fetching curope from URL
    curope_dep = (
        "curope @ git+https://github.com/naver/croco.git@croco_module"
        "#egg=curope&subdirectory=curope"
    )

setup(
    name="croco",
    version="1.0.0",
    packages=find_packages(include=["croco", "croco.*"]),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'numpy',
        'numpy-quaternion',
        'opencv-python',
        'einops',
        'tensorboard',
        'h5py',
        'pillow'
    ],
    python_requires=">=3.7",
    extras_require={
        "curope": [curope_dep],
        "all": [curope_dep],
    },
)