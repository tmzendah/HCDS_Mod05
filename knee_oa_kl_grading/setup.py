from setuptools import setup, find_packages

setup(
    name="knee_oa_kl_grading",
    version="0.1.0",
    author="Talita Mzendah",
    description="Automated Kellgren-Lawrence grading of knee OA from clinical radiographs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.2",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "grad-cam>=1.4.8",
        "pyyaml>=6.0",
    ],
)
