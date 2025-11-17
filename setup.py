from setuptools import setup, find_packages

setup(
    name="nnUNet",
    version="2.1.0",
    author="Fabian Isensee et al.",
    description="nnU-Net: self-configuring deep learning for biomedical image segmentation",
    packages=find_packages(include=["nnunetv2", "nnunetv2.*"]),
    python_requires=">=3.9",
)
