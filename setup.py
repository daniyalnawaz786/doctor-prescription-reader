from setuptools import find_packages, setup

setup(
    name="doctor-prescription-reader",
    version="0.1.0",
    description="Fine-tuned TrOCR for reading handwritten medicine names from prescriptions",
    author="Daniyal",
    packages=find_packages(exclude=["notebooks", "assets", "data"]),
    python_requires=">=3.9",
)
