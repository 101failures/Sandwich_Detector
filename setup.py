from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sandwich_detector",
    version="0.1.0",
    author="Khalifa University CS07",
    description="BERT4ETH-based sandwich attack detection for Ethereum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/101failures/Sandwich_Detector",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    install_requires=[
        "tensorflow>=2.9.2",
        "numpy>=1.23.5",
        "pandas>=1.5.3",
        "scikit-learn>=1.2.2",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
        ],
    },
)
