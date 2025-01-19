from setuptools import setup, find_packages

setup(
    name="amt-augpy",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'librosa==0.9.1',  # Specific version for compatibility with torchcrepe
        'soundfile>=0.12.1',
        'numpy==1.23.5',   # More compatible version
        'pedalboard>=0.7.3',
        'pretty_midi>=0.2.10',
        'tqdm>=4.65.0',
        'numba==0.56.4',   # Added for compatibility
        'llvmlite==0.39.1' # Added for compatibility
    ],
    author="Lars Monstad",
    author_email="your.email@example.com",
    description="A toolkit for augmenting AMT datasets with various audio transformations",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/LarsMonstad/amt-augpy1.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'amt-augpy=amt_augpy.main:main',
        ],
    }
)
