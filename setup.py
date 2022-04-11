from setuptools import setup

setup(
    name = "NeuralCompress",
    version = "0.0.1",
    author = "Yi Huang",
    author_email = "TBD",
    description = ("Implementation of neural compression"),
    license = "MIT",
    keywords = "example documentation tutorial",
    url = "https://github.com/BNL-DAQ-LDRD/NeuralCompression",
    packages=['neuralcompress',],
    long_description="",
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "tqdm"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
		"Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
