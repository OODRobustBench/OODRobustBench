import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="oodrb",
    version="0.1",
    author="Lin Li",
    author_email="linli.tree@outlook.com",
    description="This package provides the data for RobustBench together with the model zoo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/OODRobustBench/OODRobustBench",
    packages=setuptools.find_packages(),
    install_requires=[
        "robustbench@git+https://github.com/RobustBench/robustbench.git",
        "robustness@git+https://github.com/MadryLab/robustness.git",
        "perceptual-advex@git+https://github.com/cassidylaidlaw/perceptual-advex.git",
        "addict",
        "einops",
        "frozendict",
        "zenodo_get"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
