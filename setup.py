import subprocess
from pathlib import Path
from datetime import datetime
from setuptools import setup, find_packages


def shell(*args):
    out = subprocess.check_output(args)
    return out.decode("ascii").strip()


def write_version(version_core, pre_release=True):
    if pre_release:
        time = shell("git", "log", "-1", "--format=%cd", "--date=iso")
        time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S %z")
        time = time.strftime("%Y%m%d%H%M%S")
        version = f"{version_core}-dev{time}"
    else:
        version = version_core

    with open(Path("vall_e", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))

    return version


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="vall-e",
    python_requires=">=3.10.0",
    version=write_version("0.0.1"),
    description="An unofficial toy implementation of the audio LM VALL-E",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "coloredlogs>=15.0.1",
        "deepspeed>=0.7.7",
        "diskcache>=5.4.0",
        "einops>=0.6.0",
        "encodec>=0.1.1",
        "g2p_en>=2.1.0",
        "humanize>=4.4.0",
        "matplotlib>=3.6.0",
        "numpy>=1.23.3",
        "omegaconf>=2.2.3",
        "openTSNE>=0.6.2",
        "pandas>=1.5.0",
        "soundfile>=0.11.0",
        "torch>=1.13.0",
        "torchaudio>=0.13.0",
        "tqdm>=4.64.1",
    ],
    url="https://github.com/enhuiz/vall-e",
)
