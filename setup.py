from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.5",
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        "python-speech-features>=0.6,<0.7",
        "scipy>=1.15.2,<2.0.0",
        "opencv-python>=4.11.0.86,<5.0.0.0",
        "scenedetect>=0.6.6,<0.7.0",
        "pandas>=2.2.3,<3.0.0",
        "gdown>=5.2.0,<6.0.0",
        "tqdm>=4.67.1,<5.0.0",
        "numpy>=1.23.5, <2.1.1",
        "ffmpeg>=1.4,<2.0",
        "torch>=2.6.0,<3.0.0",
        "torchaudio>=2.6.0,<3.0.0",
        "torchvision>=0.21.0,<0.22.0",
    ],
    python_requires='>=3.12',
)
