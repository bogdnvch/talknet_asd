from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.2.2",
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
        "numpy>=1.23.5,<2.1.1",
        "ffmpeg>=1.4,<2.0",
        "ultralytics>=8.3.117,<8.4.0",
        "huggingface-hub>=0.30.2",
        "deepface>=0.0.93,<0.0.94",
        "tf-keras>=2.15.0,<3.0.0",
        "batch-face @ git+https://github.com/elliottzheng/batch-face.git@master",
    ],
    python_requires=">=3.12",
)
