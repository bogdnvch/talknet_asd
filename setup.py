from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.0",
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        'torch>=1.6.0',
        'torchaudio>=0.6.0',
        'numpy',
        'scipy',
        'scikit-learn',
        'tqdm',
        'scenedetect',
        'opencv-python',
        'python_speech_features',
        'torchvision',
        'ffmpeg',
        'gdown',
        'youtube-dl',
        'pandas'
    ],
    python_requires='>=3.12',
)
