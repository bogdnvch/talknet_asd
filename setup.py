from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.0",
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        'torch>=2.6.0,<2.7.0',  # Совместимо с ai-clips
        'torchaudio>=2.6.0,<2.7.0',  # Совместимо с ai-clips
        'torchvision>=0.21.0,<0.22.0',  # Точно соответствует ai-clips
        'numpy',
        'scipy',
        'scikit-learn',
        'tqdm',
        'scenedetect>=0.6.6',
        'opencv-python>=4.5.0',
        'python_speech_features>=0.6',
        'ffmpeg-python>=1.4',  # Используем ffmpeg-python вместо ffmpeg
        'gdown>=5.0.0',
        'youtube-dl>=2021.12.17',
        'pandas>=2.0.0'
    ],
    python_requires='>=3.12',
)
