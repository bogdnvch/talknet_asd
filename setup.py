from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.1",  # Увеличьте версию, чтобы Poetry точно подхватил изменения
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        'torch==2.6.0',  # Точно как в ai-clips
        'torchaudio==2.6.0',  # Точно как в ai-clips
        'torchvision==0.21.0',  # Точно как в ai-clips
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'tqdm>=4.0.0',
        'scenedetect>=0.6.0',
        'opencv-python>=4.5.0',
        'python_speech_features>=0.6',
        'ffmpeg-python>=1.4',
        'gdown>=5.0.0',
        'youtube-dl>=2021.12.17',
        'pandas>=2.0.0'
    ],
    python_requires='>=3.12',
)
