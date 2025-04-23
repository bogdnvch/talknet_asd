from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.2",  # Новая версия
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        'torch>=2.6.0,<2.8.0',
        'torchaudio>=2.6.0,<2.8.0',
        'torchvision>=0.21.0,<0.23.0',
        
        # Audio/video processing
        'opencv-python>=4.5.0',
        'python_speech_features>=0.6',
        'ffmpeg-python>=0.2.0',
        
        # Data processing
        'numpy>=2.0.0',
        'pandas>=2.0.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        
        # Utilities
        'tqdm>=4.0.0',
        'gdown>=5.0.0',
        'youtube-dl>=2021.12.17',
        
        # Scene detection
        'scenedetect>=0.6.0',
        
        # Image processing (compatible version)
        'pillow>=10.0.0,<11.0.0'  # Специально для совместимости с moviepy
    ],
    python_requires='>=3.12',
)
