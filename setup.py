from setuptools import setup, find_packages

setup(
    name="talknet_asd",
    version="0.1.1",  # Увеличиваем версию для обновления
    description="Active Speaker Detection using TalkNet",
    author="Maksim Bogdanovich",
    url="https://github.com/bogdnvch/talknet_asd",
    packages=find_packages(),
    install_requires=[
        # Основные зависимости, совместимые с ai-clips
        'torch==2.7.0',  # Точно соответствует ai-clips
        'torchaudio==2.7.0',  # Точно соответствует ai-clips
        'torchvision==0.22.0',  # Точно соответствует ai-clips
        
        # Остальные зависимости с версиями из вашего окружения
        'numpy==2.2.5',
        'scipy==1.13.1',
        'scikit-learn==1.6.1',
        'tqdm==4.67.1',
        'scenedetect==0.6.6',
        'opencv-python==4.11.0.86',
        'python_speech_features==0.6',
        'ffmpeg-python==0.2.0',  # Более стабильная альтернатива ffmpeg
        'gdown==5.0.0',
        'youtube-dl==2021.12.17',
        'pandas==2.2.3',
        
        # Дополнительные зависимости для полной совместимости
        'pillow==10.4.0',  # Совместимость с moviepy
        'imageio==2.37.0',
        'imageio-ffmpeg==0.6.0'
    ],
    python_requires='>=3.12',
)
