from setuptools import setup, find_packages
from app import __version__


setup(
    name="mlx-openai-server",
    version=__version__,
    description="OpenAI-compatible server for MLX models",
    author="Huy Vuong",
    packages=find_packages(), 
    install_requires=[
        "mlx-vlm==0.1.26",
        "mlx-lm==0.24.1",
        "torchvision",
        "fastapi",
        "uvicorn",
        "Pillow",
        "click",
        "loguru",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            "mlx-openai-server=app.cli:cli",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
) 
