from setuptools import setup, find_packages
from app import __version__


setup(
    name="mlx-openai-server",
    url="https://github.com/cubist38/mlx-openai-server",
    author="Huy Vuong",
    author_email="cubist38@gmail.com",
    version=__version__,
    description="A high-performance API server that provides OpenAI-compatible endpoints for MLX models. Built with Python and FastAPI, it enables efficient, scalable, and user-friendly local deployment of MLX-based multimodal models with an OpenAI-compatible interface. Supports text, vision, and audio processing capabilities. Perfect for developers looking to run MLX models locally while maintaining compatibility with existing OpenAI-based applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(), 
    install_requires=[
        "mlx-vlm==0.3.0",
        "mlx-lm==0.25.3",
        "mflux==0.9.3",
        "fastapi==0.115.14",
        "uvicorn==0.35.0",
        "Pillow==10.4.0",
        "click==8.2.1",
        "loguru==0.7.3",
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
