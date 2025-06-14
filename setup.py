from setuptools import setup, find_packages
from app import __version__


setup(
    name="mlx-openai-server",
    url="https://github.com/cubist38/mlx-openai-server",
    author="Huy Vuong",
    author_email="cubist38@gmail.com",
    version=__version__,
    description="A high-performance API server that provides OpenAI-compatible endpoints for MLX models. Built with Python and FastAPI, it enables efficient, scalable, and user-friendly local deployment of MLX-based vision and language models with an OpenAI-compatible interface. Perfect for developers looking to run MLX models locally while maintaining compatibility with existing OpenAI-based applications.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(), 
    install_requires=[
        "mlx-vlm==0.1.27",
        "mlx-lm==0.25.2",
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
