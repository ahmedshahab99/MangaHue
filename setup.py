"""Setup for manga_extractor - installable and reusable."""
from setuptools import setup, find_packages

setup(
    name="manga_extractor",
    version="2.0.0",
    description="Download manga from any site and forward to Telegram in batches of 10",
    author="Manga Bot",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.25",
        "beautifulsoup4>=4.12",
        "lxml>=5.0",
    ],
    extras_require={
        "bot": ["python-telegram-bot>=20.0"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
