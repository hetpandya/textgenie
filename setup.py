from setuptools import setup
from textgenie import __version__ as version

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="textgenie",
    version=version,
    description="A python library to augment text data using NLP.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Het Pandya",
    url="http://github.com/hetpandya/TextGenie",
    author_email="hetpandya6797@gmail.com",
    license="MIT",
    install_requires=[
        "torch>=1.5.0",
        "transformers",
        "sentencepiece",
        "spacy==2.2.4",
        "tqdm",
        "pattern==3.6",
    ],
    packages=["textgenie"],
)

