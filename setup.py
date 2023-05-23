# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open("README.md","r",encoding="utf-8-sig") as f:
    readme = f.read()

requirements = [
    "Cython",
    "librosa",
    "scipy",
    "numpy",
    "phonemizer",
    "torch",
    "torchvision",
    "Unidecode",
    "monotonic-align"
]

setup(
    name="ttsmms",
    version="0.2",
    description="Text-to-speech with The Massively Multilingual Speech (MMS) project",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Wannaphong",
    author_email="wannaphong@yahoo.com",
    url="https://github.com/wannaphong/ttsmms",
    packages=find_packages(),
    test_suite="tests",
    python_requires=">=3.6",
    # package_data={
    #     "laonlp": [
    #         "corpus/*",
    #     ]
    # },
    install_requires=requirements,
    license="MIT License",
    zip_safe=False,
    keywords=[
        "tts",
        "NLP",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Linguistic",
    ],
    project_urls={
        "Source": "https://github.com/wannaphong/ttsmms",
    },
)
