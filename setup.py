# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open("README.md","r",encoding="utf-8-sig") as f:
    readme = f.read()

requirements = [
    "Cython==0.29.30",
    "librosa==0.10.0",
    "scipy==1.14.0",
    "numpy==1.24.3",
    "phonemizer>=3.2.1",
    "torch>=2.3.1",
    "Unidecode>=1.3.7",
    "monotonic-align>=1.0.0",
]

setup(
    name="py3-ttsmms",
    version="0.8",
    description="Text-to-speech with The Massively Multilingual Speech (MMS) project",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="willwade",
    author_email="willwade@gmail.com",
    url="https://github.com/willwade/ttsmms",
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
    dynamic=[
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
        "Source": "https://github.com/willwade/ttsmms",
    },
)
