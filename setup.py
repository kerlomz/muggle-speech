#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

data_files = [
    ('muggle_speech', [
        'muggle_speech/decoder.onnx',
        'muggle_speech/encoder.onnx',
    ])
]
setup(
    name="muggle_speech",
    version="0.1",
    author="kerlomz",
    description="麻瓜语音",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kerlomz/muggle-speech",
    packages=find_packages(where='.', exclude=(), include=('*',)),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    data_files=data_files,
    install_requires=['numpy', 'onnxruntime', 'librosa', 'wave', 'pydub', 'resampy'],
    python_requires='<3.10',
    include_package_data=True,
)