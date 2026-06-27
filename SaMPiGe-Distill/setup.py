#!/usr/bin/env python3
"""
Setup script for SaMPiGe-Distill
Allows installation as a Python package
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

# Read long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='sampige-distill',
    version='0.1.0',
    description='Multi-Task Knowledge Distillation for Object Detection with DINOv2 and YOLO',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/sampige-distill',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'deep-learning',
        'computer-vision',
        'object-detection',
        'knowledge-distillation',
        'self-supervised-learning',
        'pytorch',
        'pytorch-lightning',
        'yolo',
        'dinov2',
        'vision-transformer'
    ],
    entry_points={
        'console_scripts': [
            'sampige-distill=sampige_distill.__main__:main',
        ],
    },
    package_data={
        'sampige_distill': [
            'config.py',
            'train.py',
            'validate.py',
            'infer.py',
            'requirements.txt',
            'README.md'
        ],
    },
    include_package_data=True,
)
