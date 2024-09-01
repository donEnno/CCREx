from setuptools import setup, find_packages

setup(
    name='ccrex',
    version='0.1.0',
    description='A module for predicting on protein sequences using CCREx models.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ccrex',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'keras',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)