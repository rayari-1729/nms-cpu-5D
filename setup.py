from setuptools import setup, find_packages

setup(
    name='cpu-rotated-nms',
    version='0.1.0',
    description='A CPU-based 3D NMS implementation with rotation (5D input)',
    author='Aritra Roy',
    author_email='',
    url='https://github.com/rayari-1729/nms-cpu-5D.git',
    license='MIT',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'torch>=1.13.0',
    ],
    python_requires='>=3.8',
)
