import re
from setuptools import setup, find_packages


PACKAGE_NAME = 'qdqonnx_scale_0to1'
APP_NAME = 'qdq_scale0to1'


def _normalize(pkg_name):
    return re.sub(r'[-_.]+', '-', pkg_name).lower()


setup(
    name=_normalize(PACKAGE_NAME),
    packages=find_packages(),
    description='convert scale=0 to 1 in the QDQ ONNX file',
    long_description='1. fold constants in the ONNX file\n2. convert scale=0 to 1',
    long_description_content_type='text/markdown',
    url='https://github.com/maminus/qdqonnx_scale_0to1',
    license='MIT',
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'onnx',
        'onnx-graphsurgeon',
    ],
    extras_require={
        # pip install -e .[test]
        'test': [
            'pytest',
            'pytest-cov',
            'flake8',
        ],
    },
    entry_points={
        'console_scripts': [
            f'{APP_NAME}={PACKAGE_NAME}.__main__:main',
        ],
    },
)
