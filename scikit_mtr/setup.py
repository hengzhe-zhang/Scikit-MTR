from setuptools import setup, find_packages

setup(
    name='scikit-MTR',
    version='0.1.0',
    packages=find_packages(),
    author='Hengzhe Zhang',
    author_email='',
    description='A sklearn-compatible framework for multi-task regression in Python.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/hengzhe-zhang/Scikit-MTR',  # Replace with your GitHub repo URL
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    python_requires='>=3.6',
)
