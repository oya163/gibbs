import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gibbs",
    version="0.0.1",
    author="Oyesh Mann Singh",
    author_email="osingh1@umbc.edu",
    description="Non-parametric bayesian segmentation model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oya163/gibbs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pytest'
    ],
    include_package_data=True,
    python_requires='>=3.7',
)