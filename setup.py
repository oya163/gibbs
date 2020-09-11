import setuptools

setuptools.setup(
    name="gibbs",
    version="0.0.1",
    author="Oyesh Mann Singh",
    author_email="osingh1@umbc.edu",
    description="Non-parametric bayesian segmentation model",
    url="https://github.com/oya163/gibbs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=[
    #     'pytest'
    # ],
    python_requires='>=3.7',
)