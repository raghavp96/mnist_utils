import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mnist-utils",
    version="0.0.1",
    author="Raghavprasanna Rajagopalan",
    author_email="raghavp96@gmail.com",
    description="A small package for working with MNIST data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raghavp96/mnist_utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.2.2"
    ]
)
