import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pils",
    version="0.1.0",
    author="Julien Jerphanion",
    author_email="git@jjerphan.xyz",
    description="pils : Propulsing Iterated Local Search 💊",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jjerphan/pils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
