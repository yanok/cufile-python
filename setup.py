from setuptools import setup, find_packages

setup(
    name="cufile-python",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[],
    author="Ilya Yanok",
    author_email="ilya.yanok@gmail.com",
    description="A basic Python wrapper for the NVidia cuFile API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yanok/cufile-python",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)
