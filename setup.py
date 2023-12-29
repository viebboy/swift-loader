import setuptools
from swift_loader.version import __version__


setuptools.setup(
    name="swift_loader",
    version=__version__,
    author="Dat Tran",
    author_email="hello@dats.bio",
    description="Multiprocess data loader for ML training",
    long_description="Multiprocess data loader for ML training",
    long_description_content_type="text",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: POSIX",
    ],
)
