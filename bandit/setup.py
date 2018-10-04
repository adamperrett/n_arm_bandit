from setuptools import setup, find_packages

# **HACK** spynnaker doesn't have __version__ set properly
# therefore >= 3.0.0, < 4.0.0 doesn't work correctly
setup(
    name="spinn_bandit",
    version="0.1.1",
    license="GNU GPLv3.0",
    packages=find_packages(),
    package_data={'spinn_bandit.model_binaries': ['*.aplx']},
    install_requires=['spynnaker7', "numpy",],
    classifiers = [
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2.7"
    ]
)
