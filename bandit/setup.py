from setuptools import setup, find_packages
from collections import defaultdict
import os


# Build a list of all project modules, as well as supplementary files
main_package = "bandit"
extensions = {".aplx", ".boot", ".cfg", ".json", ".sql", ".template", ".xml",
              ".xsd"}
main_package_dir = os.path.join(os.path.dirname(__file__), main_package)
start = len(main_package_dir)
packages = []
package_data = defaultdict(list)
for dirname, dirnames, filenames in os.walk(main_package_dir):
    if '__init__.py' in filenames:
        package = "{}{}".format(
            main_package, dirname[start:].replace(os.sep, '.'))
        packages.append(package)
    for filename in filenames:
        _, ext = os.path.splitext(filename)
        if ext in extensions:
            package = "{}{}".format(
                main_package, dirname[start:].replace(os.sep, '.'))
            package_data[package].append(filename)

# **HACK** spynnaker doesn't have __version__ set properly
# therefore >= 3.0.0, < 4.0.0 doesn't work correctly
setup(
    name="bandit",
    version="0.1.1",
    license="GNU GPLv3.0",
    packages=find_packages(),
    package_data={'spinn_bandit.model_binaries': ['*.aplx']},
    install_requires=['sPynnaker8', "numpy",],
    classifiers = [
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",

        "Programming Language :: Python :: 2.7"
    ]
)
