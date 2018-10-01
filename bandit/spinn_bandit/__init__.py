
from model_binaries import __file__ as binaries_path
from python_models.bandit import Bandit
# from visualiser.visualiser import Visualiser
# from visualiser.visualiser_subsamp import Visualiser_subsamp
import os

from spynnaker.pyNN.abstract_spinnaker_common import AbstractSpiNNakerCommon

AbstractSpiNNakerCommon.register_binary_search_path(os.path.dirname(binaries_path))

# This adds the model binaries path to the paths searched by sPyNNaker
# ef = executable_finder.ExecutableFinder(os.path.dirname(binaries_path))
# executable_finder.add_path(os.path.dirname(binaries_path))
