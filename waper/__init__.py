import warnings

warnings.filterwarnings("ignore", message=".*n_faces.*")

# The filter above has to be installed before the vtk/pyvista import chain runs.
from .interface.api import Waper, WaperConfig, WaperSingleTimestepData  # noqa: E402

__all__ = ["Waper", "WaperConfig", "WaperSingleTimestepData"]
