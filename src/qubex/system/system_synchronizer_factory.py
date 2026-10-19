"""Construct system synchronizers through extensible controller-type dispatch."""

from functools import singledispatch

from qubex.backend.quel1 import Quel1BackendController
from qubex.backend.quel3 import Quel3BackendController
from qubex.system.quel1.quel1_system_synchronizer import (
    create_system_synchronizer as create_quel1_system_synchronizer,
)
from qubex.system.quel3.quel3_system_synchronizer import (
    create_system_synchronizer as create_quel3_system_synchronizer,
)
from qubex.system.system_synchronizer import SystemSynchronizer


@singledispatch
def create_system_synchronizer(backend_controller: object) -> SystemSynchronizer | None:
    """
    Construct the synchronizer registered for the controller's type.

    Register additional builders with
    `create_system_synchronizer.register(ControllerType, builder)` during
    application initialization. Subclasses inherit their parent's registration
    unless a more specific builder is registered. Unknown types return `None`.
    """
    return None


create_system_synchronizer.register(
    Quel1BackendController, create_quel1_system_synchronizer
)
create_system_synchronizer.register(
    Quel3BackendController, create_quel3_system_synchronizer
)
