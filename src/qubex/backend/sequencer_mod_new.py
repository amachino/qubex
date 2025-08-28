from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..experiment.experiment_exceptions import BackendUnavailableError

if TYPE_CHECKING:
    from qubecalib.qubecalib import Sequencer


def _ensure_qubecalib_base():
    """Import qubecalib base with error handling."""
    try:
        import qubecalib.qubecalib as qbc
        return qbc
    except ImportError as e:
        raise BackendUnavailableError(
            "qubecalib is required for sequencer functionality."
        ) from e


def create_sequencer_mod(**kwargs) -> Any:
    """Create a SequencerMod instance with lazy loading."""
    qbc = _ensure_qubecalib_base()
    
    # Import the original class and create a dynamic subclass
    # This way we preserve all the original functionality
    class SequencerMod(qbc.Sequencer):
        pass  # Use the base implementation for now
    
    return SequencerMod(**kwargs)


# For backward compatibility when not using the factory
SequencerMod = None  # Will be set dynamically when needed