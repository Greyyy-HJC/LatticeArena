"""Task package.

Import task modules here so they register themselves via @register_task.
"""

from tasks.pion_2pt import Pion2PtTask
from tasks.two_pt_gsfit import TwoPtGsfitTask

__all__ = ["Pion2PtTask", "TwoPtGsfitTask"]
