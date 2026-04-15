"""Task package.

Import task modules here so they register themselves via @register_task.
"""

from tasks.pion_2pt import Pion2PtTask
from tasks.wilson_loop import WilsonLoopTask
from tasks.gsfit_2pt import Gsfit2PtTask

__all__ = ["Pion2PtTask", "WilsonLoopTask", "Gsfit2PtTask"]
