"""Task package.

Import task modules here so they register themselves via @register_task.
"""

from tasks.pion_2pt import Pion2PtTask

__all__ = ["Pion2PtTask"]
