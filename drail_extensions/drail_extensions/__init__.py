"""
Python module serving as a project/extension template.
"""


# Register UI extensions.
# from .load_drial_extension import *

import omni.log
import traceback

# Register Gym environments.
from .core.isaaclab_tasks.tasks import *

try:
    from .research_ashton.tasks import *
except ImportError as e:
    omni.log.error("Failed to load some tasks from research_ashton, please debug this locally.")
    omni.log.error(traceback.format_exc())
