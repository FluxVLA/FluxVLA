"""RLinf worker and simulator integrations used by RLinfTrainRunner."""

import os
import sys
import types

if os.environ.get('USE_TF') == '0':
    sys.modules.setdefault('tensorboard.compat.notf',
                           types.ModuleType('tensorboard.compat.notf'))

from .env_worker import FluxEnvWorker
from .rollout_worker import FluxRolloutWorker

__all__ = ['FluxEnvWorker', 'FluxRolloutWorker']
