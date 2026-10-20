"""Optional RLinf backend integration; import through the RL entrypoints."""

import os
import sys
import types

# TensorBoard's supported no-TensorFlow switch is the presence of this module.
# Honor USE_TF=0 without changing the shared TensorFlow installation.
# (whose protobuf requirement differs from current W&B/Ray dependencies).
if os.environ.get('USE_TF') == '0':
    sys.modules.setdefault('tensorboard.compat.notf',
                           types.ModuleType('tensorboard.compat.notf'))
