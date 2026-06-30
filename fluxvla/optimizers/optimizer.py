# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.optim as optim

from fluxvla.engines.utils.root import OPTIMIZERS

_OPTIMIZER_ALIASES = {
    'SGD': ['SGD', 'sgd'],
    'Adam': ['Adam', 'adam'],
    'AdamW': ['AdamW', 'adamw'],
    'Adamax': ['Adamax', 'adamax'],
    'NAdam': ['NAdam', 'nadam'],
    'RAdam': ['RAdam', 'radam'],
    'RMSprop': ['RMSprop', 'rmsprop'],
    'Rprop': ['Rprop', 'rprop'],
    'Adagrad': ['Adagrad', 'adagrad'],
    'Adadelta': ['Adadelta', 'adadelta'],
    'ASGD': ['ASGD', 'asgd'],
    'LBFGS': ['LBFGS', 'lbfgs'],
    'Adafactor': ['Adafactor', 'adafactor'],
}

for optimizer_name, aliases in _OPTIMIZER_ALIASES.items():
    optimizer_cls = getattr(optim, optimizer_name, None)
    if optimizer_cls is not None:
        OPTIMIZERS.register_module(name=aliases, module=optimizer_cls)
