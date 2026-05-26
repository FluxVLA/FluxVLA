from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location

_base_path = ('{{ fileDirname }}/'
              'openvla_dino_siglip_llama2_libero_10_full_finetune.py')
_base_spec = spec_from_file_location('openvla_libero_10_base', _base_path)
_base = module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base)

model = deepcopy(_base.model)
train_dataloader = deepcopy(_base.train_dataloader)
runner = deepcopy(_base.runner)
eval = deepcopy(_base.eval)

_dataset_name = 'libero_spatial_no_noops'
_task_suite_name = 'libero_spatial'

train_dataloader['dataset']['statistic_name'] = _dataset_name
train_dataloader['dataset'].pop('statistics_overrides', None)
train_dataloader['dataset']['datasets'][
    'data_root_path'] = './datasets/libero_spatial_no_noops_lerobotv2.1'
train_dataloader['dataset']['datasets']['statistic_name'] = _dataset_name
train_dataloader['dataset']['datasets']['transforms'][0][
    'dataset_name'] = _dataset_name

runner['eval']['task_suite_name'] = _task_suite_name
eval['task_suite_name'] = _task_suite_name
