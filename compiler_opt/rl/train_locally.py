# coding=utf-8
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Train and Eval LLVM Inliner decision rule with local_data_collector."""

import collections
import functools
import json
import os
import time
import threading
from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from typing import List

from compiler_opt.rl import agent_creators
from compiler_opt.rl import compilation_runner
from compiler_opt.rl import constant
from compiler_opt.rl import data_reader
from compiler_opt.rl import gin_external_configurables  # pylint: disable=unused-import
from compiler_opt.rl import local_data_collector
from compiler_opt.rl import policy_saver
from compiler_opt.rl import problem_configuration
from compiler_opt.rl import random_net_distillation
from compiler_opt.rl import registry
from compiler_opt.rl import trainer

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_string('data_path', None,
                    'Path to CNS folder containing IR files.')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel data collection workers. `None` for max available')
flags.DEFINE_integer('num_modules', 100,
                     'Number of modules to collect data for each iteration.')
flags.DEFINE_multi_string('gin_files', [],
                          'List of paths to gin configuration files.')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files.')

FLAGS = flags.FLAGS


@gin.configurable
def train_eval(agent_name=constant.AgentName.PPO,
               warmstart_policy_dir=None,
               num_policy_iterations=0,
               num_iterations=100,
               batch_size=64,
               train_sequence_length=1,
               deploy_policy_name='saved_policy',
               use_random_network_distillation=False,
               moving_average_decay_rate=1):
  """Train for LLVM inliner."""
  root_dir = FLAGS.root_dir
  problem_config = registry.get_configuration()
  time_step_spec, action_spec = problem_config.get_signature_spec()
  preprocessing_layer_creator = problem_config.get_preprocessing_layer_creator()

  # Initialize trainer and policy saver.
  tf_agent1 = agent_creators.create_agent(agent_name, time_step_spec,
                                         action_spec,
                                         preprocessing_layer_creator)
  tf_agent2 = agent_creators.create_agent(agent_name, time_step_spec,
  action_spec,
  preprocessing_layer_creator)
  tf_agent3 = agent_creators.create_agent(agent_name, time_step_spec,
  action_spec,
  preprocessing_layer_creator)
  # create the random network distillation object
  random_network_distillation = None
  if use_random_network_distillation:
    random_network_distillation = (
        random_net_distillation.RandomNetworkDistillation(
            time_step_spec=time_step_spec,
            preprocessing_layer_creator=preprocessing_layer_creator))

  root_dir1 = root_dir + 'def'
  root_dir2 = root_dir + 'mod'
  root_dir3 = root_dir + 'modshfl'

  llvm_trainer1 = trainer.Trainer(
      root_dir=root_dir1,
      agent=tf_agent1,
      random_network_distillation=random_network_distillation,
      warmstart_policy_dir=warmstart_policy_dir)

  llvm_trainer2 = trainer.Trainer(
      root_dir=root_dir2,
      agent=tf_agent2,
      random_network_distillation=random_network_distillation,
      warmstart_policy_dir=warmstart_policy_dir)

  llvm_trainer3 = trainer.Trainer(
      root_dir=root_dir3,
      agent=tf_agent3,
      random_network_distillation=random_network_distillation,
      warmstart_policy_dir=warmstart_policy_dir)

  policy_dict1 = {
      'saved_policy': tf_agent1.policy,
      'saved_collect_policy': tf_agent1.collect_policy,
  }
  policy_dict2 = {
      'saved_policy': tf_agent2.policy,
      'saved_collect_policy': tf_agent2.collect_policy,
  }
  policy_dict3 = {
      'saved_policy': tf_agent3.policy,
      'saved_collect_policy': tf_agent3.collect_policy,
  }
  saver1 = policy_saver.PolicySaver(policy_dict=policy_dict1)
  saver2 = policy_saver.PolicySaver(policy_dict=policy_dict2)
  saver3 = policy_saver.PolicySaver(policy_dict=policy_dict3)

  with open(
      os.path.join(FLAGS.data_path, 'module_paths'), 'r',
      encoding='utf-8') as f:
    module_paths = [
        os.path.join(FLAGS.data_path, name.rstrip('\n')) for name in f
    ]
    has_cmd = problem_configuration.has_cmd(module_paths)
    is_thin = problem_configuration.is_thinlto(module_paths)
    file_paths = [(path + '.bc', path + '.cmd' if has_cmd else None, path + '.thinlto.bc' if is_thin else None)
                  for path in module_paths]

  runner = problem_config.get_runner(
      moving_average_decay_rate=moving_average_decay_rate)

  dataset_fn = data_reader.create_sequence_example_dataset_fn(
      agent_name=agent_name,
      time_step_spec=time_step_spec,
      action_spec=action_spec,
      batch_size=batch_size,
      train_sequence_length=train_sequence_length,
      num_workers=FLAGS.num_workers)

  def sequence_example_iterator_fn(seq_ex: List[bytes]):
    return dataset_fn(seq_ex)

  reward_stat_map = collections.defaultdict(lambda: None)
  reward_stat_map_path1 = os.path.join(root_dir1, 'reward_stat_map')
  reward_stat_map_path2 = os.path.join(root_dir2, 'reward_stat_map')
  reward_stat_map_path3 = os.path.join(root_dir3, 'reward_stat_map')

  # Reload reward_stat_map if exists.
  # reward_stat_map of defaultdict(str, {str: RewardStat})
  if tf.io.gfile.exists(reward_stat_map_path1):
    with tf.io.gfile.GFile(reward_stat_map_path1, 'r') as f:
      data = json.load(f)
    for k, v in data.items():
      if v:
        reward_stat_map[k] = {
            sub_k: compilation_runner.RewardStat(**sub_v)
            for sub_k, sub_v in v.items()
        }
    logging.info('Loaded Reward Stat Map from disk, containing %d modules',
                 len(reward_stat_map))

  data_collector = local_data_collector.LocalDataCollector(
      file_paths=file_paths,
      num_workers=FLAGS.num_workers,
      num_modules=FLAGS.num_modules,
      runner=runner,
      parser=sequence_example_iterator_fn,
      reward_stat_map=reward_stat_map)

  # Repeat for num_policy_iterations iterations.
  t1 = time.time()
  while (llvm_trainer1.global_step_numpy() <
         num_policy_iterations * num_iterations):
    t2 = time.time()
    logging.info('Last iteration took: %f', t2 - t1)
    t1 = t2
    with tf.io.gfile.GFile(reward_stat_map_path1, 'w') as f:
      json.dump(reward_stat_map, f, cls=compilation_runner.DataClassJSONEncoder)
    with tf.io.gfile.GFile(reward_stat_map_path2, 'w') as f:
      json.dump(reward_stat_map, f, cls=compilation_runner.DataClassJSONEncoder)
    with tf.io.gfile.GFile(reward_stat_map_path3, 'w') as f:
      json.dump(reward_stat_map, f, cls=compilation_runner.DataClassJSONEncoder)

    policy_path1 = os.path.join(root_dir1, 'policy',
                               str(llvm_trainer1.global_step_numpy()))
    policy_path2 = os.path.join(root_dir2, 'policy',
                               str(llvm_trainer2.global_step_numpy()))
    policy_path3 = os.path.join(root_dir3, 'policy',
                               str(llvm_trainer3.global_step_numpy()))
    saver1.save(policy_path1)
    saver2.save(policy_path2)
    saver3.save(policy_path3)

    dataset_iter, monitor_dict = data_collector.collect_data(
        policy_path=os.path.join(policy_path2, deploy_policy_name))

    ds1, ds2, ds3 = dataset_iter
    tt2 = threading.Thread(target=train_proxy, args=(llvm_trainer2, ds2, monitor_dict, num_iterations))
    tt3 = threading.Thread(target=train_proxy, args=(llvm_trainer3, ds3, monitor_dict, num_iterations))
    tt2.start()
    tt3.start()
    llvm_trainer1.train(ds1, monitor_dict, num_iterations)
    tt2.join()
    tt3.join()
    data_collector.on_dataset_consumed(dataset_iter)

  # Save final policy.
  saver1.save(root_dir1)
  saver2.save(root_dir2)
  saver3.save(root_dir3)

  # Wait for all the workers to finish.
  data_collector.close_pool()

def train_proxy(trainer, ds, mon, n):
  trainer.train(ds, mon, n)

def main(_):
  gin.parse_config_files_and_bindings(
      FLAGS.gin_files, bindings=FLAGS.gin_bindings, skip_unknown=False)
  logging.info(gin.config_str())

  train_eval()


if __name__ == '__main__':
  flags.mark_flag_as_required('data_path')
  multiprocessing.handle_main(functools.partial(app.run, main))
