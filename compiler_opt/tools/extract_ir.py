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
"""Extract IR for training.

Extract IR for training, either from a compile_commands.json file produced by
cmake, or a linker parameter list file.

Only run with
'python compiler_opt/tools/extract_ir.py ...'

The compilation is assumed to have been performed with clang, using
-fembed-bitcode=all passed to cc1 (i.e. pass clang -Xclang=-fembed-bitcode=all)

In a ThinLTO case, the compilation is assumed to have been performed specifying
-mllvm -lto-embed-bitcode=post-merge-pre-opt.
"""
from __future__ import annotations

import json
import multiprocessing
import os
import re
import shutil
import subprocess

from dataclasses import dataclass
from typing import Dict, List, Optional

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string(
    'input', None,
    'Input file - either compile_commands.json or a linker parameter list')
flags.DEFINE_enum(
    'input_type', 'json', ['json', 'params'],
    'Input file type - json or params. The latter refers to lld params.')
flags.DEFINE_string('output_dir', None, 'Output directory')
flags.DEFINE_integer(
    'num_workers', None,
    'Number of parallel workers for objcopy. `None` for maximum available.')
flags.DEFINE_string('llvm_objcopy_path', 'llvm-objcopy', 'Path to llvm-objcopy')
flags.DEFINE_string('llvm_as_path', 'llvm-as', 'Path to llvm-as')
flags.DEFINE_string('llvm_dis_path', 'llvm-dis', 'Path to llvm-dis')
flags.DEFINE_string(
    'obj_base_dir', '',
    'Base directory for object files. Defaults to current working dir.')
flags.DEFINE_string(
    'cmd_filter', None,
    'Include only those modules with a command line matching this regexp. '
    'Setting it to None for not filtering. Note that the regexp is applied '
    'independently for each separate command line option. For example, ^-Oz$ '
    'will match Oz - built binaries. Does not work with lld_thinlto_build.')
flags.DEFINE_bool(
    'thinlto_build', False, 'Set if the build was ThinLTO, to '
    'ensure index files are also copied. The build is assumed to have had '
    '-mllvm -lto-embed-bitcode=post-merge-pre-opt passed to clang.')
flags.DEFINE_bool(
    'lld_thinlto_build', False, 'Set if the build was ThinLTO-ed via lld, to '
    'ensure index files are also copied. The build is assumed to have had '
    '-Wl,--save-temps=import -Wl,--thinlto-emit-index-files passed to clang.')

FLAGS = flags.FLAGS


@dataclass
class Flags:
  """Copy of extract_ir's flags"""
  input_file: str
  input_type: str
  output_dir: str
  num_workers: int
  llvm_objcopy_path: str
  llvm_as_path: str
  llvm_dis_path: str
  obj_base_dir: str
  cmd_filter: str
  thinlto_build: bool
  lld_thinlto_build: bool


@dataclass
class ModuleSpec:
  """Describes an input module."""
  obj_relative_path: str
  output_base_dir: str
  src_base_dir: str = ''

  def input_obj(self):
    return os.path.join(self.src_base_dir, self.obj_relative_path)

  def lld_src_bc(self):
    return os.path.join(self.src_base_dir,
                        self.obj_relative_path + '.3.import.bc')

  def lld_src_thinlto(self):
    return os.path.join(self.src_base_dir,
                        self.obj_relative_path + '.thinlto.bc')

  def module_name(self):
    return os.path.basename(self.obj_relative_path)

  def dest_dir(self):
    return os.path.join(self.output_base_dir,
                        os.path.dirname(self.obj_relative_path))

  def dest_cmd(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.cmd')

  def dest_bc(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.bc')

  def dest_thinlto(self):
    return os.path.join(self.dest_dir(), self.module_name() + '.thinlto.bc')


class CorpusDiscoverer:
  """Enumerates all modules to be retrieved."""

  def __init__(self, flg):
    self.f: Flags = flg

  def _load_from_lld_params(self, params_array: List[str]) -> List[ModuleSpec]:
    """Create an ObjectFile array based on lld's parameters."""
    # yank out -o and the output. After that, anything not starting with '-',
    # and ending in a '.o', is an object file.
    try:
      minus_o_idx = params_array.index('-o')
      del params_array[minus_o_idx:minus_o_idx + 2]
      just_obj_paths = [
          o for o in params_array if not o.startswith('-') and o.endswith('.o')
      ]
    except ValueError:
      logging.info('This params file does not have an explicit -o option.')
      just_obj_paths = params_array

    def make_spec(obj_file: str):
      return ModuleSpec(
          obj_relative_path=obj_file,
          output_base_dir=self.f.output_dir,
          src_base_dir=self.f.obj_base_dir)

    return [make_spec(obj_path) for obj_path in just_obj_paths]

  def _load_from_compile_commands(
      self, json_array: List[Dict[str, str]]) -> List[ModuleSpec]:

    def make_spec(command: Dict[str, str]):
      cmd = command['command']
      src_base_dir = command['directory']

      cmd_parts = cmd.split()
      try:
        obj_index = cmd_parts.index('-o') + 1
      except ValueError:
        logging.info('Failed to find \'-o\' in %s', cmd)
        return None

      obj_rel_path = cmd_parts[obj_index]
      # TODO(mtrofin): is the obj_base_dir correct for thinlto index bc files?
      return ModuleSpec(
          obj_relative_path=obj_rel_path,
          output_base_dir=self.f.output_dir,
          src_base_dir=src_base_dir)

    return [make_spec(cmd) for cmd in json_array]

  def _load_for_lld_thinlto(self) -> List[ModuleSpec]:
    cmd = self._get_find_command()
    try:
      with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        paths, err = p.communicate()
        if p.returncode != 0:
          subprocess.CalledProcessError(p.returncode, err)
    except subprocess.CalledProcessError as e:
      logging.error('Failed to load input paths: %s', e)
      raise e

    def make_spec(obj_file: bytes):
      return ModuleSpec(
          # Cut away ./ and .3.import.bc
          obj_relative_path=obj_file[2:-12].decode('utf-8'),
          output_base_dir=self.f.output_dir,
          src_base_dir=self.f.obj_base_dir)

    # Last line is a newline char
    return [make_spec(path) for path in paths.split(b'\n')[:-1]]

  def _get_find_command(self):
    """Call find to discover all input files output by the linker."""
    return [
        'find', '.' if self.f.obj_base_dir == '' else self.f.obj_base_dir,
        '-name', '*.3.import.bc'
    ]

  def load(self):
    if self.f.lld_thinlto_build and self.f.thinlto_build:
      logging.error(
          'lld_thinlto_build and thinlto_build are mutually exclusive.')
      raise ValueError

    if self.f.input_file is None:
      if self.f.lld_thinlto_build:
        specs = self._load_for_lld_thinlto()
      else:
        logging.error('Input flag not provided')
        raise ValueError
    elif self.f.input_type == 'json':
      with open(self.f.input_file, encoding='utf-8') as f:
        specs = self._load_from_compile_commands(json.load(f))
    elif self.f.input_type == 'params':
      if not self.f.obj_base_dir:
        logging.info(
            '-obj_base_dir is unspecified, assuming current directory.'
            'If no objects are found, use this option to specify the root'
            'directory for the object file paths in the input file.')
      with open(self.f.input_file, encoding='utf-8') as f:
        specs = self._load_from_lld_params(
            [line.strip() for line in f.readlines()])
    else:
      logging.error('Unknown input type: %s', self.f.input_type)
      raise ValueError

    metadata = {}
    if self.f.lld_thinlto_build:
      metadata['global_command_override'] = [
          f'\'global_command_override\' in {self.f.output_dir:s}/metadata.json '
          'needs to be filled out with the compile command used to invoke '
          'clang to invoke the linker.'
      ]

    return metadata, specs


class CorpusExtractor:
  """IR and command line extraction from an object file.

  The object file is assumed to have the .llvmbc and .llvmcmd sections.
  """

  # Holds absl.flags.FLAGS-like objects. Expected to have extract_ir's flags.
  # Main reason for using this (rather than the global FLAGS)
  # is for testing: `f` can be modified at will externally,
  # Without the hassle of having to redefine each flag as a class variable
  def __init__(self, flg):
    self.f: Flags = flg

  def _get_extraction_cmd_command(self, ms: ModuleSpec):
    """Call llvm_objcopy to extract the .llvmcmd section in ms.dest_cmd."""
    return [
        self.f.llvm_objcopy_path, '--dump-section=.llvmcmd=' + ms.dest_cmd(),
        ms.input_obj(), '/dev/null'
    ]

  def _get_extraction_bc_command(self, ms: ModuleSpec):
    """Call llvm_objcopy to extract the .llvmbc section in ms.dest_bc."""
    return [
        self.f.llvm_objcopy_path, '--dump-section=.llvmbc=' + ms.dest_bc(),
        ms.input_obj(), '/dev/null'
    ]

  def _get_dis_bc_command(self, ms: ModuleSpec):
    """Call llvm_dis to disassemble the module bitcode."""
    return [self.f.llvm_dis_path, ms.lld_src_bc(), '-o', '-']

  def _get_dis_thinlto_command(self, ms: ModuleSpec):
    """Call llvm_dis to disassemble the ThinLTO index bitcode."""
    return [self.f.llvm_dis_path, ms.lld_src_thinlto(), '-o', '-']

  def _get_as_bc_command(self, ms: ModuleSpec):
    """Call llvm_as to assemble the output bitcode file."""
    return [self.f.llvm_as_path, '-o=' + ms.dest_bc()]

  def _extract_clang(self, ms: ModuleSpec) -> Optional[str]:
    """Extract the .bc, .cmd, and .thinlto.bc files from a clang invocation."""
    if ms is None:
      return None
    if not os.path.exists(ms.input_obj()):
      logging.info('%s does not exist.', ms.input_obj())
      return None
    os.makedirs(ms.dest_dir(), exist_ok=True)
    try:
      subprocess.run(self._get_extraction_cmd_command(ms), check=True)
      if self.f.cmd_filter is not None or self.f.thinlto_build:
        with open(ms.dest_cmd(), encoding='utf-8') as f:
          lines = f.readlines()
        assert len(lines) == 1
        cmdline = lines[0]
        if not self._should_include_module(cmdline, self.f.cmd_filter):
          logging.info(
              'Excluding module %s because it does not match the filter',
              ms.input_obj())
          os.remove(ms.dest_cmd())
          return None
        if self.f.thinlto_build:
          index_file = self._get_thinlto_index(cmdline, ms.src_base_dir)
          shutil.copy(index_file, ms.dest_thinlto())

      subprocess.run(self._get_extraction_bc_command(ms), check=True)
    except subprocess.CalledProcessError as e:
      # This may happen if  .o file was build from asm (.S source).
      logging.warning('%s was not processed: %s', ms.input_obj(), e)
      return None
    assert os.path.exists(ms.dest_cmd()) and os.path.exists(ms.dest_bc())
    assert not self.f.thinlto_build or os.path.exists(ms.dest_thinlto())
    return ms.obj_relative_path

  def _extract_lld(self, ms: ModuleSpec) -> Optional[str]:
    """Extract the .bc file with ThinLTO index from an lld ThinLTO invocation.
    """
    if ms is None:
      return None
    if not os.path.exists(ms.lld_src_bc()):
      logging.info('%s does not exist.', ms.lld_src_bc())
      return None
    if not os.path.exists(ms.lld_src_thinlto()):
      logging.info('%s does not exist.', ms.lld_src_thinlto())
      return None
    os.makedirs(ms.dest_dir(), exist_ok=True)
    try:
      # Get the module in textual IR.
      shutil.copy(ms.lld_src_bc(), ms.dest_bc())
      shutil.copy(ms.lld_src_thinlto(), ms.dest_thinlto())
      if 1 is 0:
        cmd = self._get_dis_bc_command(ms)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
          module_ll, err = p.communicate()
          if p.returncode != 0:
            subprocess.CalledProcessError(p.returncode, err)

        # Get the module's index in textual IR.
        cmd = self._get_dis_thinlto_command(ms)
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
          module_index, err = p.communicate()
          if p.returncode != 0:
            subprocess.CalledProcessError(p.returncode, err)

        # Eat the first two lines of the index, if they exist.
        # These are the ModuleID and source_filename lines.
        # We just want the index.
        truncated_index = module_index.split(b'\n', 2)[2:]
        module_index = truncated_index[0] if len(truncated_index) > 0 else b''

        # Concatenate the module and its index.
        module_ll += module_index

        # Re-assemble the module to its destination file.
        subprocess.run(self._get_as_bc_command(ms), input=module_ll, check=True)
    except subprocess.CalledProcessError as e:
      # This may happen if  .o file was build from asm (.S source).
      logging.warning('%s was not processed: %s', ms.input_obj(), e)
      return None
    assert os.path.exists(ms.dest_bc())
    return ms.obj_relative_path

  def _extract(self, ms: ModuleSpec) -> str:
    if self.f.lld_thinlto_build:
      return self._extract_lld(ms)
    else:
      return self._extract_clang(ms)

  def extract(self, module_specs) -> List[str]:
    with multiprocessing.Pool(self.f.num_workers) as pool:
      return [s for s in pool.map(self._extract, module_specs) if s is not None]

  # TODO(ml-compiler-opt): maybe we can also convert here the cmdline file,from a
  # \0 - separated list of strings, to a \n one.
  @staticmethod
  def _should_include_module(cmdline: str, match_regexp: Optional[str]) -> bool:
    """Determine if the module should be included."""
    if match_regexp is None:
      return True
    lines = cmdline.split('\0')
    return any(len(re.findall(match_regexp, l)) for l in lines)

  @staticmethod
  def _get_thinlto_index(cmdline: str, basedir: str) -> Optional[str]:
    opts = cmdline.split('\0')
    for option in opts:
      if option.startswith('-fthinlto-index'):
        return os.path.join(basedir, option.split('=')[1])
    return None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flags_as_required(['output_dir'])
  flg = Flags(FLAGS.input, FLAGS.input_type, FLAGS.output_dir,
              FLAGS.num_workers, FLAGS.llvm_objcopy_path, FLAGS.llvm_as_path,
              FLAGS.llvm_dis_path, FLAGS.obj_base_dir, FLAGS.cmd_filter,
              FLAGS.thinlto_build, FLAGS.lld_thinlto_build)

  logging.info('Collecting input manifest.')
  discoverer = CorpusDiscoverer(flg)
  metadata, module_specs = discoverer.load()

  logging.info('Moving/processing files in manifest.')
  extractor = CorpusExtractor(flg)
  len_old = len(module_specs)
  metadata['modules'] = extractor.extract(module_specs)

  logging.info('Converted %d files out of %d', len(metadata['modules']),
               len_old)

  metadata_path = os.path.join(FLAGS.output_dir, 'metadata.json')
  logging.info('Saving %s', metadata_path)
  with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4)


if __name__ == '__main__':
  app.run(main)
