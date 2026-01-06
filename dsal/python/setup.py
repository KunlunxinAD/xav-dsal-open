import os
import platform
import shutil
import sys
import glob
import warnings
from os import path as osp
from setuptools import find_packages, setup
import pathlib
import json


project_root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(project_root_dir)

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension, CUDAExtension)
import utils.cpp_extension as xmlir_cpp_extension


def make_xpu_ext(name,
                 module,
                 source_dirs,
                 extra_include_path=[]):

    source_dirs=[os.path.join(*module.split("."), p) for p in source_dirs]
    print("source_dirs -> ", source_dirs, type(source_dirs))
    define_macros = []
    sources = []
    for dir in source_dirs:
        print(type(dir), dir)
        sources += glob.glob(os.path.join(dir, "*.xpu"))
        sources += glob.glob(os.path.join(dir, "*.cpp"))
    print("sources -> ", sources)

    base_dir = os.getcwd()
    extra_include_path=[os.path.join(base_dir, *module.split("."), p) \
        for p in extra_include_path]
    print("extra_include_path -> ", extra_include_path)

    return sources, extra_include_path

def get_extensions():
    extensions = []
    ext_name = 'xav_dsal._ext_xpu'
    define_macros = []

    all_extra_include_path = [
        '.',
        './csrc',
        os.path.join(project_root_dir, "c++/include"),
        '/opt/xre/include',
        '/opt/xdnn/include',
        '/opt/xccl/include',
    ]

    if os.environ.get("USE_XPYTORCH_LOWER_VERSION") == "TRUE":
        print("Detected USE_XPYTORCH_LOWER_VERSION=TRUE, skip adding xpu_external include path")
    else:
        all_extra_include_path.append(
            os.path.join(project_root_dir, "tmp/include/xpu_external/include")
        )

    source_dirs = [
        './csrc']

    all_sources = []

    for dir in source_dirs:
        all_sources += glob.glob(os.path.join(dir, "*.cpp"))

    print("all_sources -> ", all_sources)
    print("all_extra_include_path -> ", all_extra_include_path)

    # target_dir = os.path.abspath('mmcv')
    # os.makedirs(target_dir, exist_ok=True)


    library_directories = [
        os.path.join(project_root_dir, "c++/lib"),
    ]

    libraries = [
        "xav_dsal"
    ]

    extension = xmlir_cpp_extension.XPUExtension
    extra_compile_args = {"cxx": ["-Wno-sign-compare"],
                          "xpu": [],
                          "nvcc": [
                              "-DCUDA_HAS_FP16=1",
                              "-D__CUDA_NO_HALF_OPERATORS__",
                              "-D__CUDA_NO_HALF_CONVERSIONS__",
                              "-D__CUDA_NO_HALF2_OPERATORS__",
                          ],
                          }

    extra_link_args=["-Wl,-rpath,$ORIGIN/."]

    ext_ops = extension(
        name=ext_name,
        sources=all_sources,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        include_dirs=all_extra_include_path,
        library_dirs=library_directories,
        libraries=libraries,
        extra_link_args=extra_link_args,
    )
    extensions.append(ext_ops)

    return extensions


try:
    # https://setuptools.pypa.io/en/latest/deprecated/distutils-legacy.html
    from setuptools.command.build import build
except ImportError:
    from distutils.command.build import build

from setuptools.command.develop import develop
from setuptools.command.easy_install import easy_install
from setuptools.command.install_lib import install_lib

class BuildWithPTH(build):
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        pth_path = str(pathlib.Path(__file__).parent / 'xav_dsal.pth')
        pth_dest = str(pathlib.Path(self.build_lib) / pathlib.Path(pth_path).name)
        hook_path = str(pathlib.Path(__file__).parent / 'xav_dsal_import_hook.py')
        hook_dest = str(pathlib.Path(self.build_lib) / pathlib.Path(hook_path).name)
        lib_path = str(pathlib.Path(__file__).parent / '../c++/lib/libxav_dsal.so')
        lib_dest = str(pathlib.Path(self.build_lib) / 'xav_dsal/libxav_dsal.so')
        self.copy_file(pth_path, pth_dest)
        self.copy_file(hook_path, hook_dest)
        self.copy_file(lib_path, lib_dest)


class EasyInstallWithPTH(easy_install):
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        pth_path = str(pathlib.Path(__file__).parent / 'xav_dsal.pth')
        pth_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(pth_path).name)
        hook_path = str(pathlib.Path(__file__).parent / 'xav_dsal_import_hook.py')
        hook_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(hook_path).name)
        lib_path = str(pathlib.Path(__file__).parent / '../c++/lib/libxav_dsal.so')
        lib_dest = str(pathlib.Path(self.install_dir) / 'xav_dsal/libxav_dsal.so')
        self.copy_file(pth_path, pth_dest)
        self.copy_file(hook_path, hook_dest)
        self.copy_file(lib_path, lib_dest)


class InstallLibWithPTH(install_lib):
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        pth_path = str(pathlib.Path(__file__).parent / 'xav_dsal.pth')
        pth_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(pth_path).name)
        hook_path = str(pathlib.Path(__file__).parent / 'xav_dsal_import_hook.py')
        hook_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(hook_path).name)
        lib_path = str(pathlib.Path(__file__).parent / '../c++/lib/libxav_dsal.so')
        lib_dest = str(pathlib.Path(self.install_dir) / 'xav_dsal/libxav_dsal.so')
        self.copy_file(pth_path, pth_dest)
        self.copy_file(hook_path, hook_dest)
        self.copy_file(lib_path, lib_dest)
        self.outputs = [pth_dest, hook_dest, lib_dest]

    def get_outputs(self):
        return chain(super().get_outputs(), self.outputs)


class DevelopWithPTH(develop):
    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        pth_path = str(pathlib.Path(__file__).parent / 'xav_dsal.pth')
        pth_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(pth_path).name)
        hook_path = str(pathlib.Path(__file__).parent / 'xav_dsal_import_hook.py')
        hook_dest = str(pathlib.Path(self.install_dir) / pathlib.Path(hook_path).name)
        lib_path = str(pathlib.Path(__file__).parent / '../c++/lib/libxav_dsal.so')
        lib_dest = str(pathlib.Path(__file__).parent / 'xav_dsal/libxav_dsal.so')
        self.copy_file(pth_path, pth_dest)
        self.copy_file(hook_path, hook_dest)
        self.copy_file(lib_path, lib_dest)


def get_git_commit_hash():
    import subprocess
    command = "git rev-parse --short=10 HEAD"
    try:
        output = subprocess.check_output(command, shell=True)
        return 'COMMIT-'+output.strip().decode()
    except subprocess.CalledProcessError as e:
        return 'COMMIT-NIL'

def get_version():
    try:
        with open('version.json', 'r') as f:
            data = json.load(f)
            return data.get('version', 'unknown')
    except (FileNotFoundError, json.JSONDecodeError):
        return 'unknown'


if __name__ == '__main__':
    setup(
        name='xav_dsal',
        version=get_version(),
        description='XAV torch extensions, ' + get_git_commit_hash(),
        author='XAV team',
        keywords='',
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': xmlir_cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True),
                  'build': BuildWithPTH,
                  'easy_install': EasyInstallWithPTH,
                  'install_lib': InstallLibWithPTH,
                  'develop': DevelopWithPTH,
                  },
        zip_safe=False,
    )

