import copy
import os
import os.path as osp
import shlex
import subprocess
import sys
import warnings
from typing import List, Optional

import importlib

import setuptools
import torch
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import (  # NOTE: use original torch CppExtension
    COMMON_NVCC_FLAGS,
    IS_HIP_EXTENSION,
    IS_WINDOWS,
    PLAT_TO_VCVARS,
    SUBPROCESS_DECODE_ARGS,
    CppExtension,
    _get_num_workers,
    _is_cuda_file,
)
from torch.utils.cpp_extension import include_paths as torch_include_paths
from torch.utils.cpp_extension import (
    is_ninja_available,
)  # NOTE: use original torch CppExtension
from torch.utils.cpp_extension import library_paths as torch_library_paths
from torch.utils.cpp_extension import (
    verify_ninja_availability,
)  # NOTE: use original torch CppExtension

__all__ = [
    "xpu_include_paths",
    "xpu_library_paths",
    "CppExtension",
    "XPUExtension",
    "BuildExtension",
]

# _XPYTORCH_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
_XPYTORCH_PATH = importlib.util.find_spec( 'torch_xmlir').submodule_search_locations[0]
# _XPYTORCH_PATH = "/workspace/XMLIR/build/tools/torch_xmlir/xpu/src/extern_xpu"

def _maybe_write(filename, new_content):
    r"""
    Equivalent to writing the content into the file but will not touch the file
    if it already had the right content (to avoid triggering recompile).
    """
    if osp.exists(filename):
        with open(filename) as f:
            content = f.read()

        if content == new_content:
            # The file already contains the right thing!
            return

    with open(filename, "w") as source_file:
        source_file.write(new_content)


def _run_ninja_build(build_directory: str, verbose: bool, error_prefix: str) -> None:
    command = ["ninja", "-v"]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command.extend(["-j", str(num_workers)])
    env = os.environ.copy()
    # Try to activate the vc env for the users
    if IS_WINDOWS and "VSCMD_ARG_TGT_ARCH" not in env:
        from setuptools import distutils  # type: ignore[import]

        plat_name = distutils.util.get_platform()
        plat_spec = PLAT_TO_VCVARS[plat_name]

        vc_env = distutils._msvccompiler._get_vc_env(plat_spec)
        vc_env = {k.upper(): v for k, v in vc_env.items()}
        for k, v in env.items():
            uk = k.upper()
            if uk not in vc_env:
                vc_env[uk] = v
        env = vc_env
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Warning: don't pass stdout=None to subprocess.run to get output.
        # subprocess.run assumes that sys.__stdout__ has not been modified and
        # attempts to write to it by default.  However, when we call _run_ninja_build
        # from ahead-of-time cpp extensions, the following happens:
        # 1) If the stdout encoding is not utf-8, setuptools detachs __stdout__.
        #    https://github.com/pypa/setuptools/blob/7e97def47723303fafabe48b22168bbc11bb4821/setuptools/dist.py#L1110
        #    (it probably shouldn't do this)
        # 2) subprocess.run (on POSIX, with no stdout override) relies on
        #    __stdout__ not being detached:
        #    https://github.com/python/cpython/blob/c352e6c7446c894b13643f538db312092b351789/Lib/subprocess.py#L1214
        # To work around this, we pass in the fileno directly and hope that
        # it is valid.
        stdout_fileno = 1
        subprocess.run(
            command,
            stdout=stdout_fileno if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=build_directory,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        # Python 2 and 3 compatible way of getting the error object.
        _, error, _ = sys.exc_info()
        # error.output contains the stdout and stderr of the build attempt.
        message = error_prefix
        # `error` is a CalledProcessError (which has an `output`) attribute, but
        # mypy thinks it's Optional[BaseException] and doesn't narrow
        if hasattr(error, "output") and error.output:  # type: ignore[union-attr]
            message += f": {error.output.decode(*SUBPROCESS_DECODE_ARGS)}"  # type: ignore[union-attr]
        raise RuntimeError(message) from e


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    xpu_cflags,
    xpu_post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_xpu,
    with_cuda,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `xpu_cflags`: list of flags to pass to $xtdk. Can be None.
    `xpu_postflags`: list of flags to append to the $xtdk invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_xpu`: If we should be compiling with XPU.
    `with_cuda`: If we should be compiling XPU kernels with xtrans.
    """
    ident = " " * 2

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    xpu_cflags = sanitize_flags(xpu_cflags)
    xpu_post_cflags = sanitize_flags(xpu_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    if "XPYTORCH_XTDK" in os.environ:
        xtdk = os.getenv(
            "XPYTORCH_XTDK"
        )  # user can set xtdk compiler with ccache using the environment variable here
        print(f"Using local xtdk compiler: {xtdk}")
    else:
        xtdk = osp.join(_XPYTORCH_PATH, "codegen_resource", "xtdk", "bin", "clang++")
        print(f"Using xmlir xtdk compiler: {xtdk}")

    if "XTRANS_PATH" in os.environ:
        nvcc = osp.join(os.getenv("XTRANS_PATH"), "bin", "nvcc")
        print(f"Using local xtrans compiler: {nvcc}")
    elif with_cuda:
        # TODO: package xtrans in xpytorch
        raise NotImplementedError(
            "Only support local installed xtrans. Not found xtrans path in environment variable XTRANS_PATH"
        )
    else:
        nvcc = None

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {xtdk}")
    if with_xpu:
        config.append(f"xtdk = {xtdk}")
    if with_cuda:
        config.append(f"nvcc = {nvcc}")

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_xpu:
        flags.append(f'xpu_cflags = {" ".join(xpu_cflags)}')
        flags.append(f'xpu_post_cflags = {" ".join(xpu_post_cflags)}')
    if with_cuda:
        flags.append(f'cuda_cflags = {" ".join(cuda_cflags)}')
        flags.append(f'cuda_post_cflags = {" ".join(cuda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [osp.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        # TODO: support xpu compile on windows
        compile_rule.append(
            f"{ident}command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags"
        )
        compile_rule.append(f"{ident}deps = msvc")
    else:
        compile_rule.append(
            f"{ident}command = $cxx -x xpu --xpu-host-only -MMD -MF $out.d $cflags -c $in -o $out $post_cflags"
        )
        compile_rule.append(f"{ident}depfile = $out.d")
        compile_rule.append(f"{ident}deps = gcc")

    if with_xpu:
        xpu_compile_rule = ["rule xpu_compile"]
        xpu_compile_rule.append(
            f"{ident}command = "
            "wdir=$$(dirname $$(echo $out | cut -d' ' -f1))"
            "&& xpufile=$$(basename $in)"
            "&& $xtdk --basename $$wdir/$${xpufile%.*} $xpu_cflags -c $in $xpu_post_cflags"
        )

    if with_cuda:
        cuda_compile_rule = ["rule cuda_compile"]
        cuda_compile_rule.append(
            f"  command = $nvcc $cuda_cflags -c $in -o $out $cuda_post_cflags"
        )

    result_objects = []
    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_xpu_source = _is_xpu_file(source_file) and with_xpu
        is_cuda_source = _is_cuda_file(source_file) and with_cuda
        if is_xpu_source:
            rule = "xpu_compile"
        elif is_cuda_source:
            rule = "cuda_compile"
        else:
            rule = "compile"

        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        if is_xpu_source:
            # NOTE: xpu source file will generate two object files, one is host object file,
            # the other is device object file named with .device.bin.o and .o suffix respectively.
            object_file = f"{object_file.replace('.o', '.device.bin.o')} {object_file}"
            result_objects.extend(object_file.split(" "))
        else:
            result_objects.append(object_file)
        build.append(f"build {object_file}: {rule} {source_file}")

    # write result objects to `objects`
    objects.clear()
    objects.extend(result_objects)

    devlink_rule, devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            cl_paths = (
                subprocess.check_output(["where", "cl"])
                .decode(*SUBPROCESS_DECODE_ARGS)
                .split("\r\n")
            )
            if len(cl_paths) >= 1:
                cl_path = osp.dirname(cl_paths[0]).replace(":", "$:")
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(
                f'{ident}command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out'
            )
        else:
            link_rule.append(f"{ident}command = $cxx $in $ldflags -o $out")

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_xpu:
        blocks.append(xpu_compile_rule)  # type: ignore
    if with_cuda:
        blocks.append(cuda_compile_rule)  # type: ignore
    blocks += [devlink_rule, link_rule, build, devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


def _write_ninja_file_and_compile_objects(
    sources: List[str],
    objects,
    cflags,
    post_cflags,
    xpu_cflags,
    xpu_post_cflags,
    cuda_cflags,
    cuda_post_cflags,
    build_directory: str,
    verbose: bool,
    with_xpu: Optional[bool],
    with_cuda: Optional[bool],
) -> None:
    verify_ninja_availability()

    if with_xpu is None:
        with_xpu = any(map(_is_xpu_file, sources))
    build_file_path = osp.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)
    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        xpu_cflags=xpu_cflags,
        xpu_post_cflags=xpu_post_cflags,
        cuda_cflags=cuda_cflags,
        cuda_post_cflags=cuda_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_xpu=with_xpu,
        with_cuda=with_cuda,
    )
    if verbose:
        print("Compiling objects...", file=sys.stderr)
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def _is_xpu_file(path: str) -> bool:
    valid_ext = [".xpu"]
    return osp.splitext(path)[1] in valid_ext


def xpu_include_paths():
    """
    Get the include paths for XPU C++ API.

    Returns:
        List of include paths.
    """
    paths = []

    # XPytorch export headers
    paths.append(osp.join(_XPYTORCH_PATH, "include"))

    # XPU C++ API headers
    paths.append(osp.join(_XPYTORCH_PATH, "xpu_external", "include"))

    return paths


def xpu_library_paths():
    """
    Get the library paths for XPU C++ API.

    Returns:
        List of library paths.
    """
    paths = []

    paths.append(_XPYTORCH_PATH)

    # Default path of XAV docker
    paths.append("/opt/xre/so")

    return paths


class BuildExtension(build_ext):
    r"""
    A custom :mod:`setuptools` build extension.

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/XPU compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``xpu`` or ``cuda``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and XPU compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.
    It only supports Ninja backend now.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        r"""
        Returns a subclass with alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        """

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        assert self.use_ninja, "Only ninja backend is supported."

        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "{}. Falling back to using the slow distutils backend."
            )
            if not is_ninja_available():
                warnings.warn(msg.format("we could not find ninja."))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'xtdk' and 'nvcc'
            # when extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'xpu' and 'nvcc'
            # is passed to extra_compile_args in XPUExtension, i.e.
            #   XPUExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   XPUExtension(..., extra_compile_args={'xpu': [...]})
            # or
            #   XPUExtension(..., extra_compile_args={'nvcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "xtdk", "nvcc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}")  # type: ignore[import]
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        # Register .xpu as valid source extensions.
        self.compiler.src_extensions += [".xpu", ".cu", ".cuh"]

        def append_std17_if_no_std_present(cflags) -> None:
            # Just pass once -std flag.
            cpp_format_prefix = (
                "/{}:" if self.compiler.compiler_type == "msvc" else "-{}="
            )
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_cuda_flags(cflags):
            cflags = COMMON_NVCC_FLAGS + ["--compiler-options", "-fPIC"] + cflags

            # NVCC does not allow multiple -ccbin/--compiler-bindir to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            _ccbin = os.getenv("CC")
            if _ccbin is not None and not any(
                flag.startswith(("-ccbin", "--compiler-bindir")) for flag in cflags
            ):
                cflags.extend(["-ccbin", _ccbin])

            return cflags

        def unix_xpu_flags(cflags):
            # TODO: default to xpu2 arch and support more arch later
            cflags = ["-fPIC", "-O2", "-fno-builtin", "--xpu-arch=xpu3", "-D", "__XPU3__",
                      "-Wno-int-to-void-pointer-cast", "-Wno-int-to-pointer-cast",
                      "-mllvm", "--xpu-inline-cost", "-mllvm", "--xpu-inline-hot-call"]

            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not osp.isabs(paths[i]):
                        paths[i] = osp.abspath(paths[i])

        def unix_wrap_ninja_compile(
            sources,
            output_dir,
            macros=None,
            include_dirs=None,
            debug=False,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""
            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = osp.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)

            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]

            # extra_postargs can be either:
            # - a dict mapping cxx/xpu to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)
            append_std17_if_no_std_present(post_cflags)

            # xtdk compile flags
            with_xpu = any(map(_is_xpu_file, sources))
            xpu_post_cflags = None
            xpu_cflags = None
            if with_xpu:
                xpu_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    xpu_post_cflags = extra_postargs["xpu"]
                else:
                    xpu_post_cflags = list(extra_postargs)
                xpu_post_cflags = unix_xpu_flags(xpu_post_cflags)
                append_std17_if_no_std_present(xpu_post_cflags)
                xpu_cflags = [shlex.quote(f) for f in xpu_cflags]
                xpu_post_cflags = [shlex.quote(f) for f in xpu_post_cflags]

            # CUDA compile flags
            with_cuda = any(map(_is_cuda_file, sources))
            cuda_post_cflags = None
            cuda_cflags = None
            if with_cuda:
                cuda_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    cuda_post_cflags = extra_postargs["nvcc"]
                else:
                    cuda_post_cflags = list(extra_postargs)
                cuda_post_cflags = unix_cuda_flags(cuda_post_cflags)
                append_std17_if_no_std_present(cuda_post_cflags)
                cuda_cflags = [shlex.quote(f) for f in cuda_cflags]
                cuda_post_cflags = [shlex.quote(f) for f in cuda_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                xpu_cflags=xpu_cflags,
                xpu_post_cflags=xpu_post_cflags,
                cuda_cflags=cuda_cflags,
                cuda_post_cflags=cuda_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_xpu=with_xpu,
                with_cuda=with_cuda,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511
        assert (
            self.compiler.compiler_type != "msvc" and self.use_ninja
        ), "XPUExtension only support compile in unix with ninja now."
        self.compiler.compile = unix_wrap_ninja_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = ".".join(without_abi)
        return ext_filename

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = f"-DTORCH_EXTENSION_NAME={name}"
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(extension, "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))  # type: ignore[import]


def XPUExtension(name, sources, *args, **kwargs):
    r"""
    Creates a :class:`setuptools.Extension` for XPU/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a XPU/C++
    extension. This includes the XPU C++ api include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from torch_xmlir.utils.cpp_extension import BuildExtension, XPUExtension
        >>> setup(
        ...     name='xpu_extension',
        ...     ext_modules=[
        ...         XPUExtension(
        ...                 name='xpu_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.xpu, extension_kernels.cu'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'xtdk': ['-O2'],
        ...                                     'nvcc': ['-O2']})
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """
    include_dirs = kwargs.get("include_dirs", [])
    include_dirs += torch_include_paths(cuda=False)
    include_dirs += xpu_include_paths()
    if os.getenv("XTRANS_PATH") is not None:
        include_dirs.append(os.path.join(str(os.getenv("XTRANS_PATH")), "include"))
    kwargs["include_dirs"] = include_dirs

    library_dirs = kwargs.get("library_dirs", [])
    library_dirs += torch_library_paths(cuda=False)
    library_dirs += xpu_library_paths()

    libraries = kwargs.get("libraries", [])
    # torch libs
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    libraries.append("torch_python")
    # xpu libs
    libraries.append("XMLIRRuntime")
    libraries.append("xdnn_pytorch")
    libraries.append("xpuapi")
    libraries.append("xlog_adapter")
    libraries.append("xpurt")
    # cuda libs
    library_dirs.append("/usr/local/cuda/lib64")
    libraries.append("cudart")

    kwargs["library_dirs"] = library_dirs
    kwargs["libraries"] = libraries

    kwargs["language"] = "c++"
    return setuptools.Extension(name, sources, *args, **kwargs)

