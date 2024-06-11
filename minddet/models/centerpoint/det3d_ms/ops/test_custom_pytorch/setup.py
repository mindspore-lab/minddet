# encoding: utf-8
import os
import sys

from setuptools import setup
from torch.utils import cpp_extension


def main():
    cpp_name = sys.argv[1]
    file_name, suffix = cpp_name.split(".")
    so_name = sys.argv[2]
    sys.argv[1:] = ["build_ext", "-i"]
    if suffix in ["cpp", "cc", "c"]:
        setup(
            name=file_name,  # 编译后的链接库名称
            ext_modules=[
                cpp_extension.CppExtension(
                    name=file_name,
                    sources=[cpp_name, "ms_ext.cpp"],  # 待编译文件
                    extra_compile_args=[],
                )
            ],
            cmdclass={"build_ext": cpp_extension.BuildExtension},  # 执行编译命令设置
        )
    if suffix in ["cu"]:
        setup(
            name=file_name,  # 编译后的链接库名称
            ext_modules=[
                cpp_extension.CUDAExtension(
                    name=file_name,
                    sources=[cpp_name, "ms_ext.cpp"],  # 待编译文件
                    extra_compile_args=[],
                )
            ],
            cmdclass={"build_ext": cpp_extension.BuildExtension},  # 执行编译命令设置
        )
    files = os.listdir(".")
    old_name = None
    for f in files:
        if f.startswith(file_name) and f.endswith(".so"):
            old_name = f
    if old_name:
        os.rename(old_name, so_name)


if __name__ == "__main__":
    main()
