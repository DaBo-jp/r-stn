from setuptools import setup, Extension
import pybind11
import os

# C++ソースファイルがあるディレクトリ
LIB_DIR = "rstn/cpp_src"

# ソースファイルのリスト (パス付きで指定)
sources = [
    os.path.join(LIB_DIR, "bindings.cpp"),
    os.path.join(LIB_DIR, "RSTNBox.cpp"),
    os.path.join(LIB_DIR, "RSTNNode.cpp"),
]

# コンパイルオプション
extra_compile_args = ['-O3', '-fopenmp', '-std=c++17']
extra_link_args = ['-fopenmp']

# 拡張モジュールの定義
ext_modules = [
    Extension(
        "rstn_cpp",               # モジュール名 (import rstn_cpp)
        sources,
        include_dirs=[
            pybind11.get_include(),
            LIB_DIR               # <--- 重要: .hpp ファイルを探す場所を指定
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++'
    ),
]

setup(
    name="rstn_cpp",
    version="1.1.0",
    ext_modules=ext_modules,
    setup_requires=['pybind11'],
)