from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("warpu.coupledMH", ["warpu/coupledMH.pyx"]),
    Extension("warpu.utils", ["warpu/utils.pyx"]),
    Extension("warpu.coupledGradMCMC", ["warpu/coupledGradMCMC.pyx"]),
    Extension("warpu.WarpUSampling", ["warpu/WarpUSampling.pyx"]),
    Extension("warpu.gaussmix", ["warpu/gaussmix.pyx"]),
    Extension("warpu.gaussmixdiag", ["warpu/gaussmixdiag.pyx"]),
    Extension("warpu.gaussmixfull", ["warpu/gaussmixfull.pyx"]),
    Extension("warpu.skewt", ["warpu/skewt.pyx"]),
    Extension("warpu.GaussGammaMix", ["warpu/gaussgammamix.pyx"]),
    Extension("warpu.GaussSkewMix", ["warpu/gaussskewmix.pyx"]),
    Extension("warpu.tempering", ["warpu/tempering.pyx"]),
    Extension("warpu.batchchain", ["warpu/batchchain.pyx"]),
]

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
    name="warpu",
    version="0.1.0",
    description="Warp-U sampling and stochastic Warp-U bridge sampling for multimodal distributions",
    long_description=open("../README.md", encoding="utf-8").read() if __import__("os").path.exists("../README.md") else "",
    long_description_content_type="text/markdown",
    author="Fei Ding, Shiyuan He, David E. Jones, Xiao-Li Meng",
    license="MIT",
    python_requires=">=3.9",
    packages=["warpu"],
    package_data={"warpu": ["*.pxd", "*.pyx"]},
    ext_modules=extensions,
    include_dirs=[np.get_include()],
    install_requires=["numpy", "scipy"],
    zip_safe=False,
)
