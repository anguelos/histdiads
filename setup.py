from setuptools import setup, Extension

setup(
    name='histdiads',
    version='0.1.0',
    packages=['histdiads'],
    license='MIT',
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/histdiads',
    description="Pytorch packaged historical document image analysis datasets.",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    keywords=["pytorch", "datasets", "DIA", "computer vision"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"],
    install_requires=["torch", "tqdm", "torchvision", "Pillow", "requests"],
)