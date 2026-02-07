from setuptools import setup, find_packages

setup(
    name="aecp-langchain",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "aecp",
        "langchain-core",
        "numpy"
    ],
    author="AECP Team",
    description="LangChain integration for AECP",
)
