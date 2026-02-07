from setuptools import setup, find_packages

setup(
    name="aecp-demo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "aecp",
        "sentence-transformers>=2.2.0",
        "rich>=10.0.0",
        "typer>=0.9.0"
    ],
    entry_points={
        "console_scripts": [
            "aecp-demo=aecp_demo.main:main",
        ],
    },
    author="AECP Team",
    description="Zero-friction CLI demo for AECP (Python)",
    keywords="agent, embedding, communication, protocol, demo",
)
