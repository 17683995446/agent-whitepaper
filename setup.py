from setuptools import setup, find_packages

setup(
    name="agent-whitepaper",
    version="0.1.0",
    description="自驱永续进化预测修正Agent系统",
    author="Agent Whitepaper Team",
    author_email="2358384171@qq.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pyyaml>=6.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "full": [
            "msgpack>=1.0.0",
            "sympy>=1.10.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
