from typing import Sequence
from setuptools import setup


def get_requirements() -> Sequence[str]:
    with open("requirements.txt") as file:
        requirements = file.read().splitlines()

    return requirements


setup(
    name="edupra-model-server",
    maintainer="Niapoll",
    maintainer_email="Kolya21082001@gmail.com",
    version="1.0.0",
    description="Model server module for edupra project",
    install_requires=get_requirements(),
    python_requires="~= 3.6",
    options={"bdist_wheel": {"plat_name": "any"}},
    entry_points={
        "console_scripts": ["edupra-server=edupra_model_server.main:main"]
    },
)
