from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()

    # Remove newline characters
    requirements = [req.strip() for req in requirements]

    # Remove "-e ." if present
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name="student_performance",
    version="0.0.1",
    author="Naveen",
    author_email="naveen8296088066@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
