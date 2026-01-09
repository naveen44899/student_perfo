from setuptools import find_packages,setup
from typing import List

hypen_e_dot = "-e ."
def  get_requirements(file_path:str)->List[str]:
    requirements = []
    with open("requirements.txt")as file_obj:
        requirements = file_obj.readlines()
        [req.replace("\n","") for req in requirements] # in requirements.txt new line(\n) also printed so we are removing it
    
        if hypen_e_dot in requirements:
            requirements.remove(hypen_e_dot)
    return requirements

setup(
    name = "student_performance",
    version = "0.0.1",
    author = "naveen",
    author_email = "naveen8296088066@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements("requirements.txt")
)