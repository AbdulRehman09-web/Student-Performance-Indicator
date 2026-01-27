from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will retrive the requirements.txt file and return the list of requirements
    1. open requirements.txt file
    2. read all the lines
    3. return a list of requirements
    4. ignore -e .
    '''
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n',"") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name = 'Student-Performance-Indicator',
    version = '0.0.1',
    author = 'Abdul Rehman',
    author_email = 'abdulrehmannadeem825@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)