from setuptools import setup , find_packages
from typing import List


def get_requirements(path:str)->List :

    with open(path) as file_obj :
        requirements = file_obj.readlines()
    
    print(requirements)
    requirements = [req.replace('\n','') for req in requirements if req != '-e .']
    return requirements 



setup(
    name = 'Airline Customer Classification Prediction',
    version= '0.0.1',
    author= 'Kasa Pavan',
    author_email = 'pavankasa86@gmail.com',
    packages = find_packages() ,
    requires = get_requirements('requirements.txt')
)