from setuptools import setup

setup(name='sumo_gym_env',
      version='1.0.0',
      install_requires=[
          'gymnasium~=1.0.0', 
          'traci~=1.25.0', 
          'sumolib~=1.25.0', 
          'libsumo~=1.25.0', 
          'eclipse-sumo~=1.25.0',
          'scipy',]
)