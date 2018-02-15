from setuptools import setup
from setuptools import find_packages

setup(name='mlopt',
      version='1.0.1',
      description='Hyperparameter optimization for machine learning algorithms.',
      author='Arnaud Van Looveren',
      author_email='arnaudvlooveren@gmail.com',
      url='https://github.com/arnaudvl/ml-parameter-optimization',
      license='MIT',
      install_requires=['lightgbm','numpy','pandas','scikit-learn','scipy','xgboost'],
packages=find_packages())
