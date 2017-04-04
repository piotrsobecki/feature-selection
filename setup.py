from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(name='opt',
      version='0.8',
      description='Optimization',
      long_description=readme,
      url='https://github.com/piotrsobecki/opt',
      author='Piotr Sobecki',
      author_email='ptrsbck@gmail.com',
      license=license,
      packages=find_packages(exclude=('tests', 'docs')),
      install_requires=['deap'],
      zip_safe=False)