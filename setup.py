## useful links to create a package
## https://python-packaging.readthedocs.io/en/latest/minimal.html
## https://docs.python-guide.org/writing/structure
## https://docutils.sourceforge.io/docs/user/rst/quickref.html
## https://medium.com/@udiyosovzon/things-you-should-know-when-developing-python-package-5fefc1ea3606
## https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56

from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()


if __name__=='__main__':
    setup(name='postMerger',
          version='0.0.1',
          description='surrogate fits for binary black-hole remnants',
          long_description=readme(),
          url='https://github.com/cpacilio/postMerger',
          author='Costantino Pacilio',
          author_email='costantinopacilio1990@gmail.com',
          license='MIT',
          packages=find_packages(),
          install_requires=[
              'numpy>=1.24.4',
              'joblib>=1.3.2',
              'scipy>=1.10.1'
              'scikit-learn>=1.3.2',
              ],
          classifiers=[
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python :: 3.8",
              "Topic :: Scientific/Engineering :: Physics",
              "Topic :: Scientific/Engineering :: Astronomy",
              ]
          include_package_data=True,
          zip_safe=False)
