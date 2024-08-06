from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


if __name__=='__main__':
    setup(name='postmerger',
          version='0.0.2',
          description='surrogate fits for binary black-hole remnants',
          long_description=readme(),
          url='https://github.com/cpacilio/postmerger',
          author='Costantino Pacilio',
          author_email='costantinopacilio1990@gmail.com',
          license='MIT',
          packages=find_packages(),
          install_requires=[
              'numpy<2.0.0',
              'joblib>=1.3.2',
              'scipy>=1.10.1',
              'scikit-learn>=1.3.2',
              ],
          classifiers=[
              "Intended Audience :: Science/Research",
              "License :: OSI Approved :: MIT License",
              "Natural Language :: English",
              "Programming Language :: Python :: 3.8",
              "Topic :: Scientific/Engineering :: Physics",
              "Topic :: Scientific/Engineering :: Astronomy",
              ],
          include_package_data=True,
          zip_safe=False)
