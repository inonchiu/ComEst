from distutils.core import setup

setup(
    name='comest',
    version='2.0',
    packages=[
              'example',
              'comest',
              'comest.templates',
              'comest.templates.filters'],
      package_data={
      'comest.templates':['sex.config', 'sex.params', 'default.conv', 'sex.extended.params'],
      'comest.templates.filters':['*']
      },
    url='',
    license='',
    author='I-Non Chiu',
    author_email='inchiu@asiaa.sinica.edu.tw',
    description='An estimator of completeness and purity of the source extractor'
)
