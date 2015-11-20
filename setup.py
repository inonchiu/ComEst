from distutils.core import setup

setup(
    name='comest',
    version='1.0',
    packages=['comest',
              'comest.templates',
              'comest.templates.filters'],
      package_data={
      'comest.templates':['sex.config', 'sex.params', 'default.conv', 'sex.extended.params'],
      'comest.templates.filters':['*']
      },
    url='',
    license='',
    author='I-Non Chiu',
    author_email='inonchiu@usm.lmu.de',
    description='Completeness Estimator'
)
