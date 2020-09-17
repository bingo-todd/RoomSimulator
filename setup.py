from distutils.core import setup
setup(name='RoomSimulator',  # project_name
      version='1.0',  # number of version
      author='Todd',  # publisher
      author_email='todd_stan@163.com',
      packages=['RoomSimulator', 'examples', 'validation', 'images'], 
      package_data={'RoomSimulator': ['SENSOR/Types/*']})
