from distutils.core import setup
setup(name='RoomSimulator',  # project_name
      version='1.0',  # number of version
      author='Todd',  # publisher
      author_email='todd_stan@163.com',
      packages=['RoomSimulator', 'examples', 'validation', 'images'], 
      package_data={'RoomSimulator/SENSOR/Types': [
          'bidirectional.npy', 'binaural_L.npy', 'binaural_R.npy', 
          'cardoid.npy', 'dipole.npy', 'hemisphere.npy', 'hypercardoid.npy', 
          'null_sensor.npy', 'omnidirectional.npy', 'subcardoid.npy', 
          'supercardoid.npy', 'unidirectional.npy']})
