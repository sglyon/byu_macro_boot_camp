import h5py as hh
import numpy as np

x = np.arange(0,20,.01)
y = np.cos(x)

newFile = hh.File('myNewFile33.hdf5')
newGroup = newFile.create_group('This is my new group')
newSubgroup = newGroup.create_group('this is my new subgroup')

dataSet = newSubgroup.create_dataset(name = 'x, cos', data = np.array([x,y]))
