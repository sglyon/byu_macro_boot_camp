import h5py as hh
import numpy as np

x = np.arange(0,20,.1)
y = np.cox(x)

newFile = hh.File('/Users/Dropbox/MacromyNewFile.hdf5', 'w')
newGroup = newFile.create_group("This is a new group")
newSub = newGroup.create_group("This is a new subgroup")

newDataset = newSub.create_dataset(name = 'theData', data = (x,y))
