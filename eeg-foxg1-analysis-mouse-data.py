#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.io import loadmat

mat = loadmat('/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.mat', struct_as_record=True)


# In[2]:


mat


# In[3]:


mat['d']


# In[4]:


print(mat.keys())



# In[5]:


print(type(mat['d']))
print(mat['d'].shape)


# In[6]:


import numpy as np

if isinstance(mat['d'], np.ndarray) and mat['d'].dtype.names:
    print(mat['d'].dtype.names)


# In[7]:


event_markers_data = mat['d']['event_markers'][0, 0]
print(event_markers_data)


# In[8]:


# Accessing a specific field. Let's start with 'kof' as an example.
kof_data = mat['d']['conm2']

# Since 'kof_data' itself can be a structured array or a complex type, let's inspect its first element.
# Note: Adjust this depending on whether 'kof_data' contains multiple items or nested structures.
first_kof_element = kof_data[0, 0]  # Accessing the first item; structured arrays in NumPy are 2D.

# Now, you can inspect 'first_kof_element' further, such as checking its type or contents.
print(type(first_kof_element))
print(first_kof_element)

# If 'first_kof_element' contains more structured data or arrays, you can access these in a similar manner,
# either by field names (if it's a structured array) or by indices (if it's a regular ndarray).


# In[9]:


kof_data.shape


# In[10]:


mat['__globals__']


# In[12]:


get_ipython().run_line_magic('pylab', 'inline')
import bioread
pylab.rcParams['figure.figsize'] = (14.0, 12.0)  # Make figures a bit bigger


# In[13]:


# This file is included in bioread
data = bioread.read_file('/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.acq')


# In[14]:


data


# In[15]:


data.channels


# In[16]:


plt.subplot(211)

for chan in data.channels:
    plt.plot(chan.time_index, chan.data, label='{} ({})'.format(chan.name, chan.units))

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
None  # Don't print a silly legend thing


# In[30]:


import mne
import numpy as np
import h5py
from h5py import File

f = h5py.File('/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.hdf5', 'r')
for item in f.keys():
    print(f[item])


# In[33]:


import h5py
import numpy as np

file_path = '/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.hdf5'

# Open the HDF5 file
with h5py.File(file_path, 'r') as f:
    # Initialize a list to store channel data
    data_list = []
    channel_names = []  # List to keep track of channel names
    
    # Assuming '/channels' is a group containing datasets for each channel
    channels_group = f['/channels']
    
    # Iterating through each channel dataset in the '/channels' group
    for channel_name in channels_group:
        # Append the name of the channel to channel_names list
        channel_names.append(channel_name)
        
        # Access and append channel data to data_list
        channel_data = channels_group[channel_name][:]
        data_list.append(channel_data)
        
    # Convert the list of arrays into a single (channels x samples) array
    data = np.stack(data_list, axis=0)  # Ensure data is in shape (n_channels, n_samples)



# In[36]:


import h5py

file_path = '/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.hdf5'

with h5py.File(file_path, 'r') as f:
    event_markers = {}  # Dictionary to hold event marker data
    
    if "/event_markers" in f:
        event_markers_group = f['/event_markers']
        
        for marker_name in event_markers_group.keys():
            marker_group = event_markers_group[marker_name]
            # Assuming the metadata of interest is stored as attributes of each group
            attributes = {attr: marker_group.attrs[attr] for attr in marker_group.attrs}
            print(f"Attributes from {marker_name}: {attributes}")
            # Again, for simplicity, attributes are printed. Adjust handling as needed.


# In[38]:


import mne

# Example metadata - adjust these values based on your actual data
channel_names = [f'Channel{i+1}' for i in range(data.shape[0])]  # or use real names if available
sfreq = 500  # Sampling frequency in Hz
ch_types = 'eeg'  # Adjust based on your data: 'eeg', 'meg', 'ecg', etc.

info = mne.create_info(ch_names=channel_names, sfreq=sfreq, ch_types=ch_types)


# In[39]:


data


# In[40]:


data.shape


# In[41]:


info


# In[42]:


# Ensure data is in the correct shape: (n_channels, n_samples)
if data.shape[0] > data.shape[1]:
    data = data.T

raw = mne.io.RawArray(data, info)


# In[45]:


raw


# In[46]:


# Assuming 'sfreq' is your sampling frequency
sfreq = 500 

annotations = []

for marker_name, attrs in event_markers.items():
    onset = attrs['global_sample_index'] / sfreq  # Convert sample index to time in seconds
    duration = 0  # Assuming instant events. Adjust if your events have a duration
    description = attrs['label']
    
    annotations.append(mne.Annotations(onset=[onset], duration=[duration], description=[description]))

# Assuming 'raw' is your MNE Raw object
for ann in annotations:
    raw.set_annotations(ann)


# In[47]:


raw.plot(block=True, scalings='auto')


# In[48]:
import mne

picks = mne.pick_channels(raw.ch_names, include=['Channel7'])
raw_selected = raw.copy().pick(picks=picks)

# Now plot the new Raw object with only the selected channel
raw_selected.plot(block=True, scalings='auto')




# In[49]:

spectrum = raw.compute_psd()
spectrum.plot(average=True, picks="data", exclude="bads")


# In[50]:
    
    # First, let's determine how many data points correspond to 1000 seconds
n_data_points = 0.5 * raw.info['sfreq']

# Now, create a new Raw object containing only the first 1000 seconds
raw_short = raw.copy().crop(tmax=n_data_points / raw.info['sfreq'])

# Plot the shorter dataset
raw_short.plot(scalings='auto', block=True)


# In[51]:
print(raw.get_montage())
# In[52]:
xRaw = raw.get_data()

# In[53]:
xRaw.shape

# In[54]:
ica = mne.preprocessing.ICA(n_components=3, random_state=0)

# In[55]:
%matplotlib
ica.fit(raw.copy())
# In[56]:
ica.plot_components(outlines="skirt")

# In[57]:
import pandas as pd    
df = pd.read_csv('/Users/nikhil/Documents/eeg-mne-analysis/montage.csv')


import mne

# Convert the DataFrame to a dictionary with positions as tuples
montage_dict = {row['ch_name']: (row['X_ML'], row['Y_AP'], row['Z_DV']) for _, row in df.iterrows()}

# Create a montage
montage = mne.channels.make_dig_montage(ch_pos=montage_dict, coord_frame='head')


# Assuming your channel names match those in the montage
raw.set_montage(montage)

# Now you can try plotting the sensor locations again
# Plot the sensor montage
raw.plot_sensors(kind='3d', show_names=True)
# In[58]:
print(raw.ch_names)
events, event_ids = mne.events_from_annotations(raw)
print(mne.events_from_annotations(raw))
#events = mne.find_events(raw)
# In[59]:
# This file is included in bioread
data = bioread.read_file('/Users/nikhil/Documents/eeg-mne-analysis/foxg1-eeg/eeg-24h-2-data.acq')
# In[60]:
data.channels
