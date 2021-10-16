"""
Description:
Cutting objects from images to use them for Classification
Assemble and save prepared dataset
"""

# Algorithm:
# --> Setting up full paths
# --> Defining list of classes' names
# --> Defining Numpy arrays to collect processed images
# --> Cutting objects from images
# --> Shuffling data along the first axis
# --> Splitting arrays into train, validation and test
# --> Saving arrays into HDF5 binary file
#
# Result: HDF5 binary file with custom dataset


# Importing libraries
import pandas as pd
import numpy as np
import h5py
import cv2
import os

from sklearn.utils import shuffle


"""
Setting up full paths
"""
full_path_to_codes = \
    'C:/Users/yashs/PycharmProjects/InternshipCNN/TCS'

full_path_to_images = \
    'C:/Users/yashs/PycharmProjects/InternshipCNN/toolkit/OID/CustomDataset'


"""
Defining list of classes' names
"""
labels = ['Horse', 'Tiger', 'Cat', 'Dog', 'Polar bear']


"""
Defining Numpy arrays to collect processed images
"""
# Preparing zero-valued Numpy array for cut objects
# Shape: image number, height, width, number of channels
x_train = np.zeros((1, 64, 64, 3))


# Preparing zero-valued Numpy array for classes' numbers
# Shape: class's number
y_train = np.zeros(1)


# Preparing temp zero-valued Numpy array for current cut object
# Shape: image number, height, width, number of channels
x_temp = np.zeros((1, 64, 64, 3))


# Preparing temp zero-valued Numpy array for class's number
# Shape: class's number
y_temp = np.zeros(1)


# Defining boolean variable to track arrays' shapes
first_object = True


"""
Cutting objects from images
"""
# Showing currently active directory
print('Currently active directory is:')
print(os.getcwd())
print()


# Activating needed directory with downloaded images
os.chdir(full_path_to_images)


# Showing currently active directory
print('Currently active directory after 1st changing is:')
print(os.getcwd())
print()


# Using method 'os.walk' to iterate all directories and all files
# It starts from currently active directory
for current_dir, dirs, files in os.walk('.'):
    # Iterating all files
    for f in files:
        # Checking if filename ends with '.jpg'
        if f.endswith('.jpg'):
            # Reading current image by OpenCV library
            # In this way image is opened already as Numpy array
            # (!) OpenCV by default reads images in BGR order of channels
            image_array = cv2.imread(f)

            # Swapping channels from BGR to RGB by OpenCV function
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Slicing only name from current file without extension
            image_name = f[:-4]

            # Preparing path to current annotation txt file
            path_to_annotation = 'Label' + '/' + image_name + '.txt'

            # Getting Pandas dataFrame from current annotation txt file
            a = pd.read_csv(path_to_annotation,
                            header=None,
                            delim_whitespace=True)

            # Getting number of rows and columns from current Pandas dataFrame
            a_rows = a.shape[0]
            a_columns = a.shape[1]

            # Defining variable for current class's name
            class_name = ''

            # Getting class's name from current Pandas dataFrame
            # Assembling complex name if it consists of few words
            for i in range(a_columns - 4):
                class_name += a.loc[0, i]

                # Adding space at the end
                class_name += ' '

            # Deleting space at the end
            class_name = class_name.rstrip(' ')

            # Getting index of class's name
            # according to its order in the list 'labels'
            class_index = labels.index(class_name)

            # Preparing index to start reading coordinates of objects from
            ii = a_columns - 4

            # Iterating all rows from current Pandas dataFrame
            for i in range(a_rows):
                # Getting coordinates of current object
                # Making numbers as integers
                x_min = int(a.loc[i, ii])
                y_min = int(a.loc[i, ii + 1])
                x_max = int(a.loc[i, ii + 2])
                y_max = int(a.loc[i, ii + 3])

                # Checking if current object's height & width are bigger than 64
                if (y_max - y_min) >= 64 and (x_max - x_min) >= 64:
                    # Cutting object from entire image
                    cut_object = image_array[y_min:y_max, x_min:x_max]

                    # Resizing cut object to 64 by 64 pixels size
                    cut_object = cv2.resize(cut_object,
                                            (64, 64),
                                            interpolation=cv2.INTER_CUBIC)

                    # Checking if it is the first object
                    if first_object:
                        # Assigning to the first position first object
                        x_train[0, :, :, :] = cut_object

                        # Assigning to the first position its class index
                        y_train[0] = class_index

                        # Changing boolean variable
                        first_object = False

                    # Collecting next objects into temp arrays
                    # Concatenating arrays vertically
                    else:
                        # Assigning to temp array current object
                        x_temp[0, :, :, :] = cut_object

                        # Assigning to temp array its class index
                        y_temp[0] = class_index

                        # Concatenating vertically temp arrays to main arrays
                        x_train = np.concatenate((x_train, x_temp), axis=0)
                        y_train = np.concatenate((y_train, y_temp), axis=0)


"""
Shuffling data along the first axis
"""
# Saving appropriate connection: image --> label
x_train, y_train = shuffle(x_train, y_train)


"""
Splitting arrays into train, validation and test
"""
# Showing total number of collected images
print(x_train.shape)
print(y_train.shape)
print()


# Slicing first 30% of elements from Numpy arrays for training
# Assigning sliced elements to temp Numpy arrays
x_temp = x_train[:int(x_train.shape[0] * 0.3), :, :, :]
y_temp = y_train[:int(y_train.shape[0] * 0.3)]


# Slicing last 70% of elements from Numpy arrays for training
# Re-assigning sliced elements to train Numpy arrays
x_train = x_train[int(x_train.shape[0] * 0.3):, :, :, :]
y_train = y_train[int(y_train.shape[0] * 0.3):]


# Slicing first 80% of elements from temp Numpy arrays
# Assigning sliced elements to validation Numpy arrays
x_validation = x_temp[:int(x_temp.shape[0] * 0.8), :, :, :]
y_validation = y_temp[:int(y_temp.shape[0] * 0.8)]


# Slicing last 20% of elements from temp Numpy arrays
# Assigning sliced elements to test Numpy arrays
x_test = x_temp[int(x_temp.shape[0] * 0.8):, :, :, :]
y_test = y_temp[int(y_temp.shape[0] * 0.8):]


"""
Saving arrays into HDF5 binary file
"""
# Activating needed directory with code files
os.chdir(full_path_to_codes)


# Showing currently active directory
print('Currently active directory after 2nd changing is:')
print(os.getcwd())


# Saving prepared Numpy arrays into HDF5 binary file
# Initiating File object
# Creating file with name 'custom_dataset.hdf5'
# Opening it in writing mode by 'w'
with h5py.File('custom_dataset.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')












