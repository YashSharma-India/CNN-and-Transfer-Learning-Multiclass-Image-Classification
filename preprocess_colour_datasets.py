"""
Description:
Constructing set of datasets with colour images by producing datasets from
prepared ones by applying preprocessing techniques.
Save set of processed datasets in colour
"""

# Algorithm:
# --> Setting up full paths
# --> Preprocessing Custom Dataset
#
# Result: 5 new HDF5 binary files for processed dataset

# Importing needed libraries
import numpy as np
import h5py


"""
Setting up full path
"""
full_path_to_codes = \
    'C:/Users/yashs/PycharmProjects/InternshipCNN/TCS'


"""
Preprocessing Dataset
"""
# Opening saved Dataset from HDF5 binary file
# Initiating File object
# Opening file in reading mode by 'r'
with h5py.File(full_path_to_codes + '/' + 'custom_dataset.hdf5', 'r') as f:
    # Extracting saved arrays for training by appropriate keys
    # Saving them into new variables
    x_train = f['x_train']  # HDF5 dataset
    y_train = f['y_train']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_train = np.array(x_train)  # Numpy arrays
    y_train = np.array(y_train)  # Numpy arrays

    # Extracting saved arrays for validation by appropriate keys
    # Saving them into new variables
    x_validation = f['x_validation']  # HDF5 dataset
    y_validation = f['y_validation']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_validation = np.array(x_validation)  # Numpy arrays
    y_validation = np.array(y_validation)  # Numpy arrays

    # Extracting saved arrays for testing by appropriate keys
    # Saving them into new variables
    x_test = f['x_test']  # HDF5 dataset
    y_test = f['y_test']  # HDF5 dataset
    # Converting them into Numpy arrays
    x_test = np.array(x_test)  # Numpy arrays
    y_test = np.array(y_test)  # Numpy arrays


# Showing shapes of Numpy arrays with RGB images
print('Numpy arrays of Custom Dataset')
print(x_train.shape)
print(x_validation.shape)
print(x_test.shape)
print()


# Implementing normalization by dividing images pixels on 255.0
# Purpose: to make computation more efficient by reducing values between 0 and 1
x_train_255 = x_train / 255.0
x_validation_255 = x_validation / 255.0
x_test_255 = x_test / 255.0


# Saving processed Numpy arrays into new HDF5 binary file
# Creating file with name 'dataset_custom_rgb_255.hdf5'
with h5py.File('custom' + '/' + 'dataset_custom_rgb_255.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Calculating Mean Image from training dataset and apply it to all sub-datasets
mean_rgb_dataset_custom = np.mean(x_train_255, axis=0)  # (64, 64, 3)

# Implementing normalization by subtracting Mean Image
x_train_255_mean = x_train_255 - mean_rgb_dataset_custom
x_validation_255_mean = x_validation_255 - mean_rgb_dataset_custom
x_test_255_mean = x_test_255 - mean_rgb_dataset_custom


# Creating file with name 'mean_rgb_dataset_custom.hdf5'
with h5py.File('custom' + '/' + 'mean_rgb_dataset_custom.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy array for Mean Image
    f.create_dataset('mean', data=mean_rgb_dataset_custom, dtype='f')


# Saving processed Numpy arrays into new HDF5 binary file
# Creating file with name 'dataset_custom_rgb_255_mean.hdf5'
with h5py.File('custom' + '/' + 'dataset_custom_rgb_255_mean.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255_mean, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255_mean, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255_mean, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Calculating Standard Deviation from training dataset and apply it to all sub-datasets
std_rgb_dataset_custom = np.std(x_train_255_mean, axis=0)  # (64, 64, 3)

# Implementing preprocessing by dividing on Standard Deviation
x_train_255_mean_std = x_train_255_mean / std_rgb_dataset_custom
x_validation_255_mean_std = x_validation_255_mean / std_rgb_dataset_custom
x_test_255_mean_std = x_test_255_mean / std_rgb_dataset_custom

# Creating file with name 'std_rgb_dataset_custom.hdf5'
with h5py.File('custom' + '/' + 'std_rgb_dataset_custom.hdf5', 'w') as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy array for Mean Image
    f.create_dataset('std', data=std_rgb_dataset_custom, dtype='f')


# Saving processed Numpy arrays into new HDF5 binary file
# Creating file with name 'dataset_custom_rgb_255_mean_std.hdf5'
with h5py.File('custom' + '/' + 'dataset_custom_rgb_255_mean_std.hdf5', 'w') \
        as f:
    # Calling methods to create datasets of given shapes and types
    # Saving Numpy arrays for training
    f.create_dataset('x_train', data=x_train_255_mean_std, dtype='f')
    f.create_dataset('y_train', data=y_train, dtype='i')

    # Saving Numpy arrays for validation
    f.create_dataset('x_validation', data=x_validation_255_mean_std, dtype='f')
    f.create_dataset('y_validation', data=y_validation, dtype='i')

    # Saving Numpy arrays for testing
    f.create_dataset('x_test', data=x_test_255_mean_std, dtype='f')
    f.create_dataset('y_test', data=y_test, dtype='i')


# Printing some values from matrices
print('Original:            ', x_train_255[0, 0, :5, 0])
print('- Mean Image:        ', x_train_255_mean[0, 0, :5, 0])
print('/ Standard Deviation:', x_train_255_mean_std[0, 0, :5, 0])
print()


# Printing some values of Mean Image and Standard Deviation
print('Mean Image:          ', mean_rgb_dataset_custom[0, :5, 0])
print('Standard Deviation:  ', std_rgb_dataset_custom[0, :5, 0])
print()


















