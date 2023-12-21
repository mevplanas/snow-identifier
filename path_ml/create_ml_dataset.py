# OS module 
import os 
import shutil

# Train test split 
from sklearn.model_selection import train_test_split

def pipeline(): 
    # Infering the current file path
    cur_path = os.path.dirname(os.path.realpath(__file__))

    # Defining the dir to labels and images 
    labels_dir = os.path.join(cur_path, 'labels')
    images_dir = os.path.join(cur_path, 'images')

    # Listing all the labels and images 
    labels = os.listdir(labels_dir)
    images = os.listdir(images_dir)

    # Creating a dictionary for labels and images where the key is the 
    # name of the image without the extension and the value is the
    # name of the label without the extension
    labels_dict = {}
    for label in labels:
        label_name = label.split('.')[0]
        labels_dict[label_name] = label

    images_dict = {}
    for image in images:
        image_name = image.split('.')[0]
        images_dict[image_name] = image

    # Only leaving the intersecting entries 
    intersecting_keys = set(labels_dict.keys()).intersection(set(images_dict.keys()))
    labels_dict = {k: labels_dict[k] for k in intersecting_keys}
    images_dict = {k: images_dict[k] for k in intersecting_keys}

    # Creating the train test split
    train_keys, test_keys = train_test_split(list(intersecting_keys), test_size=0.2)

    # Defining the machine_learning dir 
    machine_learning_dir = os.path.join(cur_path, 'machine_learning')
    if not os.path.exists(machine_learning_dir):
        os.makedirs(machine_learning_dir)

    # Creating the train and test dirs
    train_dir = os.path.join(machine_learning_dir, 'train')
    test_dir = os.path.join(machine_learning_dir, 'val')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Creating the train images, labels and test images, labels dirs 
    train_images_dir = os.path.join(train_dir, 'images')
    train_labels_dir = os.path.join(train_dir, 'labels')
    test_images_dir = os.path.join(test_dir, 'images')
    test_labels_dir = os.path.join(test_dir, 'labels')

    # Creating the train images and labels dirs
    if not os.path.exists(train_images_dir):
        os.makedirs(train_images_dir)
    else: 
        shutil.rmtree(train_images_dir)
        os.makedirs(train_images_dir)
    if not os.path.exists(train_labels_dir):
        os.makedirs(train_labels_dir)
    else:
        shutil.rmtree(train_labels_dir)
        os.makedirs(train_labels_dir)

    # Creating the test images and labels dirs
    if not os.path.exists(test_images_dir):
        os.makedirs(test_images_dir)
    else:
        shutil.rmtree(test_images_dir)
        os.makedirs(test_images_dir)
    if not os.path.exists(test_labels_dir):
        os.makedirs(test_labels_dir)
    else:
        shutil.rmtree(test_labels_dir)
        os.makedirs(test_labels_dir)

    # Copying the train images and labels
    for key in train_keys:
        # Copying the image 
        image_name = images_dict[key]
        image_path = os.path.join(images_dir, image_name)
        shutil.copy(image_path, train_images_dir)

        # Copying the label 
        label_name = labels_dict[key]
        label_path = os.path.join(labels_dir, label_name)
        shutil.copy(label_path, train_labels_dir)

    # Copying the test images and labels
    for key in test_keys:
        # Copying the image 
        image_name = images_dict[key]
        image_path = os.path.join(images_dir, image_name)
        shutil.copy(image_path, test_images_dir)

        # Copying the label 
        label_name = labels_dict[key]
        label_path = os.path.join(labels_dir, label_name)
        shutil.copy(label_path, test_labels_dir)

if __name__ == '__main__':
    pipeline()