from azureml.core import Workspace, Dataset
import os 

subscription_id = 'aba0e549-d244-475e-9d08-a277687b1608'
resource_group = 'duomenu-platforma'
workspace_name = 'vasa-ml-workspace'

workspace = Workspace(subscription_id, resource_group, workspace_name)

def pipeline():

    # Infering the current file path
    cur_path = os.path.dirname(os.path.realpath(__file__))

    # Getting the dataset
    dataset = Dataset.get_by_name(workspace, name='snow-identifier-labeling_20231221_121709')
    dataset = dataset.to_pandas_dataframe()

    # Defining the directory where to save the labels 
    labels_dir = os.path.join(cur_path, 'labels')
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    # Saving each label into a separate txt file 
    for i, row in dataset.iterrows():
        # Ifering the image name 
        img_name = str(row['image_url'])

        # Only taking the basename 
        img_name = os.path.basename(img_name)

        # Creating the name for the label 
        label_name = img_name.split('.')[0]
        label_name = label_name + '.txt'

        # Defining the path to the label 
        label_path = os.path.join(labels_dir, label_name)

        # Extracting the labels 
        labels = row['label']
        labels = [x['polygon'][0] for x in labels]

        # Converting each polygon to a string separated by whitespace
        labels_str = []
        for label in labels:
            # Converting to str 
            label = ' '.join([str(x) for x in label])

            # Adding the class index 
            label = f"0 {label}"

            # Appending 
            labels_str.append(label)

        # Iterating over the labels and saving the polygons 
        with open(label_path, 'w') as f:
            for label in labels_str:
                f.write(label + '\n')

if __name__ == '__main__': 
    pipeline()