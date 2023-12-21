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
    dataset = Dataset.get_by_name(workspace, name='snow-identifier-data')
    
    # Creating the images dir 
    images_dir = os.path.join(cur_path, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Downloading the dataset
    dataset.download(target_path=images_dir, overwrite=True)

if __name__ == '__main__':
    pipeline()