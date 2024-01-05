# Ultralytics model 
from ultralytics import YOLO 

# OS traversal 
import os 

def pipeline():
    # Infering the current file path 
    cur_path = os.path.dirname(os.path.realpath(__file__))

    # Loading the m segment model 
    model = YOLO('yolov8s-seg.pt') 

    # Defining the path to "data"
    data_path = os.path.join(cur_path, 'configuration.yaml')

    # Initiating the model 
    model.train(data=data_path, batch=16, epochs=400)

if __name__ == '__main__':
    # Setting the KMP_DUPLICATE_LIB_OK to True 
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # Running the pipeline
    pipeline()