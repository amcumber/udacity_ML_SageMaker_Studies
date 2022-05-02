from __future__ import print_function

import argparse
import os

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. 
# from sklearn.externals import joblib
# Import joblib package directly
import joblib

## TODO: Import any additional libraries you need to define a model


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model

def get_model(max_depth):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('tree', DecisionTreeClassifier(max_depth=max_depth)),
    ])

def get_training_data(training_dir):
    # MOVED to function for encapsulation
    
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    return train_x, train_y

def main(training_dir, model_dir, **kwargs) -> None:
    """Main Entrypoint"""
    ## --- Your code here --- ##
    # Read in csv training file
    train_x, train_y = get_training_data(training_dir)
    

    ## TODO: Define a model 
    model = get_model(**kwargs)
     
    ## TODO: Train the model
    model.fit(train_x, train_y)
      
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    
## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('--max-depth', type=int, default=4, help='max depth of decision tree')
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    # --- [ACM] -- Moved data handling code into a main function
    main(training_dir=args.data_dir, model_dir=args.model_dir, max_depth=args.max_depth)   
