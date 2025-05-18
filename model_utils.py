import numpy as np
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# Split the data
def split_data(X, y, test_size=0.2):
    """
    Split the data into the training and testing sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)

# Train the model
def train_model(X_train, y_train, model_type="Linear Regression"):
    """
    Train a model based on the specified type 
    """
    if model_type == "Linear Regression":
        model = LinearRegression()
        
    else:
        model = RandomForestRegressor(n_estimators=70, max_depth=15, 
                                      min_samples_split=5, n_jobs=1, random_state=42)

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model return metrics
    """

    #make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    #calculate the metrics
    metrics = {
            'train_rmse' : np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse' : np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2' : r2_score(y_train, y_pred_train),
            'test_r2' : r2_score(y_test, y_pred_test),
            'y_test' : y_test,
            'y_pred_test' :y_pred_test
            }
    
    return metrics

#save the model
def save_model(model, filename = 'climate_model.pkl'):

    """
    Save the model to the disk
    """

    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load the model
def load_model(filename = 'climate_model.pkl'):
    """
    Load the model that was saved 
    """    

    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        return None