import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

def preprocess_labels(y):
    # Implement your label preprocessing here if necessary
    return y

def format_predictions(preds, entity_names, unit_mapping, allowed_units):
    formatted_preds = []
    for idx, pred in enumerate(preds):
        entity_name = entity_names[idx]
        unit = unit_mapping.get(entity_name, 'unit')  # Default unit
        if unit not in allowed_units:
            unit = 'unit'  # If the unit is not allowed, use a default or empty unit
        formatted_preds.append(f"{pred:.2f} {unit}")
    return formatted_preds

# Define entity_name to unit mapping based on sanity file
unit_mapping = {
    'height': 'centimetre',
    'width': 'centimetre',
    'depth': 'centimetre',
    'item_weight': 'gram',
    'maximum_weight_recommendation': 'gram',
    'wattage': 'watt',
    'voltage': 'volt',
    'item_volume': 'litre'
}

# Load and preprocess data
X = np.load('train_features.npy')
y = np.load('train_labels.npy')

# Preprocess labels
y_processed = preprocess_labels(y)
label_encoder = LabelEncoder()
y_class_labels = label_encoder.fit_transform(y_processed)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_class_labels, test_size=0.2, random_state=42)

# Define parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Train model with early stopping
bst = xgb.train(
    params, 
    dtrain, 
    num_boost_round=1000,
    evals=[(dtrain, 'train'), (dval, 'eval')],
    early_stopping_rounds=10,
    verbose_eval=True
)

# Save model
bst.save_model('xgb_model.json')

# Load test data and indices
X_test = np.load('test_features.npy')
test_indices = np.load('test_indices.npy')  # Ensure this file exists
test_data = pd.read_csv('test.csv')  # Load test CSV to get entity_names
entity_names = test_data['entity_name'].values

dtest = xgb.DMatrix(X_test)
preds = bst.predict(dtest)

# Allowed units
allowed_units = {'inch', 'ounce', 'fluid ounce', 'pound', 'gallon', 'centilitre', 'watt', 'gram', 'litre', 'cup', 'cubic inch', 'millilitre', 'metre', 'microgram', 'ton', 'millivolt', 'foot', 'imperial gallon', 'decilitre', 'milligram', 'kilovolt', 'millimetre', 'yard', 'centimetre', 'cubic foot', 'microlitre', 'volt', 'quart', 'kilogram', 'kilowatt', 'pint'}

# Format predictions
formatted_preds = format_predictions(preds, entity_names, unit_mapping, allowed_units)

# Create submission DataFrame
submission = pd.DataFrame({'index': test_indices, 'prediction': formatted_preds})

# Ensure all indices from test set are included
test_indices_set = set(test_indices)
submission_indices_set = set(submission['index'])
missing_indices = test_indices_set - submission_indices_set

# Add missing indices with empty predictions
for idx in missing_indices:
    submission = submission.append({'index': idx, 'prediction': ''}, ignore_index=True)

# Sort by index to match test set order
submission = submission.sort_values(by='index').reset_index(drop=True)

# Save submission file
submission.to_csv('submission.csv', index=False)

# Calculate and display F1 Score
# Since F1 score is a classification metric, ensure you have predictions and true labels as class labels
y_val_pred = bst.predict(xgb.DMatrix(X_val))
y_val_pred_class = np.round(y_val_pred).astype(int)
print("F1 Score:", f1_score(y_val, y_val_pred_class, average='weighted'))
