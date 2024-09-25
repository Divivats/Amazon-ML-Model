import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

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
y_processed = y

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_processed, test_size=0.2, random_state=42)

# Define hyperparameter tuning space
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.5, 1],
    'n_estimators': [50, 100, 200],
    'gamma': [0, 0.1, 0.5],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [0, 0.1, 0.5]
}

# Set XGBoost parameters to use CUDA GPU acceleration
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_bin': 16,
    'grow_policy': 'lossguide',
    'updater': 'grow_gpu',
    'n_gpus': 1
}

# Perform hyperparameter tuning with GPU acceleration
grid_search = GridSearchCV(xgb.XGBRegressor(**xgb_params), param_grid, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", np.sqrt(-grid_search.best_score_))

# Train model with best hyperparameters using GPU acceleration
best_params = grid_search.best_params_
bst = xgb.XGBRegressor(**xgb_params, **best_params)
dtrain = xgb.DMatrix(X_train, label=y_train, device='cuda')
bst.fit(dtrain, eval_set=[(dtrain, 'train')], early_stopping_rounds=10)

# Save model
bst.save_model('xgb_model.json')

# Load test data and indices
X_test = np.load('test_features.npy')
test_indices = np.load('test_indices.npy')  # Ensure this file exists
test_data = pd.read_csv('test.csv')  # Load test CSV to get entity_names
entity_names = test_data['entity_name'].values

dtest = xgb.DMatrix(X_test, device='cuda')
preds = bst.predict(dtest)

# Allowed units
allowed_units = {'inch', 'ounce', 'fluid ounce', 'pound', 'gallon', 'centilitre', 'watt', 'gram', 'litre', 'cup', 'cubic inch', 'millilitre', 'metre', 'microgram', 'ton', 'millivolt', 'foot', 'imperial gallon', 'decilitre', 'milligram', 'kilovolt', 'millimetre', 'yard', 'centimetre', 'cubic foot', 'microlitre', 'volt', 'quart', 'kilogram', 'kilowatt'}

# Format predictions
formatted_preds = []
for idx, pred in enumerate(preds):
    entity_name = entity_names[idx]
    unit = unit_mapping.get(entity_name, 'unit')  # Default unit
    if unit not in allowed_units:
        unit = 'unit'  # If the unit is not allowed, use a default or empty unit
    formatted_preds.append(f"{pred:.2f} {unit}")

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