import pandas as pd
import numpy as np

# Load test CSV
test_df = pd.read_csv('test.csv')

# Extract indices and save to a numpy file
test_indices = test_df['index'].values
np.save('test_indices.npy', test_indices)
