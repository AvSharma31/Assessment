import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

file = 'instagram_reach.csv'

df = pd.read_csv(file)

# Step 2: Feature Engineering

df['caption_length'] = len(str(df['Caption']))
df['hashtag_count'] = df['Hashtags'].apply(lambda x: len(x.split()))

# Step 3: Feature Selection
features = df[['Followers', 'caption_length', 'hashtag_count']]

# Step 4: Target Variables
target_likes = df['Likes']
target_Time_since_posted = df['Time since posted']

# Step 5: Train-Test Split
X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(features, target_likes, target_time_since_posted, test_size=0.2, random_state=42)

# Step 6: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train a Model for Likes Prediction
likes_model = RandomForestRegressor() 
likes_model.fit(X_train_scaled, y_likes_train)

# Step 8: Evaluate the Likes Model
likes_predictions = likes_model.predict(X_test_scaled)
mse_likes = mean_squared_error(y_likes_test, likes_predictions)
mae_likes = mean_absolute_error(y_likes_test, likes_predictions)
print(f'Mean Squared Error (Likes): {mse_likes}')
print(f'Mean Absolute Error (Likes): {mae_likes}')

# Step 9: Train a Model for Time Since Posted Prediction
time_model = RandomForestRegressor() 
time_model.fit(X_train_scaled, y_time_train)

# Step 10: Evaluate the Time Since Posted Model
time_predictions = time_model.predict(X_test_scaled)
mse_time = mean_squared_error(y_time_test, time_predictions)
mae_time = mean_absolute_error(y_time_test, time_predictions)
print(f'Mean Squared Error (Time): {mse_time}')
print(f'Mean Absolute Error (Time): {mae_time}')
