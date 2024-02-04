import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

file = 'instagram_reach.csv'
df = pd.read_csv(file)

# Convert 'Time Since Posted' to numerical format
def convert_time_since_posted(time_str):
    if 'hour' in time_str:
        return int(time_str.split()[0])
    else:
        return 0

df['time_since_posted_numeric'] = df['Time since posted'].apply(convert_time_since_posted)

df = df.drop(['USERNAME', 'Caption', 'Hashtags', 'Time since posted'], axis=1)

# Step 2: Feature Selection

# Select input features and target variables
X = df.drop(['Likes', 'time_since_posted_numeric'], axis=1)
y_likes = df['Likes']
y_time_since_posted = df['time_since_posted_numeric']

# Step 3: Train-Test Split
X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)

# Step 4: Model Building

likes_model = RandomForestRegressor()
likes_model.fit(X_train, y_likes_train)

time_model = RandomForestRegressor()
time_model.fit(X_train, y_time_train)

# Step 5: Model Evaluation

likes_predictions = likes_model.predict(X_test)
mse_likes = mean_squared_error(y_likes_test, likes_predictions)
mae_likes = mean_absolute_error(y_likes_test, likes_predictions)
print(f'Mean Squared Error (Likes): {mse_likes}')
print(f'Mean Absolute Error (Likes): {mae_likes}')

time_predictions = time_model.predict(X_test)
mse_time = mean_squared_error(y_time_test, time_predictions)
mae_time = mean_absolute_error(y_time_test, time_predictions)
print(f'Mean Squared Error (Time): {mse_time}')
print(f'Mean Absolute Error (Time): {mae_time}')
