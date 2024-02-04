import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

df = pd.read_csv('Bengaluru_House_Data.csv')

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['area_type', 'availability', 'location', 'size', 'society']
numerical_features = ['total_sqft', 'bath', 'balcony']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ])

svr_regressor = SVR()

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', svr_regressor)])

model.fit(X_train, y_train)

test_predictions = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, test_predictions)
print(f'Mean Squared Error: {mse}')