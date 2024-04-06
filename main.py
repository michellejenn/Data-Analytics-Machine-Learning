# PART 1 

# QUESTION 1 

 

import sqlite3 

import pandas as pd 

# TASK 1 

# # creating an sqlite database 

My_CarSharing_Database = sqlite3.connect("Car_Sharing_Company.db", isolation_level = None) 

   

# # Create a cursor object used to traverse the records of the database  

Car_CursorObject = My_CarSharing_Database.cursor() 

 

# Reading csv file and then copying table from csv file into Car_Sharing_Company database 

Train = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

Train.to_sql('CarSharing', My_CarSharing_Database, if_exists = 'append', index = False) 

 

# CREATING TWO BACKUP TABLES, BACK_UP1 AND BACK_UP2 

Car_CursorObject.executescript(''' 

                   BEGIN; 

                  CREATE TABLE Back_up1( 

                  id integer,  

                  timestamp date,  

                  season text,  

                  holiday text,  

                  workingday text, 

                  weather text); 

                  CREATE TABLE Back_up2( 

                  id integer, 

                  temp numeric, 

                  temp_feel numeric,  

                  humidity integer,  

                  windspeed numeric,  

                  demand numeric); 

                  COMMIT; 

 

''') 

    

 

# INSERTING DATA FROM CarSharing table in csv INTO Back_up1 and Back_up2 

 

Car_CursorObject.executescript(''' 

                  INSERT INTO Back_up1(id,timestamp,season,holiday,workingday,weather) SELECT id,timestamp,season,holiday,workingday,weather FROM CarSharing; 

 

                  INSERT INTO Back_up2(id, temp, temp_feel, humidity, windspeed, demand) SELECT id, temp, temp_feel, humidity, windspeed, demand FROM CarSharing; 

 

''') 

 

# Car_CursorObject.fetchone('Back_up1') 

Car_CursorObject.execute('SELECT * FROM CarSharing limit 15') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART 1 

# QUESTION 2 

 

# Adding a column to the CarSharing Table 

Car_CursorObject.execute ( '''ALTER TABLE CarSharing  

                              ADD COLUMN humidity_category text''') 

 

# Setting values for humidity_category column in CarSharing table 

Car_CursorObject.execute (''' 

UPDATE CarSharing 

SET humidity_category = 

  CASE  

    WHEN humidity <= 55 THEN 'Dry' 

    WHEN humidity > 55 AND humidity < 65 THEN 'Sticky' 

    ELSE 'Oppressive' 

  END'''); 

 

# PRINTING DATA FROM THE CARSHARING TABLE, THE FIRST 20 ROWS 

Car_CursorObject.execute('SELECT * FROM CarSharing limit 20') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART 1 

# TASK 3A 

# Create a table WEATHER and select columns from the Carsharing Table 

Car_CursorObject.execute (''' 

                        CREATE TABLE WEATHER AS 

                        SELECT id, weather, temp, temp_feel, humidity, windspeed,humidity_category  

                        FROM CarSharing''') 

 

# Temporarily rename the Carsharing table to CarSharing_org 

# Create a new CarSharing table with only the id column 

# Insert the id column data into the new table CarSharing 

# Drop the CarSharing_org table 

Car_CursorObject.executescript(''' 

    ALTER TABLE CarSharing RENAME TO CarSharing_org; 

    CREATE TABLE CarSharing (id integer PRIMARY KEY,timestamp date, season text,holiday text, workingday text, 

    demand numeric);  

    INSERT INTO CarSharing SELECT id, timestamp,  season, holiday, workingday,  demand FROM CarSharing_org; 

    DROP TABLE CarSharing_org;  

    ''') 

 

# PRINTING DATA FROM THE WEATHER TABLE. FIRST 15 ROWS 

Car_CursorObject.execute('SELECT * FROM WEATHER limit 15') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

 
 
 

# PART 1 

# QUESTION 3B 

 

# SELECT WORKINGDAY FROM CARSHARING 

Car_CursorObject.execute(''' 

SELECT DISTINCT workingday FROM CarSharing; 

 

''') 

 

# PRINT THE WORKINGDAY DISTINCT VALUES 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

 

# SELECT HOLIDAY FROM THE CARSHARING TABLE 

Car_CursorObject.execute(''' 

SELECT DISTINCT holiday FROM CarSharing; 

 

''') 

 

# PRINT THE HOLIDAY DISTINCT VALUES 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART 1 

# QUESTION 3B 

#  Connecting to the SQLite database 

Car_CursorObject.executescript(''' 

 

              ALTER TABLE CarSharing 

              ADD COLUMN workingday_code integer; 

 

              ALTER TABLE CarSharing 

              ADD COLUMN holiday_code integer; 

 
 

              UPDATE CarSharing 

              SET workingday_code = 

                CASE  

                  WHEN workingday = 'Yes' THEN 1 

                   

                  ELSE 0 

                  END; 

 

              UPDATE CarSharing 

              SET holiday_code = 

                CASE  

                  WHEN holiday = 'Yes' THEN 1 

                   

                  ELSE 0 

                END;'''); 

 

Car_CursorObject.execute('SELECT * FROM CarSharing limit 15') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART1 

# QUESTION 3C 

 

# EXECUTE THE FOLLOWING SQL STATEMENTS 

Car_CursorObject.executescript(''' 

 

CREATE TABLE Holiday(id integer PRIMARY KEY, holiday text, workingday text, holiday_code integer, workingday_code integer); 

INSERT INTO Holiday SELECT id, holiday, workingday, holiday_code, workingday_code FROM CarSharing; 

 

ALTER TABLE CarSharing RENAME TO CarSharing_org; 

CREATE TABLE CarSharing (id integer PRIMARY KEY,timestamp date, season text, 

demand numeric);  

INSERT INTO CarSharing SELECT id, timestamp,  season, demand FROM CarSharing_org; 

DROP TABLE CarSharing_org;  

 

''') 

 

Car_CursorObject.execute('SELECT * FROM Holiday limit 10') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART 1 

# QUESTION 3D 

 

# EXECUTING THE FOLLOWING SQL STATEMENTS 

Car_CursorObject.executescript( 

     

    ''' 

    DROP TABLE IF EXISTS TIME; 

    CREATE TABLE TIME ( 

    id integer PRIMARY KEY, timestamp datetime, hour integer, weekday_name text, month text, season_name text 

    ); 

 

    INSERT INTO TIME SELECT id, timestamp, strftime('%H', timestamp),  

     

    CASE strftime('%w', timestamp)  

      WHEN  '0' THEN 'Sunday' 

      WHEN  '1' THEN 'Monday' 

      WHEN  '2' THEN 'Tuesday' 

      WHEN  '3' THEN 'Wednesday'  

      WHEN  '4' THEN 'Thursday' 

      WHEN  '5' THEN 'Friday'  

      ELSE 'Saturday' 

    END, 

   strftime('%m', timestamp),season FROM CarSharing; 

 

     

 

    ALTER TABLE CarSharing RENAME TO CarSharing_org; 

     

    CREATE TABLE CarSharing (id integer PRIMARY KEY, demand integer); 

    INSERT INTO CarSharing SELECT id, demand FROM CarSharing_org; 

     

    DROP TABLE CarSharing_org 

 

    

 

    ''' 

) 

 

# PRINT TIME TABLE WITH. 10 ROWS OF THE TABLE 

Car_CursorObject.execute('SELECT * FROM Time limit 10') 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

# PART 1 

# QUESTION 4A 

 
 

Car_CursorObject.execute(''' 

SELECT t.timestamp, c.demand,w.temp 

    FROM CarSharing AS c 

    JOIN Time AS t ON c.id = t.id 

    JOIN Weather AS w ON c.id = w.id 

    WHERE t.id = ( 

        SELECT t_min.id 

        FROM Time AS t_min 

        JOIN Weather AS w ON t_min.id = w.id 

        WHERE w.temp = ( 

            SELECT MIN(temp) 

            FROM Weather 

        ) 

    ) 

''') 

 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

  print(row) 

 

 

# PART 1 

# QUESTION 4B 

Car_CursorObject.execute(''' 

    SELECT h.workingday, AVG(w.windspeed) AS avg_windspeed, MAX(w.windspeed) AS max_windspeed, MIN(w.windspeed) AS min_windspeed, 

           AVG(w.humidity) AS avg_humidity, MAX(w.humidity) AS max_humidity, MIN(w.humidity) AS min_humidity 

    FROM Weather AS w 

    JOIN Time AS t ON w.id = t.id 

    JOIN Holiday AS h ON w.id = h.id 

    WHERE t.timestamp BETWEEN '2017-01-01' AND '2017-12-31' 

    GROUP BY h.workingday 

''') 

 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

    print(row) 

# PART 1 

# QUESTION 4C 

# EXECUTE THE FOLLOWING SCRIPT 

Car_CursorObject.execute(''' 

    SELECT t.weekday_name, t.month, t.season_name, AVG(c.demand) AS avg_demand 

    FROM CarSharing AS c 

    JOIN Time AS t ON c.id = t.id 

    WHERE t.timestamp BETWEEN '2017-01-01' AND '2017-12-31' 

    GROUP BY t.weekday_name, t.month, t.season_name 

    HAVING AVG(c.demand) = ( 

        SELECT MAX(avg_demand) 

        FROM ( 

            SELECT AVG(demand) AS avg_demand 

            FROM CarSharing AS c 

            JOIN Time AS t ON c.id = t.id 

            WHERE t.timestamp BETWEEN '2017-01-01' AND '2017-12-31' 

            GROUP BY t.weekday_name, t.month, t.season_name 

        ) 

    ) 

''') 

 

# PRINTS THE COMPUTED DATA ABOVE 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

    print(row) 

 

 

 

# PART 1 

# QUESTION 4D 

# EXECUTE THE FOLLOWING SCRIPT 

Car_CursorObject.execute(''' 

    SELECT w.humidity_category, AVG(c.demand) AS avg_demand 

    FROM CarSharing AS c 

    JOIN Time AS t ON c.id = t.id 

    JOIN Weather AS w ON c.id = w.id 

    WHERE strftime('%Y', t.timestamp) = '2017' 

    GROUP BY w.humidity_category 

    ORDER BY avg_demand DESC 

''') 

 

# PRINT DATA ABOVE 

print([dec[0] for dec in Car_CursorObject.description]) 

printout = Car_CursorObject.fetchall() 

for row in printout: 

    print(row) 

# PART 2 

# QUESTION 1 

 

# IMPORTING MODULES 

import pandas as pd 

from sklearn.preprocessing import MinMaxScaler 

 

# READING DATA FROM CSV FILE INTO PANDAS DATA FRAME df 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

 

df.interpolate(method="linear", inplace=True) 

 

df['timestamp'] = pd.to_datetime(df['timestamp']) 

 

df = pd.get_dummies(df, columns=['weather', 'season', 'workingday', 'holiday']) 

 
 

print(df.info()) 

print(df) 

# PART 2 

# QUESTION 2 

 

# IMPORTING MODULES 

import pandas as pd 

import statsmodels.api as sm 

import matplotlib.pyplot as plt 

 

#create DataFrame 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

df.interpolate(method="linear", inplace=True) 

 

#define predictor and response variables 

y = df['demand'] 

x = df[['temp', 'humidity', 'windspeed']] 

x['workingday'] = df['workingday'].map({'Yes': 1, 'No': 0}) 

 

#add constant to predictor variables 

x = sm.add_constant(x) 

#fit linear regression model 

model = sm.OLS(y, x).fit() 

y_pred = model.predict(x) 

 

#view model summary 

print(model.summary() 

 

 

# PART 2 

# QUESTION 3 

 

# importing all the necessary modules 

import pandas as pd 

from sklearn.model_selection import train_test_split 

from matplotlib import pyplot as plt 

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 

from statsmodels.tsa.stattools import adfuller 

from statsmodels.tsa.arima.model import ARIMA 

from sklearn.preprocessing import normalize 

 

# reading my csv file into a pandas dataframe df 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

 

# interpolating my temperature data to remove null values 

temperature_ = df['temp'].interpolate() 

 

# converting timestamp to datetime and calculating the mean 

df['timestamp'] = pd.to_datetime(df['timestamp']) 

weekly_avg_temp_2017 = df_2017['temp'].resample('W').mean() 

 

# Seperating my train data 

train_step = int(len(weekly_avg_temp_2017) * 0.7)   

train_data, test_data = weekly_avg_temp_2017[:train_step], weekly_avg_temp_2017[train_step:] 

 

#  fitting my model and printing the model summary 

model = ARIMA(train, order=(1, 0, 1))   

model_fit = model.fit() 

print(model_fit.summary()) 

 

# setting a start and end date for my prediction 

start_date = '2017-01-01' 

 

end_date = '2017-12-31' 

data_predictions = model_fit.predict(start=start_date, end=end_date, typ='levels') 

test_data = weekly_data[start_date:end_date] 

 

# dropping null values in my data 

test_data = test_data.dropna() 

data_predictions = data_predictions.dropna() 

data_predictions = data_predictions[:len(test_data)] 

 

# calculating the mean squared error of my model 

mse = mean_squared_error(test_data, data_predictions) 

 

print('Mean Squared Error:', mse) 

 

# testing the model with the test data 

plt.plot(test_data.index, test_data, label='Actual') 

plt.plot(predictions.index, data_predictions, label='Predicted') 

plt.xlabel('Date') 

plt.ylabel('Weekly Average Temperature') 

plt.title('ARIMA Model - Weekly Average Temperature Prediction for 2017') 

plt.legend() 

plt.show() 

 

# the rersidual plot 

residuals = pd.DataFrame(model_fit.resid) 

fig, ax = plt.subplots(1, 2) 

residuals.plot(title="Residuals", ax=ax[0]) 

residuals.plot(kind='kde', title='Density', ax=ax[1]) 

plt.show() 

 
 

 

 

# PART 2 

# QUESTION 4 

 

# IMPORTING THE FOLLOWING MODULES 

import pandas as pd 

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.preprocessing import StandardScaler, OneHotEncoder 

from sklearn.compose import ColumnTransformer 

from sklearn.metrics import accuracy_score 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.neural_network import MLPClassifier 

 

# READING THE CSV FILE INTO A PANDAS DATAFRAME df 

 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

df.interpolate(method="linear", inplace=True) 

df['timestamp'] = pd.to_datetime(df['timestamp'])  

df_2017 = df[df['timestamp'].dt.year == 2017]  

      

 

# SELECTING THE TARGET AND FEATURE VARIABLES 

features = ['temp', 'humidity', 'windspeed', 'season'] 

target = 'weather' 

X = df_2017[features] 

y = df_2017[target] 

 

# SPLITTING THE DATA INTO TRAIN AND TEST 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

 

# Preprocessing: One-hot encoding for categorical variable, standardization for continuous variables 

preprocessor = ColumnTransformer( 

    transformers=[ 

        ('num', StandardScaler(), ['temp', 'humidity', 'windspeed']), 

        ('cat', OneHotEncoder(), ['season']) 

    ]) 

 

X_train_preprocessed = preprocessor.fit_transform(X_train) 

X_test_preprocessed = preprocessor.transform(X_test) 

 

# Model training and evaluation with SVM 

svm = SVC() 

svm.fit(X_train_preprocessed, y_train) 

y_pred_svm = svm.predict(X_test_preprocessed) 

accuracy_svm = accuracy_score(y_test, y_pred_svm) 

 

# Model training and evaluation with Random Forest 

rf = RandomForestClassifier() 

rf.fit(X_train_preprocessed, y_train) 

y_pred_rf = rf.predict(X_test_preprocessed) 

accuracy_rf = accuracy_score(y_test, y_pred_rf) 

 

# Model training and evaluation with MLPClassifier (Neural Network) 

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500) 

mlp.fit(X_train_preprocessed, y_train) 

y_pred_mlp = mlp.predict(X_test_preprocessed) 

accuracy_mlp = accuracy_score(y_test, y_pred_mlp) 

 

# fitting the preprocessed tain data into themodel 

model_fit = svm.fit(X_train_preprocessed, y_train) 

 

# Printing the accuracy of the SVM model, random forest and neural network accuracy 

print("Random Forest Accuracy:", accuracy_rf) 

print("Support Vector Machines (SVM) Accuracy:", accuracy_svm) 

print("Neural Network Accuracy:", accuracy_mlp) 

 

# PART 2 

# QUESTION 5 

 

# IMPORTING NECESSARY MODULES 

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV 

from sklearn.neural_network import MLPRegressor 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.metrics import mean_squared_error 

from sklearn.compose import ColumnTransformer 

from sklearn.preprocessing import StandardScaler, OneHotEncoder 

 

# READING THE CSV FILE INTO A PANDAS DATAFRAME df 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

 

# Data preprocessing to deal with null values 

df.interpolate(method="linear", inplace=True) 

df['timestamp'] = pd.to_datetime(df['timestamp']) 

 

# selecting feature and target variables 

features = ['temp', 'humidity', 'windspeed', 'temp_feel', 'season', 'weather', 'holiday', 'workingday'] 

target = 'demand' 

X = df[features] 

y = df[target] 

 

# Splitting the data into training and testing sets 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

 

# Preprocessing: One-hot encoding for categorical variables, standardization for continuous variables 

preprocessor = ColumnTransformer( 

    transformers=[ 

        ('num', StandardScaler(), ['temp', 'humidity', 'windspeed', 'temp_feel']), 

        ('cat', OneHotEncoder(), ['season', 'weather', 'holiday', 'workingday']) 

    ]) 

 

X_train_preprocessed = preprocessor.fit_transform(X_train) 

X_test_preprocessed = preprocessor.transform(X_test) 

 

# Performing hyperparameter tuning for MLPRegressor 

param_grid = { 

    'hidden_layer_sizes': [(100,), (100, 50), (50, 50, 50)], 

    # 'max_iter': [100, 200, 300, 400], 

    # Add more hyperparameters to tune if desired 

} 

 

grid_search = GridSearchCV(estimator=MLPRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=3) 

grid_search.fit(X_train_preprocessed, y_train) 

 

# Get the best MLPRegressor model and evaluate on the test set 

best_mlp = grid_search.best_estimator_ 

y_pred_mlp = best_mlp.predict(X_test_preprocessed) 

mse_mlp = mean_squared_error(y_test, y_pred_mlp) 

print("Optimal MLP - MSE:", mse_mlp) 

 

# Train and evaluate the Random Forest Regressor 

rf = RandomForestRegressor() 

rf.fit(X_train_preprocessed, y_train) 

y_pred_rf = rf.predict(X_test_preprocessed) 

mse_rf = mean_squared_error(y_test, y_pred_rf) 

print("Random Forest - MSE:", mse_rf) 

 

 

# PART 2 

# QUESTION 6 

# IMPORTING THE NECESSARY MODULES 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 

 

# reading the csv file into a pandas dataframe df 

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/CarSharing.csv') 

data = df['humidity'] 

 

# fill rows with missing values 

data_interpolated = data.interpolate(method='linear') 

 

data_reshaped = data_interpolated.values.reshape(-1, 1) 

inertias = [] 

for i in range(1, 11): 

    kmeans = KMeans(n_clusters=i, n_init=10)  # Set the value of n_init explicitly 

    kmeans.fit(data_reshaped) 

    inertias.append(kmeans.inertia_) 

 

# plotting the data 

plt.plot(range(1, 11), inertias, marker='o') 

plt.title('Elbow Method') 

plt.xlabel('Number of Clusters') 

plt.ylabel('Inertia') 

plt.show() 

 

 

 