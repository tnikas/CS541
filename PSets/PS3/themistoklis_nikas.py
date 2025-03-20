model = LinearRegression()
model.fit(X, y)


X_squared = X ** 2
X_new = np.hstack((X, X_squared))
return X_new


if regularization == "L1":
    model = Lasso(alpha=alpha_1)
else:
    model = Ridge(alpha=alpha_2)
steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', model)
]

linear_pipe = Pipeline(steps)
linear_pipe.fit(X_train, y_train)
train_score = linear_pipe.score(X_train, y_train)
test_score = linear_pipe.score(X_test, y_test)
return train_score, test_score


for col in columns:
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
return df


model = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth, min_samples_split=min_samples_split)
model.fit(X, y)
return model


squared_errors = (y_pred - y_true) ** 2
mse = np.mean(squared_errors)
rmse = np.sqrt(mse)
return rmse


model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.2, random_state=2022)


## Step 2: fit the model on X_train, y_train
# In `fit()`, pass (X_train, y_train), (X_test, y_test) as  2 datasets for `eval_set`. You should refer to
# the document to see how to use `eval_set` with `fit()`
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

## Step 3: return the trained model
return model


train_loss = results['validation_0']['rmse']
test_loss = results['validation_1']['rmse']

plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Validation Loss')
plt.xlabel('Number of trees')
plt.ylabel('Loss')
plt.legend()
plt.show()


data = {
    "W. Unst": (207.7, 311),
    "Hermaness": (1570, 3872),
    "N.E. Unst": (1588, 495),
    "W. Yell": (125.7, 134),
    "Buravoe": (353.4, 485),
    "Fetlar": (931, 372),
    "Out Skerries": (1616, 284),
    "Noss": (1317, 10767),
    "Moussa": (614, 1975),
    "Dalsetter": (60.12, 970),
    "Sumburgh": (1273, 3243),
    "Fitful Head": (595.7, 500),
    "St Ninians": (105.6, 250),
    "S. Havra": (241.9, 925),
    "Reawick": (111.1, 970),
    "Vaila": (302.4, 278),
    "Papa Stour": (808.9, 1036),
    "Foula": (2927, 5570),
    "Eshaness": (1069, 2430),
    "Uyea": (898.2, 731),
    "Gruney": (564.8, 1364),
    "Fair Isle": (3957, 17000),
}

areas = np.array([data[colony][0] for colony in data])
populations = np.array([data[colony][1] for colony in data])

log_areas = np.log(areas)

model = LinearRegression()
model.fit(log_areas.reshape(-1, 1), populations)

X_pred = np.linspace(min(log_areas), max(log_areas), 100).reshape(-1, 1)
y_pred = model.predict(X_pred)

plt.scatter(log_areas, populations)
plt.plot(X_pred, y_pred, color='red')

plt.xlabel('Log area')
plt.ylabel('Population size')
plt.title("Regression Model With Outliers")
plt.show()


#create a deep copy of the data
filtered_data = data.copy()

#remove outliers from the data
del filtered_data["Noss"]
del filtered_data["Fair Isle"]

areas_filtered = np.array([filtered_data[colony][0] for colony in filtered_data])
populations_filtered = np.array([filtered_data[colony][1] for colony in filtered_data])

log_areas_filtered = np.log(areas_filtered)

model = LinearRegression()
model.fit(log_areas_filtered.reshape(-1, 1), populations_filtered)

X_pred = np.linspace(min(log_areas_filtered), max(log_areas_filtered), 100).reshape(-1, 1)
y_pred = model.predict(X_pred)

plt.scatter(log_areas, populations)
plt.plot(X_pred, y_pred, color='red')

plt.xlabel('Log area')
plt.ylabel('Population size')
plt.title("Regression Model Excluding Outliers")
plt.show()