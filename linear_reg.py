
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = load_diabetes()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# This function trains a model, logs parameters, and metrics
def train_model(model, model_name):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Model: {model_name}, MSE: {mse}, R2: {r2}")

        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, f"model_{model_name}")

# Start MLflow experiment
mlflow.set_experiment('Diabetes_Model_Comparison')

# Train and log Linear Regression model
train_model(LinearRegression(), "Linear_Regression")

# Train and log Decision Tree model
train_model(DecisionTreeRegressor(max_depth=3), "Decision_Tree")
