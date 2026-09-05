import numpy as np
from sklearn.metrics import classification_report
from mlops_churn_api import settings,logging_config
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
StandardScaler,
)
from sklearn.pipeline import Pipeline
import logging
import joblib
import warnings
from pathlib import Path
import json

warnings.filterwarnings("ignore")
class ModelTrainer:
    """Handle data preprocessing, model training, evaluation, and saving."""
    def __init__(self):
        """Initialize paths and the evaluation logger."""
        self.model_path = settings.model_pickle_path
        self.data_path = settings.data_path
        self.evaluation_logger = logging.getLogger("evaluation")
        self.BASE_DIR=BASE_DIR = Path(__file__).resolve().parents[1]


    def data_load(self)->pd.DataFrame:
        """Load the customer churn dataset.

        Returns:
            A DataFrame containing the raw customer churn data.
        """
        return pd.read_csv(self.data_path)

    def pre_processing(self)->tuple[pd.DataFrame,pd.Series]:
        """Clean and transform the dataset for model training.

        Returns:
            A tuple containing the processed features and target variable.
        """
        df = self.data_load()
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df = df.drop(df[df["tenure"] == 0].index)
        df = df.drop(["customerID","PhoneService"], axis=1)

        df.to_csv(self.BASE_DIR/"data"/"processed"/"Telco-Customer-Churn-processed.csv", index=False)

        df = pd.get_dummies(df, drop_first=True)

        x = df.drop("Churn", axis=1)
        y = df["Churn"]

        feature_path = self.BASE_DIR / "models" / "feature_names.json"
        with open(feature_path, "w") as f:
            json.dump(x.columns.tolist(), f)

        return x, y


    def split_data(self)->tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the processed dataset into training and test sets.

        Returns:
            A tuple containing X_train, X_test, y_train, and y_test.
        """
        x,y=self.pre_processing()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return x_train, x_test, y_train, y_test

    def create_model(self)->GridSearchCV:
        """Create a logistic regression pipeline with hyperparameter tuning.

        Returns:
            A configured GridSearchCV instance for model selection.
        """
        log_regression_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("log_regression", LogisticRegression(
                C=10.0,
                penalty="l2",
                solver="liblinear",
                max_iter=10000
            ))
        ])

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        param_grid = [
            {
                "log_regression__solver": ["liblinear"],
                "log_regression__penalty": ["l1", "l2"],
                "log_regression__C": np.logspace(-4, 4, 9),
            },
            {
                "log_regression__solver": ["lbfgs"],
                "log_regression__penalty": ["l2"],
                "log_regression__C": np.logspace(-4, 4, 9),

            },
            {
                "log_regression__solver": ["saga"],
                "log_regression__penalty": ["elasticnet"],
                "log_regression__l1_ratio": [0.25, 0.5, 0.75],
                "log_regression__C": np.logspace(-4, 4, 9),

            },

        ]

        log_regression_grid = GridSearchCV(
            log_regression_pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1
        )

        return log_regression_grid

    def train_model(self)->tuple[Pipeline,pd.DataFrame,pd.Series]:
        """Train the model and return the best fitted pipeline.

        Returns:
            A tuple containing the best fitted pipeline, test features,
            and test target values.
        """
        log_regression_pipe=self.create_model()
        x_train, x_test, y_train, y_test = self.split_data()
        log_regression_pipe.fit(x_train, y_train)
        return log_regression_pipe.best_estimator_ , x_test, y_test

    def evaluate_model(self,x_test:pd.DataFrame, y_test:pd.Series,model:GridSearchCV) -> None:
        """Evaluate the trained model and log classification metrics.

        Args:
            x_test: Test feature data.
            y_test: True target values for the test data.
            model: Fitted machine learning pipeline.
        """
        y_pred = model.predict(x_test)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )

        self.evaluation_logger.info("===== Model Evaluation =====")
        self.evaluation_logger.info("Accuracy : %.4f", report["accuracy"])
        self.evaluation_logger.info("Precision: %.4f", report["1"]["precision"])
        self.evaluation_logger.info("Recall   : %.4f", report["1"]["recall"])
        self.evaluation_logger.info("F1 Score : %.4f", report["1"]["f1-score"])
        self.evaluation_logger.info("============================")

    def save_model(self, model) -> None:
        """Save the trained model to the configured path.

        Args:
            model: Fitted machine learning pipeline to save.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

    def run(self) -> None:
        """Run the complete model training pipeline."""
        model, x_test, y_test = self.train_model()

        self.evaluate_model(x_test, y_test, model)

        self.save_model(model)



if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()














