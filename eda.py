import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# ⚙️ Configuration & Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

class CarMPGPredictor:
    """End-to-End Machine Learning Pipeline for Car MPG Prediction."""
    
    def __init__(self, data_path: str, target_col: str = 'MPG_City', seed: int = 42):
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.seed = seed
        
        # Setup directories
        self.output_path = Path.cwd() / "plots"
        self.model_path = Path.cwd() / "models"
        self.output_path.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)
        
        # State variables
        self.data: pd.DataFrame = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.preprocessor: ColumnTransformer = None
        self.best_model: Pipeline = None

    def load_and_clean_data(self) -> None:
        """Loads dataset, drops irrelevant columns, and removes duplicates/NaNs."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            logger.error(f"File not found at {self.data_path}. Please check the path.")
            raise

        logger.info(f"Original Shape: {df.shape}")
        
        # Cleaning
        df.drop(['MSRP', 'Invoice'], axis=1, errors='ignore', inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
        self.data = df
        logger.info(f"Cleaned Shape: {self.data.shape}")

    def prepare_pipeline(self) -> None:
        """Splits data and creates the preprocessing pipeline."""
        logger.info("Preparing data split and preprocessing pipelines...")
        
        X = self.data.drop(self.target_col, axis=1)
        y = self.data[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

        num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = X.select_dtypes(include=['object']).columns

        num_pipeline = Pipeline([('scaler', StandardScaler())])
        cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

        self.preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

    def train_and_evaluate(self) -> None:
        """Trains basic linear regression and an optimized Random Forest."""
        logger.info("Training models...")
        
        # 1. Baseline Model: Linear Regression
        lr_pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', LinearRegression())
        ])
        lr_pipe.fit(self.X_train, self.y_train)
        lr_preds = lr_pipe.predict(self.X_test)
        logger.info(f"Linear Regression - R2: {r2_score(self.y_test, lr_preds):.4f}, MAE: {mean_absolute_error(self.y_test, lr_preds):.4f}")

        # 2. Advanced Model: Random Forest with GridSearch
        logger.info("Running GridSearchCV for RandomForest...")
        rf_pipe = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', RandomForestRegressor(random_state=self.seed))
        ])
        
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(rf_pipe, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_model = grid_search.best_estimator_
        rf_preds = self.best_model.predict(self.X_test)
        
        logger.info(f"Best RF Params: {grid_search.best_params_}")
        logger.info(f"Random Forest (Tuned) - R2: {r2_score(self.y_test, rf_preds):.4f}, MAE: {mean_absolute_error(self.y_test, rf_preds):.4f}")

    def generate_visualizations(self) -> None:
        """Generates and saves exploratory and explanatory plots."""
        logger.info("Generating visualizations...")
        sns.set_theme(style="whitegrid")

        # 1. Target Distribution
        plt.figure(figsize=(8, 5))
        sns.histplot(self.data[self.target_col], kde=True, color="blue")
        plt.title(f"Distribution of {self.target_col}")
        plt.savefig(self.output_path / "target_distribution.png", bbox_inches='tight')
        plt.close()

        # 2. Correlation Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Numeric Feature Correlation")
        plt.savefig(self.output_path / "heatmap.png", bbox_inches='tight')
        plt.close()

        # 3. Feature Importance (from tuned RF)
        logger.info("Extracting feature importances...")
        model_step = self.best_model.named_steps['model']
        prep_step = self.best_model.named_steps['preprocessor']
        
        num_cols = prep_step.transformers_[0][2]
        cat_cols = prep_step.transformers_[1][2]
        
        ohe = prep_step.named_transformers_['cat'].named_steps['onehot']
        encoded_features = list(ohe.get_feature_names_out(cat_cols))
        all_features = list(num_cols) + encoded_features
        
        feat_imp = pd.Series(model_step.feature_importances_, index=all_features).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        feat_imp.head(15).plot(kind='barh', color="teal").invert_yaxis()
        plt.title("Top 15 Feature Importances (Random Forest)")
        plt.savefig(self.output_path / "feature_importance.png", bbox_inches='tight')
        plt.close()

    def save_model(self) -> None:
        """Saves the best performing pipeline to disk."""
        model_file = self.model_path / "best_car_model.pkl"
        try:
            joblib.dump(self.best_model, model_file)
            logger.info(f"✅ Model successfully saved to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def run_pipeline(self) -> None:
        """Executes the entire workflow."""
        logger.info("🚀 Starting ML Pipeline...")
        self.load_and_clean_data()
        self.prepare_pipeline()
        self.train_and_evaluate()
        self.generate_visualizations()
        self.save_model()
        logger.info("🏁 Pipeline execution finished.")


# ---------------------------------------------------------
# 🏃‍♂️ Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ensure you replace this with your actual file path
    DATA_FILE = "PASTE_YOUR_FILE_PATH" 
    
    # Initialize and run
    pipeline = CarMPGPredictor(data_path=DATA_FILE, target_col='MPG_City')
    
    # Remove the comment below to execute when you have the data path set
    # pipeline.run_pipeline()
