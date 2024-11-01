import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
import logging
import os
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

# Suppress warnings and set random seeds for reproducibility
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging and directories
log_dir = "ml_logs"
model_dir = "saved_models"
plots_dir = "analysis_plots"

for directory in [log_dir, model_dir, plots_dir]:
    os.makedirs(directory, exist_ok=True)

log_file = os.path.join(log_dir, f'prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedNumberPredictor:
    def __init__(self, sequence_length=15):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.rf_model = None
        self.gb_model = None
        self.et_model = None
        self.best_params = None
        self.number_frequencies = np.zeros(51)
        self.history = None

    def analyze_data_patterns(self, data):
        # Analyze the distribution of numbers in the dataset
        self.number_frequencies = np.zeros(51)
        for num in data:
            num_int = int(round(num))
            if 1 <= num_int <= 50:
                self.number_frequencies[num_int] += 1

        total_numbers = np.sum(self.number_frequencies)
        if total_numbers > 0:
            self.number_frequencies = self.number_frequencies / total_numbers

        plt.figure(figsize=(15, 6))
        plt.bar(range(1, 51), self.number_frequencies[1:])
        plt.title('Number Frequency Distribution')
        plt.xlabel('Number')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(plots_dir, 'number_frequencies.png'))
        plt.close()

    def create_sequences(self, data):
        # Create sequences with enhanced feature set
        X, y = [], []
        valid_data = [round(float(num)) for num in data if 1 <= float(num) <= 50]
        valid_data = np.array(valid_data)

        for i in range(len(valid_data) - self.sequence_length):
            sequence = valid_data[i:(i + self.sequence_length)]
            target = valid_data[i + self.sequence_length]

            seq_features = [
                np.mean(sequence),
                np.std(sequence),
                np.min(sequence),
                np.max(sequence),
                np.median(sequence),
                np.percentile(sequence, 25),
                np.percentile(sequence, 75),
                self.number_frequencies[int(round(target))],
                np.diff(sequence).mean(),
                np.diff(sequence).std()
            ]

            X.append(np.concatenate([sequence, seq_features]))
            y.append(target)

        return np.array(X), np.array(y)

    def optimize_ensemble_models(self, X_train, y_train):
        # Optimize each ensemble model's hyperparameters using Optuna
        for model_name, model_cls in [
            ("RandomForest", RandomForestRegressor),
            ("GradientBoosting", GradientBoostingRegressor),
            ("ExtraTrees", ExtraTreesRegressor)
        ]:
            def objective(trial):
                model = model_cls(
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    max_depth=trial.suggest_int('max_depth', 5, 20),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                    random_state=42
                )
                scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5)
                return -scores.mean()

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=5)
            logging.info(f"{model_name} best params: {study.best_params}")

            if model_name == "RandomForest":
                self.rf_model = RandomForestRegressor(**study.best_params, random_state=42)
            elif model_name == "GradientBoosting":
                self.gb_model = GradientBoostingRegressor(**study.best_params, random_state=42)
            elif model_name == "ExtraTrees":
                self.et_model = ExtraTreesRegressor(**study.best_params, random_state=42)

    def optimize_hyperparameters(self, X_train, y_train):
        def objective(trial):
            inputs = Input(shape=(X_train.shape[1], 1))

            x = Bidirectional(LSTM(trial.suggest_int('lstm_units', 32, 128)))(inputs)
            x = BatchNormalization()(x)
            x = Dropout(trial.suggest_float('dropout1', 0.2, 0.4))(x)

            x = Dense(trial.suggest_int('dense_1', 32, 128), activation='relu')(x)
            x = Dropout(trial.suggest_float('dropout2', 0.1, 0.3))(x)

            outputs = Dense(1)(x)
            model = Model(inputs=inputs, outputs=outputs)
            learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='huber')

            early_stopping = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
            history = model.fit(X_train, y_train, epochs=50, batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
                                validation_split=0.2, callbacks=[early_stopping], verbose=0)
            val_loss = min(history.history['val_loss'])
            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)
        self.best_params = study.best_params
        return self.best_params

    def build_model(self, params):
        inputs = Input(shape=(self.sequence_length + 10, 1))
        x = Bidirectional(LSTM(params['lstm_units']))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout1'])(x)
        x = Dense(params['dense_1'], activation='relu')(x)
        x = Dropout(params['dropout2'])(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='huber')
        return model

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        self.optimize_ensemble_models(X_train, y_train)
        self.rf_model.fit(X_train, y_train)
        self.gb_model.fit(X_train, y_train)
        self.et_model.fit(X_train, y_train)

        best_params = self.optimize_hyperparameters(X_train_reshaped, y_train)
        self.lstm_model = self.build_model(best_params)

        checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_model.keras'), monitor='val_loss', save_best_only=True)
        history = self.lstm_model.fit(X_train_reshaped, y_train, epochs=100, batch_size=best_params['batch_size'],
                                      validation_split=0.2, callbacks=[EarlyStopping(patience=10, restore_best_weights=True), checkpoint],
                                      verbose=1)
        self.history = history

        predictions = {
            'lstm': self.lstm_model.predict(X_test_reshaped, verbose=0).flatten(),
            'rf': self.rf_model.predict(X_test),
            'gb': self.gb_model.predict(X_test),
            'et': self.et_model.predict(X_test)
        }

        print("\nModel Performance:")
        for name, pred in predictions.items():
            mae = mean_absolute_error(y_test, pred)
            mse = mean_squared_error(y_test, pred)
            print(f"{name.upper()} - MAE: {mae:.4f}, MSE: {mse:.4f}")

    def predict_next_numbers(self, last_sequence, num_predictions=5):
        predictions = []
        current_sequence = np.array(last_sequence[-self.sequence_length:])

        for _ in range(num_predictions):
            seq_features = [
                np.mean(current_sequence), np.std(current_sequence), np.min(current_sequence),
                np.max(current_sequence), np.median(current_sequence), np.percentile(current_sequence, 25),
                np.percentile(current_sequence, 75), np.mean(self.number_frequencies),
                np.diff(current_sequence).mean(), np.diff(current_sequence).std()
            ]
            full_sequence = np.concatenate([current_sequence, seq_features])
            scaled_sequence = self.scaler.transform(full_sequence.reshape(1, -1))

            lstm_pred = self.lstm_model.predict(scaled_sequence.reshape(1, -1, 1), verbose=0)[0][0]
            rf_pred = self.rf_model.predict(scaled_sequence)[0]
            gb_pred = self.gb_model.predict(scaled_sequence)[0]
            et_pred = self.et_model.predict(scaled_sequence)[0]

            ensemble_pred = (0.4 * lstm_pred + 0.2 * rf_pred + 0.2 * gb_pred + 0.2 * et_pred)
            next_number = int(np.clip(round(ensemble_pred), 1, 50))
            predictions.append(next_number)
            current_sequence = np.append(current_sequence[1:], next_number)

        return np.array(predictions)

def main():
    try:
        data = pd.read_csv('data.txt', sep='\t', header=None).values.flatten()
        predictor = AdvancedNumberPredictor(sequence_length=15)
        predictor.analyze_data_patterns(data)
        X, y = predictor.create_sequences(data)

        if len(X) == 0 or len(y) == 0:
            print("Error: No valid sequences could be created from the data.")
            return

        predictor.fit(X, y)

        last_sequence = data[-predictor.sequence_length:]
        predictions = predictor.predict_next_numbers(last_sequence)

        print("\n=== Final Predictions ===")
        print("Next 5 numbers:", predictions.tolist())

        with open('predictions.txt', 'w') as f:
            f.write("Predicted numbers with details:\n")
            for i, pred in enumerate(predictions, 1):
                f.write(f"Number {i}: {int(pred)}\n")

        print("Predictions saved to 'predictions.txt'")
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
