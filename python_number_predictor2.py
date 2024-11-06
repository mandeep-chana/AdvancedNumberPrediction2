import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
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

    def preprocess_data(self, data):
        # Convert to numpy array if not already
        data = np.array(data)

        # Remove outliers using IQR method
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter outliers and ensure values are within [1, 50]
        cleaned_data = data[(data >= max(1, lower_bound)) & (data <= min(50, upper_bound))]

        # Round to nearest integer
        cleaned_data = np.round(cleaned_data)

        return cleaned_data

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
        X, y = [], []
        valid_data = [round(float(num)) for num in data if 1 <= float(num) <= 50]
        valid_data = np.array(valid_data)

        for i in range(len(valid_data) - self.sequence_length):
            sequence = valid_data[i:(i + self.sequence_length)]
            target = valid_data[i + self.sequence_length]

            # Enhanced feature set
            try:
                mode_result = stats.mode(sequence)
                if isinstance(mode_result, tuple):
                    mode_value = mode_result[0]  # For older scipy versions
                else:
                    mode_value = mode_result.mode[0]  # For newer scipy versions
            except:
                mode_value = np.median(sequence)  # Fallback if mode fails

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
                np.diff(sequence).std(),
                np.count_nonzero(np.diff(sequence) > 0) / (len(sequence) - 1),
                np.polyfit(range(len(sequence)), sequence, 1)[0],
                np.sum(np.abs(np.diff(sequence))),
                mode_value,
                stats.kurtosis(sequence, nan_policy='omit'),
                stats.skew(sequence, nan_policy='omit')
            ]

            X.append(np.concatenate([sequence, seq_features]))
            y.append(target)

        return np.array(X), np.array(y)

    def optimize_ensemble_models(self, X_train, y_train):
        for model_name, model_cls in [
            ("RandomForest", RandomForestRegressor),
            ("GradientBoosting", GradientBoostingRegressor),
            ("ExtraTrees", ExtraTreesRegressor)
        ]:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 200, 1000),  # Increased
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 8),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),  # Reduced
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 'auto']),
                }

                if model_name == "GradientBoosting":
                    params.update({
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                        'loss': trial.suggest_categorical('loss', ['squared_error', 'huber', 'absolute_error'])
                    })

                model = model_cls(**params, random_state=42)

                # Use stratified k-fold cross-validation
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = []

                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

                    model.fit(X_fold_train, y_fold_train)
                    pred = model.predict(X_fold_val)
                    score = mean_squared_error(y_fold_val, pred)
                    scores.append(score)

                return np.mean(scores)

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=30)  # Increased number of trials

            best_params = study.best_params
            logging.info(f"{model_name} best params: {best_params}")

            if model_name == "RandomForest":
                self.rf_model = RandomForestRegressor(**best_params, random_state=42)
            elif model_name == "GradientBoosting":
                self.gb_model = GradientBoostingRegressor(**best_params, random_state=42)
            elif model_name == "ExtraTrees":
                self.et_model = ExtraTreesRegressor(**best_params, random_state=42)

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
        # sequence_length + number of additional features (16)
        total_features = self.sequence_length + 16

        inputs = Input(shape=(total_features, 1))

        x = Bidirectional(LSTM(params['lstm_units'], return_sequences=True))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout1'])(x)

        residual = x
        x = Bidirectional(LSTM(params['lstm_units'] // 2, return_sequences=True))(x)
        x = Add()([x, residual])

        x = Bidirectional(LSTM(params['lstm_units'] // 4))(x)
        x = BatchNormalization()(x)

        x = Dense(params['dense_1'], activation='relu')(x)
        x = Dropout(params['dropout2'])(x)
        x = Dense(params['dense_1'] // 2, activation='relu')(x)

        outputs = Dense(1, activation='relu')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                      loss='huber',
                      metrics=['mae', 'mse'])
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

    def create_sequences(self, data):
        X, y = [], []
        valid_data = [round(float(num)) for num in data if 1 <= float(num) <= 50]
        valid_data = np.array(valid_data)

        for i in range(len(valid_data) - self.sequence_length):
            sequence = valid_data[i:(i + self.sequence_length)]
            target = valid_data[i + self.sequence_length]

            # Enhanced feature set
            try:
                mode_result = stats.mode(sequence)
                if isinstance(mode_result, tuple):
                    mode_value = mode_result[0]  # For older scipy versions
                else:
                    mode_value = mode_result.mode[0]  # For newer scipy versions
            except:
                mode_value = np.median(sequence)  # Fallback if mode fails

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
                np.diff(sequence).std(),
                np.count_nonzero(np.diff(sequence) > 0) / (len(sequence) - 1),
                np.polyfit(range(len(sequence)), sequence, 1)[0],
                np.sum(np.abs(np.diff(sequence))),
                mode_value,
                stats.kurtosis(sequence, nan_policy='omit'),
                stats.skew(sequence, nan_policy='omit')
            ]

            X.append(np.concatenate([sequence, seq_features]))
            y.append(target)

        return np.array(X), np.array(y)

    def refine_prediction(self, pred, current_sequence):
        """Refine the prediction using historical patterns"""
        # Round to nearest integer
        rounded_pred = round(pred)

        # Ensure prediction is within bounds
        if rounded_pred < 1:
            rounded_pred = 1
        elif rounded_pred > 50:
            rounded_pred = 50

        # Check if prediction is too far from recent numbers
        recent_mean = np.mean(current_sequence[-5:])
        recent_std = np.std(current_sequence[-5:])

        if abs(rounded_pred - recent_mean) > 3 * recent_std:
            # If prediction is too far, bring it closer to recent mean
            direction = 1 if rounded_pred > recent_mean else -1
            rounded_pred = int(recent_mean + direction * 2 * recent_std)
            rounded_pred = max(1, min(50, rounded_pred))

        # Avoid recent duplicates
        if rounded_pred in current_sequence[-3:]:
            # Adjust slightly up or down
            adjustment = 1 if np.random.random() > 0.5 else -1
            rounded_pred = max(1, min(50, rounded_pred + adjustment))

        return rounded_pred

    def predict_next_numbers(self, last_sequence, num_predictions=5):
        predictions = []
        current_sequence = np.array(last_sequence[-self.sequence_length:])

        # Dynamic weights based on recent performance
        model_weights = {
            'lstm': 0.35,
            'rf': 0.25,
            'gb': 0.25,
            'et': 0.15
        }

        recent_errors = {model: [] for model in model_weights.keys()}

        for _ in range(num_predictions):
            try:
                mode_result = stats.mode(current_sequence)
                if isinstance(mode_result, tuple):
                    mode_value = mode_result[0]
                else:
                    mode_value = mode_result.mode[0]
            except:
                mode_value = np.median(current_sequence)

            seq_features = [
                np.mean(current_sequence),
                np.std(current_sequence),
                np.min(current_sequence),
                np.max(current_sequence),
                np.median(current_sequence),
                np.percentile(current_sequence, 25),
                np.percentile(current_sequence, 75),
                np.mean(self.number_frequencies),
                np.diff(current_sequence).mean(),
                np.diff(current_sequence).std(),
                np.count_nonzero(np.diff(current_sequence) > 0) / (len(current_sequence) - 1),
                np.polyfit(range(len(current_sequence)), current_sequence, 1)[0],
                np.sum(np.abs(np.diff(current_sequence))),
                mode_value,
                stats.kurtosis(current_sequence, nan_policy='omit'),
                stats.skew(current_sequence, nan_policy='omit')
            ]

            full_sequence = np.concatenate([current_sequence, seq_features])
            scaled_sequence = self.scaler.transform(full_sequence.reshape(1, -1))

            predictions_dict = {
                'lstm': self.lstm_model.predict(scaled_sequence.reshape(1, -1, 1), verbose=0)[0][0],
                'rf': self.rf_model.predict(scaled_sequence)[0],
                'gb': self.gb_model.predict(scaled_sequence)[0],
                'et': self.et_model.predict(scaled_sequence)[0]
            }

            # Update weights based on recent performance
            if len(predictions) > 0:
                for model in predictions_dict.keys():
                    error = abs(predictions[-1] - predictions_dict[model])
                    recent_errors[model].append(error)
                    if len(recent_errors[model]) > 3:
                        recent_errors[model].pop(0)

                    # Update weights inversely proportional to error
                    if len(recent_errors[model]) > 0:
                        avg_error = np.mean(recent_errors[model])
                        model_weights[model] = 1.0 / (1.0 + avg_error)

                # Normalize weights
                total_weight = sum(model_weights.values())
                model_weights = {k: v / total_weight for k, v in model_weights.items()}

            weighted_pred = sum(pred * model_weights[model] for model, pred in predictions_dict.items())

            # Refine the prediction
            next_number = self.refine_prediction(weighted_pred, current_sequence)

            predictions.append(next_number)
            current_sequence = np.append(current_sequence[1:], next_number)

        return np.array(predictions)

def main():
    try:
        data = pd.read_csv('data.txt', sep='\t', header=None).values.flatten()
        predictor = AdvancedNumberPredictor(sequence_length=15)

        # Preprocess data
        cleaned_data = predictor.preprocess_data(data)
        predictor.analyze_data_patterns(cleaned_data)
        X, y = predictor.create_sequences(cleaned_data)

        if len(X) == 0 or len(y) == 0:
            print("Error: No valid sequences could be created from the data.")
            return

        predictor.fit(X, y)

        last_sequence = cleaned_data[-predictor.sequence_length:]
        predictions = predictor.predict_next_numbers(last_sequence)

        print("\n=== Final Predictions ===")
        print("Next 5 numbers:", predictions.tolist())

        # Validate predictions
        for pred in predictions:
            if not (1 <= pred <= 50):
                print(f"Warning: Invalid prediction detected: {pred}")

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
