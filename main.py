import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import matplotlib.pyplot as plt

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.file_button = tk.Button(self)
        self.file_button["text"] = "Выбрать файл"
        self.file_button["command"] = self.choose_file
        self.file_button.pack(side="top")

        self.run_button = tk.Button(self)
        self.run_button["text"] = "Запустить"
        self.run_button["command"] = self.run_model
        self.run_button.pack(side="top")

        self.plot_button = tk.Button(self)
        self.plot_button["text"] = "Показать график"
        self.plot_button["command"] = self.plot_graph
        self.plot_button.pack(side="top")

    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        print(f"Выбран файл: {self.file_path}")

    def run_model(self):
        # Загрузка данных
        data = pd.read_csv(self.file_path)

        # Предобработка данных
        # Изменение формата даты
        data['open_time'] = pd.to_datetime(data['open_time'])

        # Разделение на признаки (X) и целевую переменную (y)
        X = data[['open', 'high', 'low', 'close', 'volume']].values
        y = data['taker_base_vol'].values

        # Разделение на обучающий и тестовый наборы
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Масштабирование данных
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Создание и обучение модели LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

        model.fit(X_train_reshaped, y_train, batch_size=64, epochs=2)

        # Прогнозирование
        predictions = model.predict(X_test_reshaped)

        # Масштабирование обратно в исходный диапазон
        predictions = scaler.inverse_transform(predictions.reshape(-1, 5))
        result_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})  # Flatten predictions to make them 1D array
        result_df.to_csv('crypto_predictions.csv', index=False)  # Reshape the predictions to match the original shape

    def plot_graph(self):
        old_data = pd.read_csv(self.file_path)
        new_data = pd.read_csv('crypto_predictions.csv')
        plt.figure(figsize=(12, 6))
        plt.plot(old_data['open_time'], old_data['open'], label='Открытие', color='blue')
        plt.xlabel('Время открытия')
        plt.ylabel('Значение')
        plt.title('График открытия криптовалюты')
        plt.legend()
        plt.show()

root = tk.Tk()
app = Application(master=root)
app.mainloop()