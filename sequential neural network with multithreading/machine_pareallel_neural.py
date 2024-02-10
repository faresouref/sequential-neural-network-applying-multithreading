import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pandas import read_csv
import numpy as np
import time


class MyThread:
    def __init__(self, target, args=()):
        self.target = target
        self.args = args

    def start(self):
        # Create a new thread and start it
        thread = _Thread(target=self.target, args=self.args)
        thread.start() 


class _Thread:
    def __init__(self, target, args):
        self.target = target
        self.args = args

    def start(self):
        # Simulate starting a new thread by calling the target function
        self.target(*self.args)

    def join(self):
        # Simulate joining a thread by waiting for the target function to finish
        pass 



# Function to create a Keras model
def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=input_data.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model_chunk(chunk, results, lock, result_text_widget):
    input_data, output_data = chunk

    model = create_model()

    model.fit(input_data, output_data, epochs=700, batch_size=30, verbose=0)

    predictions = (model.predict(input_data) > 0.5).astype(float)
    accuracy = accuracy_score(predictions, output_data)

    with lock:
        results.append((accuracy, predictions, output_data))
        print(f"Thread {time.thread_time_ns()} finished.")

        # Display evaluation matrix on the GUI
        result_text_widget.insert(tk.END, "\nEvaluation Metrics:\n")
        result_text_widget.insert(tk.END, print_evaluation_metrics(predictions, output_data))
        result_text_widget.insert(tk.END, "\n")

def print_evaluation_metrics(predictions, output_data):
    result = ""
    result += "Accuracy: " + str(accuracy_score(predictions, output_data)) + "\n"
    result += "Precision: " + str(precision_score(predictions, output_data)) + "\n"
    result += "Recall: " + str(recall_score(predictions, output_data)) + "\n"
    result += "F1 Score: " + str(f1_score(predictions, output_data)) + "\n"
    result += "Confusion Matrix:\n" + str(confusion_matrix(predictions, output_data)) + "\n"
    return result


def load_data():
    file_path = filedialog.askopenfilename()
    data = read_csv(file_path)
    return data.values[:, :-1], data.values[:, -1]

def start_processing():
    global input_data, output_data
    input_data, output_data = load_data()

    # Data preprocessing
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

    num_threads = 4
    data_chunks = list(zip(np.array_split(input_data, num_threads),
                        np.array_split(output_data, num_threads)))

    results = []
    lock = MyThreadLock()

    # GUI Setup
    result_text = "\nOverall Evaluation Metrics:\n"
    result_label.config(text=result_text)

    # Display evaluation matrix on the GUI
    result_text_widget.delete(1.0, tk.END)
    # Clears any existing text in the text widget on the user interface.

    # Train models in parallel
    threads = []
    for i, chunk in enumerate(data_chunks):
        thread = MyThread(target=train_model_chunk, args=(chunk, results, lock, result_text_widget))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish before proceeding(synchrozation)
    for thread in threads:
        thread.join()

    combined_predictions = np.concatenate([result[1] for result in results])
    combined_output_data = np.concatenate([result[2] for result in results])

    # Print overall evaluation metrics
    result_text += "\nOverall Evaluation Metrics:\n"
    result_text += print_evaluation_metrics(combined_predictions, combined_output_data)
    result_label.config(text=result_text)

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Stratified K-Folds cross-validator with 5 splits.
    keras_clf = KerasClassifier(build_fn=create_model, epochs=700, batch_size=30, verbose=0)
    cv_predictions = cross_val_predict(keras_clf, input_data, output_data, cv=skf)

    # Display cross-validation evaluation metrics on the GUI
    result_text_widget.insert(tk.END, "\nCross-Validation Evaluation Metrics:\n")
    result_text_widget.insert(tk.END, print_evaluation_metrics(cv_predictions, output_data))

# Simple lock class to simulate threading.Lock
class MyThreadLock:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

# GUI Setup
root = tk.Tk()
root.title("Machine Learning GUI")
root.geometry("600x500")  # Set initial window size

# Set a custom style for ttk widgets
style = ttk.Style()
style.configure("TButton", foreground="blue", background="blue", font=("Helvetica", 12, "bold"))
style.configure("TLabel", font=("Helvetica", 14, "bold"))

load_button = ttk.Button(root, text="Load Data", command=load_data)
load_button.pack(pady=10)

start_button = ttk.Button(root, text="Start Processing", command=start_processing)
start_button.pack(pady=10)

result_label = ttk.Label(root, text="")
result_label.pack(pady=10)

# Display evaluation matrix on the GUI
result_text_widget = tk.Text(root, height=20, width=50)
result_text_widget.pack(pady=10)

root.mainloop()