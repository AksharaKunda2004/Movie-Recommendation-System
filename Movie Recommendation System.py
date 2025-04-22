import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Function to read the dataset and preprocess it
def load_dataset(file_path):
    try:
        # Read the dataset
        data = pd.read_csv(file_path)

        # Ensure required columns are present
        if 'listed_in' not in data.columns or 'title' not in data.columns:
            raise ValueError("Dataset must contain 'listed_in' and 'title' columns.")

        # Convert genres to lowercase for case-insensitive processing
        data['listed_in'] = data['listed_in'].str.lower().str.split(', ')
        data = data.explode('listed_in')

        return data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")
        return None

# Function to preprocess data and train the SVM model
def train_model(data):
    try:
        # Convert genres into a feature matrix using CountVectorizer
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
        X = vectorizer.fit_transform(data['listed_in'].astype(str))

        # Prepare binary labels for training (for each genre)
        def train_genre_model(genre):
            genre = genre.lower()  # Convert genre to lowercase for consistency
            y = data['listed_in'].apply(lambda g: 1 if g == genre else 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = SVC(probability=True)
            model.fit(X_train, y_train)
            return model, vectorizer
        
        return train_genre_model
    except Exception as e:
        messagebox.showerror("Error", f"Model training failed: {e}")
        return None

# Function to recommend movies
def recommend_movies():
    genre = genre_var.get().lower()  # Convert user input to lowercase
    if not genre:
        messagebox.showwarning("Input Required", "Please enter a genre.")
        return

    # Check if the model is trained for the selected genre
    if genre not in genre_models:
        messagebox.showwarning("Error", f"No model available for genre: {genre}. Please select a different genre.")
        return

    # Use the trained model to predict scores
    model, vectorizer = genre_models[genre]
    X = vectorizer.transform(movie_data['listed_in'].astype(str))
    movie_data['score'] = model.predict_proba(X)[:, 1]

    # Filter and sort movies by score
    recommended_movies = movie_data[movie_data['score'] > 0.5].sort_values(by='score', ascending=False)
    if recommended_movies.empty:
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "No movies found for the selected genre.")
    else:
        sorted_movies = recommended_movies['title'].tolist()
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, "\n".join(sorted_movies))

# Function to select dataset file and train models
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        global movie_data, genre_models
        movie_data = load_dataset(file_path)
        if movie_data is not None:
            genre_models = {}
            # Train a model for each unique genre
            train_genre_model = train_model(movie_data)
            unique_genres = movie_data['listed_in'].unique()
            for genre in unique_genres:
                genre_models[genre] = train_genre_model(genre)
            messagebox.showinfo("Success", "Dataset loaded and models trained successfully!")

# Create the GUI window
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("600x400")

# Variables
genre_var = tk.StringVar()
movie_data = None
genre_models = {}

# GUI Layout
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky="NSEW")

# File selection button
file_button = ttk.Button(frame, text="Select Dataset", command=select_file)
file_button.grid(row=0, column=0, pady=10, sticky="W")

# Input field for genre
genre_label = ttk.Label(frame, text="Enter Genre:")
genre_label.grid(row=1, column=0, pady=10, sticky="W")

genre_entry = ttk.Entry(frame, textvariable=genre_var, width=30)
genre_entry.grid(row=1, column=1, pady=10, sticky="W")

# Recommend button
recommend_button = ttk.Button(frame, text="Recommend Movies", command=recommend_movies)
recommend_button.grid(row=2, column=0, columnspan=2, pady=10)

# Scrollable results display
result_frame = ttk.Frame(frame)
result_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="NSEW")

result_text = tk.Text(result_frame, wrap=tk.WORD, height=15, width=50)
result_text.grid(row=0, column=0, sticky="NSEW")

scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=result_text.yview)
scrollbar.grid(row=0, column=1, sticky="NS")

result_text.config(yscrollcommand=scrollbar.set)

# Run the GUI main loop
root.mainloop()