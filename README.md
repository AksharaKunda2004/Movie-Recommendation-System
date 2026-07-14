# Movie Recommendation System

## Project Overview

The Movie Recommendation System is a desktop application developed using Python that recommends movies based on user preferences. It utilizes Machine Learning techniques with a Support Vector Machine (SVM) classifier and provides an interactive graphical user interface (GUI) built using Tkinter. The application processes movie metadata, trains an SVM model, and recommends the top movies based on duration and rating.

---

## Objectives

- Develop an interactive movie recommendation system.
- Apply Machine Learning for movie recommendation.
- Build a user-friendly desktop GUI.
- Demonstrate data preprocessing, model training, and prediction.

---

## Features

- User-friendly GUI developed with Tkinter.
- Load custom movie datasets in CSV format.
- Automatic data preprocessing and cleaning.
- Feature scaling using MinMaxScaler.
- Movie recommendation using Support Vector Machine (SVM).
- Displays the Top 5 recommended movies.
- Error handling for invalid datasets.
- Fast prediction and easy-to-use interface.

---

## Technologies Used

- Python
- Tkinter
- Pandas
- NumPy
- Scikit-learn
- Support Vector Machine (SVM)
- MinMaxScaler
- Jupyter Notebook

---

## Project Structure

```text
Movie-Recommendation-System/
├── Movie Recommendation System.py
├── Movie Recommendation System Jupyter.ipynb
├── netflix_titles.csv
└── README.md
```

---

## Working Process

1. Load the movie dataset.
2. Validate the required columns.
3. Preprocess the data by handling missing values.
4. Extract numerical values from movie duration.
5. Normalize features using MinMaxScaler.
6. Train the SVM classifier.
7. Predict movie recommendation scores.
8. Display the Top 5 recommended movies through the GUI.

---

## Machine Learning Workflow

- Data Collection
- Data Cleaning
- Feature Engineering
- Feature Scaling
- Model Training
- Model Prediction
- Movie Recommendation

---

## Dataset

The project uses the **Netflix Movies and TV Shows Dataset** containing information such as:

- Movie Title
- Genre
- Rating
- Duration
- Release Year

The dataset is used for training and recommending movies based on learned patterns.

---

## GUI Overview

The desktop application provides:

- Dataset selection button
- Model training
- One-click movie recommendation
- Scrollable recommendation window
- Error and success notifications

---
