Fake News Detection — Project Bundle
===================================

This project implements a basic Fake News Detection system (binary classification) using Python.
It follows the approach described in the Simplilearn tutorial:
How to Create a Fake News Detection System? — https://www.simplilearn.com/tutorials/machine-learning-tutorial/how-to-create-a-fake-news-detection-system
(Referenced for guidance and structure.)

Included files:
- train_model.py         # Script to load dataset (Fake.csv / True.csv), preprocess, train and save a model.
- app.py                 # Flask web app that loads the trained model and exposes a prediction endpoint.
- requirements.txt       # Python dependencies.
- README.md              # This file.
- dataset_instructions.txt # How to get the dataset from Kaggle and where to place it.
- utils.py               # Small helper functions used by scripts.
- example_input.json     # Example JSON payload for the Flask app.

Dataset used in examples:
- 'Fake.csv' and 'True.csv' from the popular Kaggle "fake-and-real-news-dataset":
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
  (You must download these and place them in a folder named 'data' inside the project.)

How to use:
1. Create a Python virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

2. Place the dataset files:
   project_root/data/Fake.csv
   project_root/data/True.csv

3. Train the model:
   python train_model.py

   This will create a file 'model.joblib' under the project root.

4. Start the Flask app:
   python app.py

   The app will run at http://127.0.0.1:5000 and expose a /predict endpoint that accepts JSON.

Simplilearn tutorial (reference):
https://www.simplilearn.com/tutorials/machine-learning-tutorial/how-to-create-a-fake-news-detection-system