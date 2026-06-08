# Spam Email/SMS Classifier

A machine learning web application that classifies a message as **spam** or **ham** using TF-IDF vectorization and a Multinomial Naive Bayes model served through FastAPI. The app uses the SMS Spam Collection dataset and provides both an API endpoint and a simple frontend for real-time prediction.[1][2][3][4]

## Live Demo

- Backend API: `https://spam-classifier-9uh3.onrender.com`[4]
- API Docs: `https://spam-classifier-9uh3.onrender.com/docs`[4]
- Frontend: `https://spam-classifier-1-iajq.onrender.com/`.

## Features

- Classifies SMS or email-like text as spam or ham in real time.[3][2]
- Uses TF-IDF vectorization with word n-grams for text feature extraction.[2][5]
- Trains a Multinomial Naive Bayes model, a standard baseline for text classification.[3]
- Serves predictions through a FastAPI backend with interactive Swagger documentation at `/docs`.[4][6]
- Includes a simple HTML, CSS, and JavaScript frontend for user input and prediction display.
- Deployed on Render and version-controlled on GitHub for portfolio use.[7]

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | Python, Pandas, Scikit-learn, TF-IDF, Multinomial Naive Bayes |
| Backend | FastAPI, Uvicorn, Joblib |
| Frontend | HTML, CSS, JavaScript |
| Deployment | GitHub, Render |

## Project Structure

```text
spam-classifier/
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── data/
│   │   └── SMSSpamCollection
│   ├── models/
│   │   └── spam_classifier.pkl
│   ├── train_model.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Dataset

The project uses the **SMS Spam Collection** dataset, a public dataset of 5,574 labeled SMS messages collected for spam research.[1][8]

## Model Workflow

1. Load the labeled SMS dataset.[1]
2. Split the data into training and testing sets using stratified sampling to preserve class distribution.[9][10]
3. Convert text into numeric features using `TfidfVectorizer` with word n-grams.[2][5]
4. Train a `MultinomialNB` classifier for spam detection.[3]
5. Save the trained pipeline with Joblib for backend reuse.[11]
6. Load the saved model in FastAPI and expose a `/predict` endpoint.[4]

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Christopher230306/spam-classifier.git
cd spam-classifier
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows CMD**

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Train the model

Place the `SMSSpamCollection` file inside `backend/data/`, then run:

```bash
python train_model.py
```

This will generate `backend/models/spam_classifier.pkl`.

### 5. Start the backend

```bash
python -m uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/docs`

### 6. Run the frontend

Open `frontend/index.html` in your browser. Make sure `frontend/app.js` points to the correct backend URL.

## API Example

### POST `/predict`

Request:

```json
{
  "message": "Congratulations! You have won a free lottery ticket. Call now!"
}
```

Response:

```json
{
  "prediction": "spam",
  "confidence": 0.831
}
```

## Evaluation

The model was evaluated using accuracy, precision, recall, F1-score, and confusion matrix, which are common classification metrics for spam detection systems.[12][13]

Example outcomes from project testing showed that very short or ambiguous messages can still be misclassified, which is a normal limitation in text classification systems.[14][13]

## Deployment

The backend is deployed as a Render web service, and the frontend can be deployed as a Render static site connected to the same GitHub repository.[4][15][7]

Typical backend start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Future Improvements

- Add better text preprocessing such as stemming or lemmatization.[16][17]
- Compare Naive Bayes with Logistic Regression or SVM for better spam recall.[18][13]
- Add confidence threshold tuning for borderline predictions.[19]
- Improve the frontend UI and add prediction history.
- Support batch message classification.

## Resume Description

Developed and deployed a machine learning-based spam classifier that predicts whether an SMS or email message is spam or ham using TF-IDF vectorization and Multinomial Naive Bayes. Built a FastAPI backend for real-time prediction, integrated a frontend interface for user interaction, and deployed the application on Render with source code maintained on GitHub.[2][3][4]

## License

This project is intended for educational and portfolio use.
