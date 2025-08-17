# Book Recommender Web Application

A feature-rich, personalized book recommendation web application built with Python, Flask, and two different machine learning models. The application allows users to create accounts, rate books, and receive both personalized and item-based recommendations.

**Live Demo:** [https://my-book-recommender-app.onrender.com/](https://my-book-recommender-app.onrender.com/) 

## Key Features

* **Dual Recommendation Models:**
    * **Personalized "For You" Recommendations:** Uses a Singular Value Decomposition (SVD) model to provide recommendations based on a user's unique rating history.
    * **"Similar Books" Recommendations:** Uses a Cosine Similarity model to find books similar to a selected item based on user rating patterns.
* **Full User Authentication:** Users can register, log in, log out, and change their passwords. All passwords are securely hashed.
* **Interactive Rating System:** A dynamic, front-end star rating system allows users to rate books from 1 to 10.
* **Smart "Cold Start" Onboarding:** New users are prompted to rate at least 7 books to unlock their personalized recommendations, ensuring a higher quality initial experience.
* **Search Functionality:** Users can search the entire book catalog to find and rate books they have already read.
* **Offline Model Retraining:** A separate script (`retrain_model.py`) allows the recommendation models to be updated with new user ratings, ensuring the recommendations get smarter over time.
* **Responsive UI:** The frontend is built to be clean, modern, and responsive for both desktop and mobile use.

## How the Recommendation Engines Work

The application uses two different recommendation "engines" working together to create a professional user experience.

### 1. Singular Value Decomposition (SVD): The "Personalized Taste" Engine

This model powers the **"Recommended For You"** section. Its goal is to understand the deep, underlying tastes of each user.

* **What it does:** SVD is a technique from linear algebra that breaks down the huge, sparse user-book rating matrix into three smaller, denser matrices. It acts like a sophisticated librarian who, instead of just looking at what books you've read, tries to figure out *why* you like them.

* **The Analogy - Discovering a "Taste Map":**
    1.  **It finds hidden characteristics:** SVD automatically discovers hidden "features" or concepts that connect users and books. We call these **latent features**. They might represent genres (like "legal thrillers"), themes ("coming-of-age stories"), or writing styles ("fast-paced plot").
    2.  **It creates profiles:** Based on these features, it builds a unique **"taste profile"** for every user and a **"DNA profile"** for every book.
    3.  **It makes a match:** To recommend a book, the SVD model simply matches a user's taste profile with a book's DNA profile. A strong match results in a high predicted rating.

* **In this app, this engine answers the question:** "Based on this specific user's entire rating history, which new books will best match their unique taste profile?"

### 2. Cosine Similarity: The "Item-to-Item" Engine

This model powers the **"Readers who liked this book also liked..."** section on the book detail pages. Its goal is much more direct: to find books that are conceptually similar to a given book.

* **What it does:** This engine doesn't try to understand individual user tastes. Instead, it measures how similar two items are based on the latent features discovered by the SVD model.

* **The Analogy - Finding "Book Twins":**
    Imagine all your books are placed on a giant map based on their characteristics (their "DNA profile" from SVD). Cosine Similarity is the tool used to find the books that are closest to each other on that map by measuring the angle between their vectors. It doesn't care who the user is; it only cares about the properties of the books themselves. 

* **In this app, this engine answers the question:** "Regardless of the user, what other books are most conceptually similar to the one they are currently looking at?"


## Technologies Used

* **Backend:** Python, Flask, Flask-SQLAlchemy
* **Machine Learning:** Scikit-learn, Surprise, Pandas, NumPy
* **Frontend:** HTML, CSS, Jinja2
* **Database:** PostgreSQL (for production), SQLite (for local development)
* **Deployment:** Git, GitHub, Render, Gunicorn

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/qwerty070806/Book-Recommender.git](https://github.com/qwerty070806/Book-Recommender.git)
    cd Book-Recommender
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the Database:**
    (This will create a local `users.db` file)
    ```bash
    python
    >>> from app import app, db
    >>> with app.app_context():
    ...   db.create_all()
    ...
    >>> exit()
    ```
    *Note: The pre-trained model files (`model.pkl`, `cosine_sim_model.pkl`) are included in the repository. To generate them from scratch, you would need to run the `BookRecoSVD.ipynb` notebook.*

5.  **Run the Flask application:**
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Project Structure

* `app.py`: The main Flask application file containing all routes and backend logic.
* `retrain_model.py`: The offline script to retrain the ML models with new data.
* `requirements.txt`: A list of all Python dependencies.
* `model.pkl`: The saved, pre-trained SVD model.
* `cosine_sim_model.pkl`: The saved data for the Cosine Similarity model.
* `/templates`: Contains all the HTML files for the user interface.
* `/static`: Contains static assets like the background image.
