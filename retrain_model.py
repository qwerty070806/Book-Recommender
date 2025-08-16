import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

# Import the 'app' and 'Rating' model from your Flask app
# to connect to and query your database.
from app import app, Rating


def retrain_model_with_gridsearch():
    """
    Loads all data, performs a GridSearchCV to find the best parameters,
    retrains the final model on all data, and saves it.
    """
    print("Starting model retraining process with GridSearchCV...")

    # --- 1. Load and Combine Data ---
    print("Loading original and new rating data...")
    original_df = pd.read_pickle('ratings_df.pkl')
    with app.app_context():
        new_ratings_query = Rating.query.all()
        new_ratings_list = [
            {'User-ID': r.user_id, 'Book-Title': r.book_title, 'Book-Rating': r.rating}
            for r in new_ratings_query
        ]
    new_ratings_df = pd.DataFrame(new_ratings_list)

    if not new_ratings_df.empty:
        print(f"Found {len(new_ratings_df)} new ratings. Combining datasets...")
        combined_df = pd.concat([
            original_df[['User-ID', 'Book-Title', 'Book-Rating']],
            new_ratings_df[['User-ID', 'Book-Title', 'Book-Rating']]
        ]).drop_duplicates(subset=['User-ID', 'Book-Title'], keep='last')
    else:
        print("No new ratings found. Using original dataset.")
        combined_df = original_df

    # --- 2. Prepare Surprise Dataset ---
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(combined_df[['User-ID', 'Book-Title', 'Book-Rating']], reader)

    # --- 3. Perform GridSearchCV (as requested) ---
    print("Performing GridSearchCV to find best parameters on the new data...")
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30, 50],
        'lr_all': [0.002, 0.005, 0.007],
        'reg_all': [0.02, 0.05, 0.1]
    }
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1, joblib_verbose=2)
    gs.fit(data)

    print(f"Best RMSE from CV: {gs.best_score['rmse']}")
    print(f"Best parameters found: {gs.best_params['rmse']}")

    # --- 4. Train the Final Model ---
    print("Training the final model on the full, combined dataset...")
    # Get the best algorithm with the optimal hyperparameters found
    final_model = gs.best_estimator['rmse']

    # Build a training set from the entire combined dataset
    full_trainset = data.build_full_trainset()

    # Train the final model on ALL the data
    final_model.fit(full_trainset)

    # --- 5. Save the New Model ---
    print("Saving the newly trained model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    print("Retraining complete! Your model is now up-to-date. ✅")


# This makes the script runnable from the command line
if __name__ == '__main__':
    retrain_model_with_gridsearch()