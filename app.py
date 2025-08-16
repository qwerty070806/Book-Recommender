import pickle
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from surprise import SVD

# ---- App and Database Setup ----
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or 'sqlite:///users.db'
db = SQLAlchemy(app)


# Define the User model for the database
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    password_hash = db.Column(db.String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Add this class right after your User class definition
class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    book_title = db.Column(db.String(255), nullable=False)
    rating = db.Column(db.Integer, nullable=False)

# ---- Load the ML model and data ----
try:
    model = pickle.load(open('model.pkl', 'rb'))
    ratings_df = pd.read_pickle('ratings_df.pkl')
    # Add this line after loading ratings_df
    books_df = ratings_df.drop_duplicates(subset=['Book-Title'])

    cosine_data = pickle.load(open('cosine_sim_model.pkl', 'rb'))
    similarity_scores = cosine_data['similarity_scores']
    pt = cosine_data['pt']
    cosine_books_df = cosine_data['books']
    ratings_df['Image-URL-M'] = ratings_df['Image-URL-M'].str.replace('http://', 'https://')
    cosine_books_df['Image-URL-M'] = cosine_books_df['Image-URL-M'].str.replace('http://', 'https://')

except FileNotFoundError:
    print("One or more model/data files not found. Please run the notebook to generate them.")
    exit()


all_book_titles = ratings_df['Book-Title'].unique()


# ---- Recommendation Logic ----
# def get_top_n_recommendations(user_id, n=10):
#     rated_books = ratings_df[ratings_df['User-ID'] == user_id]['Book-Title'].tolist()
#     unrated_books = [book for book in all_book_titles if book not in rated_books]
#     predictions = [model.predict(user_id, book) for book in unrated_books]
#     predictions.sort(key=lambda x: x.est, reverse=True)
#     top_n = [(pred.iid, pred.est) for pred in predictions[:n]]
#     return top_n

def get_top_n_recommendations(user_id, n=10):
    # --- UPDATED LOGIC ---
    # 1. Get rated books from the original dataset
    original_rated_books = ratings_df[ratings_df['User-ID'] == user_id]['Book-Title'].tolist()

    # 2. Get newly rated books from the live database
    with app.app_context():
        newly_rated_books_query = Rating.query.filter_by(user_id=user_id).all()
    newly_rated_books = [r.book_title for r in newly_rated_books_query]

    # 3. Combine both lists to get a complete history of all rated books
    all_rated_books = set(original_rated_books + newly_rated_books)

    # 4. Get the list of books the user has NOT rated
    unrated_books = [book for book in all_book_titles if book not in all_rated_books]

    # 5. Make predictions using the current model
    predictions = [model.predict(user_id, book) for book in unrated_books]

    # Sort the predictions
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top n book titles and their details
    top_n_predictions = predictions[:n]
    top_n_titles = [pred.iid for pred in top_n_predictions]

    recommendation_details = books_df[books_df['Book-Title'].isin(top_n_titles)]
    return recommendation_details.to_dict('records')


# def get_popular_books(n=10):
#     book_summary = ratings_df.groupby('Book-Title')['Book-Rating'].agg(['count', 'mean']).reset_index()
#     popular_books = book_summary[book_summary['count'] >= 100]
#     top_popular = popular_books.sort_values(by='mean', ascending=False)
#     return top_popular.head(n)['Book-Title'].tolist()
def get_popular_books(n=10):
    # This function now returns full book details
    book_summary = ratings_df.groupby('Book-Title')['Book-Rating'].agg(['count', 'mean']).reset_index()
    popular_books = book_summary[book_summary['count'] >= 100]
    top_popular_titles = popular_books.sort_values(by='mean', ascending=False).head(n)
    # Merge with books_df to get the details
    result = pd.merge(top_popular_titles, books_df, on='Book-Title')
    return result.to_dict('records')


def get_similar_books_cosine(book_name, n=5):
    # This is the function you created in your notebook
    try:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:n + 1]

        data = []
        for i in similar_items:
            item = {}
            temp_df = cosine_books_df[cosine_books_df['Book-Title'] == pt.index[i[0]]]
            item['Book-Title'] = temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]
            item['Book-Author'] = temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0]
            item['Image-URL-M'] = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
            data.append(item)

        return data
    except IndexError:
        # This happens if the book is not in the pivot table (not popular enough)
        return []

# ---- Routes ----
@app.route('/')
def home():
    if 'user_id' not in session:
        # For logged-out users, show popular books
        popular_books_list = get_popular_books()
        return render_template('popular.html', books=popular_books_list)

    # For logged-in users
    user_id = session['user_id']

    # --- UPDATED LOGIC TO CHECK BOTH DATA SOURCES ---
    # 1. Count ratings from the new database
    with app.app_context():
        db_rating_count = Rating.query.filter_by(user_id=user_id).count()

    # 2. Count ratings from the original dataset
    df_rating_count = len(ratings_df[ratings_df['User-ID'] == user_id])

    # 3. Get the total count
    total_rating_count = db_rating_count + df_rating_count

    # Now, check the total count against our threshold of 7
    if total_rating_count < 7:
        # This will now only apply to TRULY new users
        books_to_rate = 7 - total_rating_count
        flash(
            f'You have rated {total_rating_count}/7 books. Please rate {books_to_rate} more to unlock your personalized recommendations!')
        popular_books_list = get_popular_books()
        return render_template('popular.html', books=popular_books_list, user_id=user_id)
    else:
        # This will now correctly apply to experienced users like 11676
        recommendations = get_top_n_recommendations(user_id)
        return render_template('home.html', recommendations=recommendations, user_id=user_id)


@app.route('/rate/<book_title>', methods=['POST'])
def rate_book(book_title):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    rating_value = int(request.form['rating'])

    with app.app_context():
        existing_rating = Rating.query.filter_by(user_id=user_id, book_title=book_title).first()
        if existing_rating:
            existing_rating.rating = rating_value
        else:
            new_rating = Rating(user_id=user_id, book_title=book_title, rating=rating_value)
            db.session.add(new_rating)
        db.session.commit()

    # Render the new success page
    return render_template('rated_success.html', book_title=book_title)


@app.route('/book/<book_title>')
def book_page(book_title):
    # This part for getting the main book's details is correct
    book_details = books_df[books_df['Book-Title'] == book_title].iloc[0]
    author = book_details['Book-Author']
    year = book_details['Year-Of-Publication']
    image_url = book_details['Image-URL-M']

    user_rating = None
    if 'user_id' in session:
        user_id = session['user_id']

        # --- UPDATED LOGIC TO CHECK BOTH DATA SOURCES ---
        # 1. First, check the live database for a new rating submitted via the website
        with app.app_context():
            db_rating = Rating.query.filter_by(user_id=user_id, book_title=book_title).first()

        if db_rating:
            # If a rating is found in the database, use it
            user_rating = db_rating.rating
        else:
            # 2. If not in the DB, check the user's original static rating history
            original_rating_series = \
            ratings_df[(ratings_df['User-ID'] == user_id) & (ratings_df['Book-Title'] == book_title)]['Book-Rating']
            if not original_rating_series.empty:
                # If a rating is found in the original data, use it
                user_rating = original_rating_series.iloc[0]

    # This part for getting similar books is correct
    similar_books = get_similar_books_cosine(book_title)

    return render_template('book.html',
                           title=book_title,
                           author=author,
                           year=year,
                           image_url=image_url,
                           user_rating=user_rating,
                           similar_books=similar_books)


# UPDATED: Handles registration form submission
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        password = request.form['password']

        if User.query.get(user_id):
            flash('User ID already exists. Please choose another or log in.')
            return redirect(url_for('register'))

        new_user = User(id=user_id)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))

    return render_template('register.html')


# UPDATED: Handles login form submission
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        password = request.form['password']

        user = User.query.get(user_id)

        if user and user.check_password(password):
            session['user_id'] = user.id
            flash('Login successful!')
            return redirect(url_for('home'))

        flash('Invalid User ID or password.')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))


@app.route('/my_ratings')
def my_ratings():
    # Make sure user is logged in
    if 'user_id' not in session:
        flash('You must be logged in to see your ratings.')
        return redirect(url_for('login'))

    user_id = session['user_id']

    # --- NEW, MORE ROBUST LOGIC ---

    # 1. Get all of the user's rated books from both sources
    with app.app_context():
        db_ratings_query = Rating.query.filter_by(user_id=user_id).all()
    # Create a dictionary of {title: rating} from the database
    db_ratings_dict = {r.book_title: r.rating for r in db_ratings_query}

    # Get ratings from the original dataset
    df_ratings_df = ratings_df[ratings_df['User-ID'] == user_id]
    # Create a dictionary of {title: rating} from the original data
    df_ratings_dict = {row['Book-Title']: row['Book-Rating'] for index, row in df_ratings_df.iterrows()}

    # Combine the two dictionaries, with new DB ratings overwriting old ones
    all_user_ratings = {**df_ratings_dict, **db_ratings_dict}

    # 2. Now, build a clean list, looking up details for each book
    final_ratings_list = []
    for book_title, rating in all_user_ratings.items():
        try:
            # Look up this book in our definitive book details table
            book_details = books_df[books_df['Book-Title'] == book_title].iloc[0]

            rating_item = {
                'Book-Title': book_title,
                'Book-Rating': rating,  # Use the rating from our combined dictionary
                'Book-Author': book_details['Book-Author'],
                'Image-URL-M': book_details['Image-URL-M']
            }
            final_ratings_list.append(rating_item)
        except IndexError:
            # This will skip any rated book that somehow isn't in our main books_df
            # which makes the code even more robust.
            pass

    return render_template('my_ratings.html', ratings=final_ratings_list)


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    # Make sure user is logged in before they can see this page
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the user object from the database
    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        # --- This is the new logic to handle the form submission ---
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')

        # 1. Check if the current password is correct
        if not user.check_password(current_password):
            flash('Your current password was incorrect. Please try again.')
            return redirect(url_for('profile'))

        # 2. Check if the new passwords match
        if new_password != confirm_password:
            flash('New passwords do not match.')
            return redirect(url_for('profile'))

        # 3. Update the password
        user.set_password(new_password)
        db.session.commit()

        flash('Your password has been updated successfully!')
        return redirect(url_for('profile'))

    # This part runs for a normal GET request to just show the page
    return render_template('profile.html')


@app.route('/search')
def search():
    # Get the search query from the URL arguments
    query = request.args.get('query', '')

    if not query:
        # If the query is empty, just redirect to the homepage
        return redirect(url_for('home'))

    # Perform a case-insensitive search on the book titles
    # The `na=False` handles any potential missing title data gracefully
    search_results = books_df[books_df['Book-Title'].str.contains(query, case=False, na=False)]

    # Convert the results to a list of dictionaries to pass to the template
    results_list = search_results.to_dict('records')

    return render_template('search_results.html', query=query, results=results_list)

# ---- Main execution block ----
if __name__ == '__main__':
    app.run(debug=True)