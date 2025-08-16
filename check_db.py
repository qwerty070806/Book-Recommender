from app import app, User, Rating


# This function will query the database and print the users
def view_users():
    with app.app_context():
        # Get all records from the User table
        users = User.query.all()

        if not users:
            print("The 'user' table is empty. No users have been registered yet.")
            return

        print("--- Users in Database ---")
        for user in users:
            print(f"User ID: {user.id}, Password Hash: {user.password_hash}")
        print("-------------------------")

# 2. Add a new function to print the ratings
def view_ratings():
    with app.app_context():
        ratings = Rating.query.all()
        if not ratings:
            print("\nThe 'rating' table is empty. No books have been rated yet.")
            return
        print("\n--- Ratings in Database ---")
        for rating in ratings:
            print(f"User ID: {rating.user_id}, Book: '{rating.book_title}', Rating: {rating.rating}")
        print("---------------------------")

# This makes the script runnable
if __name__ == '__main__':
    view_users()
    view_ratings()