import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Step 1: Load dataset
df = pd.read_csv('D:/settyl/ratings_Beauty.csv')

# Step 2: Data Preprocessing
df = df[['UserId', 'ProductId', 'Rating']]  # Keep only necessary columns

# Step 3: Collaborative Filtering using SVD
reader = Reader(rating_scale=(1, 5))  # Assuming ratings are between 1 and 5
data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Rating']], reader)

# Step 4: Train the SVD model
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Step 5: Make Predictions on the Test Set
predictions = model.test(testset)

# Step 6: Evaluation Metrics (RMSE, MAE, Precision, Recall, F1 Score)

# RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error([pred.est for pred in predictions], [pred.r_ui for pred in predictions]))

# MAE (Mean Absolute Error)
mae = mean_absolute_error([pred.est for pred in predictions], [pred.r_ui for pred in predictions])

# Precision and Recall (Thresholding for Good Predictions)
threshold = 4  # Consider a rating of 4 or above as "good"
recommended_items = [pred for pred in predictions if pred.est >= threshold]
true_positives = len([pred for pred in recommended_items if pred.r_ui >= threshold])
false_positives = len([pred for pred in recommended_items if pred.r_ui < threshold])
false_negatives = len([pred for pred in predictions if pred.est < threshold and pred.r_ui >= threshold])

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# Step 7: Real-Time Recommendation Function
def get_recommendations(user_id, df, model, top_n=5):
    """
    Provide recommendations for a given user.
    """
    user_data = df[df['UserId'] == user_id]
    user_rated_products = set(user_data['ProductId'].tolist())
    all_products = set(df['ProductId'].unique())
    unrated_products = list(all_products - user_rated_products)

    # Check for cold-start users
    if not unrated_products:
        logging.info(f"No unrated products for user {user_id}.")
        return []

    predicted_ratings = []
    for product in unrated_products:
        try:
            pred = model.predict(user_id, product)
            predicted_ratings.append(pred)
        except Exception as e:
            logging.error(f"Prediction error for user {user_id} and product {product}: {e}")
            continue

    # Sort by predicted rating
    sorted_predictions = sorted(predicted_ratings, key=lambda x: x.est, reverse=True)

    return [pred.iid for pred in sorted_predictions[:top_n]]


def update_model_with_new_data(df, user_id, product_id, rating):
    """
    Update dataset and retrain the model with new user-product interactions.
    """
    new_row = pd.DataFrame({'UserId': [user_id], 'ProductId': [product_id], 'Rating': [rating]})
    df = pd.concat([df, new_row], ignore_index=True)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Rating']], reader)
    full_trainset = data.build_full_trainset()

    # Retrain model
    model = SVD()
    model.fit(full_trainset)

    return model, df

# Example Usage
user_id = 1
recommendations_user1 = get_recommendations(user_id, df, model, top_n=5)
print(f"Initial recommendations for user {user_id}: {recommendations_user1}")

# Simulating a new interaction (user 1 rates a new product)
model, df = update_model_with_new_data(df, user_id=1, product_id=107, rating=4)

# Updated recommendations for user 1
updated_recommendations_user1 = get_recommendations(user_id, df, model, top_n=5)
print(f"Updated recommendations for user {user_id}: {updated_recommendations_user1}")

# Recommendations for user 2
user_id = 2
recommendations_user2 = get_recommendations(user_id, df, model, top_n=5)
print(f"Recommendations for user {user_id}: {recommendations_user2}")

# Simulating a new interaction (user 2 rates a new product)
model, df = update_model_with_new_data(df, user_id=2, product_id=108, rating=5)

# Updated recommendations for user 2
updated_recommendations_user2 = get_recommendations(user_id, df, model, top_n=5)
print(f"Updated recommendations for user {user_id}: {updated_recommendations_user2}")