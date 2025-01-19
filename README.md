# Real-Time Product Recommendation System for E-Commerce Platform

This project implements a **real-time recommendation system** for an e-commerce platform using **Collaborative Filtering** and **Singular Value Decomposition (SVD)**. The system recommends products to users based on their past interactions with the platform, helping improve user experience by suggesting products that match their interests.

## Overview

The recommendation system uses collaborative filtering to make personalized suggestions based on user-product ratings. The system is designed to be flexible and scalable, allowing real-time updates to model predictions as new user-product interactions are added.

### Features:
- **Real-Time Recommendations**: Provide product suggestions to users based on their ratings.
- **Model Retraining**: Update the recommendation model when new interactions are provided.
- **Evaluation Metrics**: Includes performance metrics such as RMSE, MAE, Precision, Recall, and F1 Score to evaluate model quality.
- **Collaborative Filtering using SVD**: Utilizes Singular Value Decomposition to create user-product interactions matrix.

## Technologies

- **Python**
- **pandas** for data manipulation.
- **numpy** for numerical computations.
- **scikit-learn** for evaluation metrics (MAE, RMSE).
- **surprise** library for collaborative filtering (SVD).
- **logging** for error tracking and real-time updates.
- 


