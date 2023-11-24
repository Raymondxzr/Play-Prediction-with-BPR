# Play Prediction with Bayesian Personalised Ranking (BPR)
Bayesian Personalized Ranking for predicting whether a user would play a game.

Recently, I participated in a Kaggle contest hosted by UC San Diego as part of the curriculum about Recommender Systems & Web Mining.
The task is to predict given a (user, game) pair whether the user would play the game (0 or 1). Accuracy will be measured in terms of the categorization accuracy (fraction of correct predictions). The test set has been constructed such that exactly 50% of the pairs correspond to played games and the other 50% do not.

At first, I experimented with other models such as Popularity-based, Jaccard Similarities, and Logistic Regression, and none of them worked as well as **Bayesian Personalized Ranking**.

Eventually, I ranked **1 / 604** students and I’m sharing my solution to the problem here.

### Dataset
https://cseweb.ucsd.edu/classes/fa23/cse258-a/files/train.json.gz
The dataset contains positive samples Steam with the following information:
- userID: The ID of the user. This is a hashed user identifier from Steam.
- gameID: The ID of the game. This is a hashed game identifier from Steam.
- text: Text of the user’s review of the game.
- date: Date when the review was entered.
- hours: How many hours the user played the game.
- hours transformed: log2 (hours+1).

### Model
The model is a Bayesian Personalized Ranking (BPR) model with TensorFlow. BPR is a pairwise ranking model that is commonly used for building recommender systems. It operates on user-item interactions and is designed to learn from implicit feedback (like views, clicks, purchases) rather than explicit ratings. The core idea is to maximize the margin between a user's interaction with a positive (observed) item and a negative (unobserved or not interacted with) item.

BPR optimizes for pairwise ranking, which is more relevant for recommendation tasks where the goal is to rank items rather than predict absolute ratings. From what we know about the test set, exactly half of the pairs are positive and half are negative. Instead of having a fixed threshold that focuses more on absolute score, we care more about the ranking of the games played by each user. Furthermore, BPR is tailored to work with implicit feedback datasets, which are common in real-world scenarios like this dataset where explicit feedback like star ratings is unavailable. The loss function of BPR is based on the difference in predictions between a positive and negative item, focusing on the correct ranking rather than the prediction accuracy.

#### Hyperparameters
- Learning Rate: 0.1, which I believe is relatively high, because I want weights are updated with larger steps during each iteration of the gradient descent. 
- Latent Factor Dimensionality: 5, which is the number of elements representing each user and each item. Larger values can increase the complexity, allowing the model to capture more nuances but can potentially lead to overfitting.
- Regularization Strength: 1e-5, a small L2 regularization coefficient to prevent overfitting.
- Iterations: 100, run 100 batches of gradient descent.

### Trick
After training the model, we would be able to predict on a single pair of user and game. The model will output an unnormalized score. Our primary interest is in the relative rankings rather than the precise scores. Therefore, we would calculate scores for all user-game pairs, organize them by user, and then identify the top half of these scores. Those within the **top half** would be assigned a prediction of 1, and the rest would be assigned a prediction of 0. However, since we know that the test set on the public leaderboard is randomly chosen from the entire dataset, which means the 50-50 structure might not hold. So the threshold for making predictions is set at the **average score** for each individual user. For a fixed user, if a game's score is higher than the mean score of this user, it will be assigned a prediction of 1, otherwise 0.

### Result
By using the Bayesian Personalized Ranking model and utilizing all information about the structure of the dataset, I was able to reach an accuracy of **75.71% on the public and 76.63% on the private leaderboard (ranked 1)**.
