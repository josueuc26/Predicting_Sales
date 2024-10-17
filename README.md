
# Predicting Video Game Sales in Japan with Machine Learning

![Game Sales](./img/coversalesgames.JPG)

## Project Overview

This project focuses on building a predictive model for video game sales in Japan using various machine learning techniques. The goal is to identify patterns in sales based on several factors and to improve the prediction of future video game sales.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Techniques Used](#techniques-used)
- [Results](#results)
- [Conclusions](#conclusions)
- [Observations](#observations)
- [Recommendations](#recommendations)
- [How to Run](#how-to-run)

## Dataset

[Source](https://www.kaggle.com/gregorut/videogamesales) of dataset.

The dataset contains sales data from video games across different regions, including Japan, North America, and Europe. The key columns used in this project are:

| Column        | Explanation                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| Rank          | Ranking of overall sales                                                      |
| Name          | Name of the game                                                              |
| Platform      | Platform of the games release (i.e. PC,PS4, etc.)                             |
| Year          | Year the game was released in                                                 |
| Genre         | Genre of the game                                                             |
| Publisher     | Publisher of the game                                                         |
| NA_Sales      | Number of sales in North America (in millions)                                |
| EU_Sales      | Number of sales in Europe (in millions)                                       |
| JP_Sales      | Number of sales in Japan (in millions)                                        |
| Other_Sales   | Number of sales in other parts of the world (in millions)                     |
| Global_Sales  | Number of total sales (in millions)                                           |

## Techniques Used

The following machine learning and data processing techniques were applied:

- **Scaling and Normalization**: To ensure features are on the same scale.
- **Categorical Encoding**: One-hot encoding was applied to categorical features like 'Publisher'.
- **Model Tuning**: Optimization with `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning.
- **Regression Models**: Various regression models were implemented and compared:
  - XGB Regressor (best performance)
  - Random Forest Regressor
  - Ridge, Lasso, and SVR

## Results

The best performing model was **XGB Regressor** with an R² score of **0.6767**. Other models like **RandomForestRegressor** also showed strong performance but did not outperform XGBoost.

Key findings:

- Adding the 'Year' column improved model performance.
- Log transformation of numerical features worsened model performance.
- Using only numerical features (without categorical ones) resulted in a poor R² score of **0.2124**.
- Including the 'Publisher' feature improved the model performance to a small extent.

## Conclusions

- **Sales patterns differ significantly** between Japan, North America, and Europe. Japanese preferences are influenced by local publishers and differ in terms of console and genre popularity.
- **XGB Regressor** outperformed all other models with an R² score of **0.6767**.
- Including categorical features like 'Publisher' improves the predictive capability of the model.
- Using only numerical features or applying log transformations led to a drop in model performance.

## Observations

- The **Global Sales** feature was excluded from the model to avoid artificially inflating the R² score, as it already includes 'JP Sales'.
- High dimensionality from the 'Publisher' feature was managed by applying Pareto's 80-20 rule, grouping the least frequent publishers under 'Other'. This simplified the model but may have removed useful information.

## Recommendations

- Consider using **Frequency Encoding** for the 'Publisher' feature instead of Pareto 80-20 and one-hot encoding. This may retain more valuable information and improve model performance.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/josueuc26/predicting_Sales.git
   cd Predicting_Sales
2. Install the required packages:

    ```python
    pip install -r requirements.txt
3. Run the Jupyter notebook:

    ```bash
    jupyter notebook
4. Load the dataset (make sure to place the csv file in the correct folder) and execute the notebook.
