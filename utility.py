# This file contains functions for different stages in the modeling process.
# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# -----------------Data Preparation-------------------!
# function to check for null and duplicate values, and handle them
def clean_nulls_and_duplicates(df):
    """
    This function cleans a dataframe by checking for, and handling null values and duplicate rows.
    It also standardizes the columns by removing the whitespaces between the words, adding a hyphen for readability and capitalizing the first letter in each word.

    Parameters:
        df(pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: A cleaned DataFrame with no duplicate or null values, and standardized columns
    """

    print("Initial shape of the dataset:", df.shape)

    # check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nNull values detected in the following columns:")
        print(null_counts[null_counts > 0])
        
        # drop the missing values if any
        df = df.dropna(axis=0)
        print("Dropped rows with missing values")
    else:
        print("\nNo null values detected.")

    # check for duplicate values
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"\nFound {duplicate_count} duplicate rows.")
        
        # drop the duplicate rows
        df = df.drop_duplicates()
        print("Dropped duplicate rows.")
    else:
        print("\nNo duplicate rows detected.")

    # Standardize the column names
    df.columns = (
        df.columns
        .str.strip()                          # Remove leading/trailing whitespace
        .str.title()                          # Capitalize first letter of each word
        .str.replace(' ', '_', regex=False)   # Replace spaces with hyphens
    )

    print(df.columns)

    print("\n Final shape of data:", df.shape)
    return df


# ---------------Exploratory Data Analysis-------------------!
# function to plot categorical features for univariate analysis
def categorical_distributions(df, feature):
    """
    This function will plot the distribution of a categorical feature on a given dataframe.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The desired column from the dataframe
    """

    # plot the distribution
    plt.figure(figsize=(14, 5))
    sns.countplot(x=feature, data=df, palette='deep', order=df[feature].value_counts().index)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# function to plot the distribution of numerical features for univariate analysis
def numerical_distribution(df, numerical_features):
    """
    Plots distribution plots with KDE curves for a list of numerical features in the given dataframe

    Parameters:
        df(pd.DataFrame): the input dataframe containing the numerical features
        numerical_features: list of column names to plot
    """

    # calculate the subplot grid size
    no_of_rows = (len(numerical_features) - 1) // 3 + 1
    no_of_cols = min(3, len(numerical_features))

    # create subplots
    fig, axes = plt.subplots(nrows=no_of_rows, ncols=no_of_cols, figsize=(16, 4 * no_of_rows))
    axes = axes.flatten() if len(numerical_features) > 1 else [axes]

    # plot each numerical feature
    for n, feature in enumerate(numerical_features):
        sns.histplot(df[feature], kde=True, ax=axes[n], color='blue', edgecolor='black')
        axes[n].set_title(f"Distribution of {feature}", fontsize=10)
        axes[n].set_xlabel(feature)
        axes[n].set_ylabel('Count')

    # omit any unused subplots
    for i in range(len(numerical_features), len(axes)):
        fig.delaxes(axes[i])

    # improve layout spacing
    fig.tight_layout()
    # plt.savefig('images/numerical_distribution.jpg')
    plt.show()


# function to plot categorical features vs Churn for bivariate analysis
def categorical_churn(df, feature):
    """
    This function plots the distribution of a categorical feature, with churn as a comparative variable

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The categorical column to investigate
    """

    # plot the distribution
    plt.figure(figsize=(10, 5))
    churn_count = df.groupby(feature)['Churn'].sum().sort_values(ascending=False)
    top_10_categories = churn_count.head(10).index.tolist()
    sns.countplot(x=feature, hue='Churn', data=df, order=top_10_categories)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.show()


# function to plot numerical columns with the churn rate for bivariate analysis.
def kde_plots_with_churn(df, feature, type_of_charge):
    """
    This function plots the distribution of the numerical features based on the churn rate.

    Parameters:
        df(pd.DataFrame): The input dataframe
        feature: The numerical feature to plot
        type_of_charge: the specific charge type(day, evening, night, international)
    """

    # kde plots
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=feature, hue='Churn', fill=True)
    plt.xlabel(f"Total {type_of_charge} Charge")
    plt.ylabel("Density")
    plt.title(f"Churn Distribution by Total {type_of_charge} Charges")
    plt.show()

# function to plot the correlation matrix for feature correlation with target variable
def correlation_heatmap(df):
    """
    This function plots a correlation heatmap that illustrates the correlation between the numerical features and the target(Churn)

    Parameters:
        df(pd.DataFrame): The input dataframe
    """

    # define plot size
    plt.figure(figsize=(14, 14))

    # compute the correlation matrix
    corr_matrix = df.corr(numeric_only=True)

    # create a mask that will hide the upper triangle
    mask = corr_matrix.where(np.tril(np.ones(corr_matrix.shape)).astype(np.bool))

    # plot the heatmap
    sns.heatmap(
        data=mask,
        cmap='viridis',
        annot=True,
        fmt=".1g",
        vmin=-1
    )

    # define the title and display plot
    plt.title('Feature Correlatiom Heatmap', fontsize=16)
    # plt.savefig('images/feature_heatmap.jpg', dpi=300)
    plt.show()


# function to drop features that are highly correlated with others (have a correlation coefficient > 0.9)
def drop_highly_correlated_features(df, threshold: float = 0.9, verbose: bool = True):
    """
    This function drops features that are highly correlated with others, beyond a specified threshold

    Parameters:
    -----------
        df(pd.DataFrame): The input dataframe
        threshold: float, default=0.9: Correlation threshold for dropping features
        verbose: bool, default=True: If True, print the names of the dropped features

    Returns:
    --------
        df: A new dataframe with highly correlated features removed.
    """

    # compute the absolute correlation matrix
    corr_matrix = df.corr(numeric_only=True).abs()

    # mask the upper triangle (including diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    tri_df = corr_matrix.mask(mask)

    # identify features to drop
    to_drop = [col for col in tri_df.columns if any(tri_df[col] > threshold)]

    # print the features that are dropped
    if verbose and to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features (r > {threshold}): {to_drop}")
    elif verbose:
        print("No features exceeded the correlation threshold.")

    # drop the features and return reduced dataframe
    return df.drop(columns=to_drop, axis=1), to_drop


# function to drop outliers(rows where any numerical column has a Z-score > 3)
def remove_outliers_zscore(df, z_threshold=3.0):
    """
    This function returns a new dataframe with rows removed where any numerical column has a Z-score
    exceeding the given threshold

    Parameters:
    -----------
        df(pd.DataFrame): The input dataframe containing both numerical and non-numerical columns.
        z_threshold: float, default=3.0: All rows where |z-score| > z_threshold for any numeric column will be dropped

    Returns:
    --------
        df: A coly of the original DataFrame with outliers removed
    """

    # separate the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Compue the Z-scores for each numeric columns
    # scipy.stats.zscore will return a masked array if there are NaNs
    # we then take the absolute value and compare to threshold
    z_scores = np.abs(stats.zscore(df[numeric_cols], nan_policy='omit'))

    # build a boolean mask: True for rows where all |z| <= threshold
    # if a particular column contained NaN (and zscore was NaN), we treat it as "not an outlier" here.
    # replace NaN z-scores with 0 before comparison
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    mask = (z_scores <= z_threshold).all(axis=1)

    # use the mask to filter the original dataframe
    cleaned_df = df.loc[mask].copy()

    return cleaned_df


# -----------------MODELING----------------------
# function to plot the confusion matrix of each model
def plot_confusion_matrix(y_true, y_pred, class_labels=None, title='Confusion Matrix', figsize=(6, 5), cmap='viridis'):
    """
    This function plots a confusion matrix using a Seaborn heatmap.

    Parameters:
        y_true: array-like of shape (n_samples,) - True labels
        y_pred: array-like of shape (n_samples,) - Predicted labels
        class_labels: list of strings - Names of the target classes
        title: string - Title of the plot
        figsize: tuple - Size of the plot
        cmap: string - Color map for the heatmap
    """

    # define the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # plot the confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap=cmap,
        xticklabels=class_labels if class_labels else 'auto',
        yticklabels=class_labels if class_labels else 'auto'
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


# function for hyperparameter tuning
def param_random_search(model, param_dist, X, y, n_iter=10, cv=5, verbose=True, n_jobs=-1):
    """
    This function performs randomized hyperparameter search with cross-validation.

    Parameters:
    -----------
    model : estimator object
        The classification model to tune.
    param_dist : dict
        Dictionary with parameters names (`str`) as keys and distributions or lists of parameters to try.
    X : pd.DataFrame or np.ndarray
        Training features.
    y : pd.Series or np.ndarray
        Training labels.
    n_iter : int, default=10
        Number of parameter settings sampled.
    cv : int, default=5
        Number of cross-validation folds.
    verbose : bool or int
        Controls the verbosity of the output.
    n_jobs : int, default=-1
        Number of jobs to run in parallel (-1 means use all processors).

    Returns:
    --------
    RandomizedSearchCV object
    """

    # instantiate the RandomizedSearchCV function
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='recall',
        verbose=verbose,
        n_jobs=n_jobs
    )

    # fit the random search with train data
    random_search.fit(X, y)

    # determing the best parameters, score and estimators
    print("Best model hyperparameters:", random_search.best_params_)
    print("Best model accuracy:", random_search.best_score_)
    print("Best model estimators:", random_search.best_estimator_)

    return random_search