import numpy as np
import tkinter as tk
import pandas as pd
from math import ceil
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.feature_selection import SelectFromModel

COLOR_PALETTE = ['#332288', '#117733', '#44AA99', '#88CCEE', '#DDCC77',
                 '#CC6677', '#AA4499', '#882255', '#AA4466', '#DDDDDD']


def get_screen_width() -> int:
    """
    This function retrieves the screen width using a tkinter root window and returns the screen width value.
    """
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()

    return screen_width


def set_font_size() -> dict:
    """
    Sets the font sizes based on the screen width.

    Returns:
        dict: A dictionary containing various font sizes for different elements.
    """
    base_font_size = round(get_screen_width() / 100, 0)
    font_sizes = {
        'font.size': base_font_size * 0.6,
        'axes.titlesize': base_font_size * 0.4,
        'axes.labelsize': base_font_size * 0.6,
        'xtick.labelsize': base_font_size * 0.4,
        'ytick.labelsize': base_font_size * 0.4,
        'legend.fontsize': base_font_size * 0.6,
        'figure.titlesize': base_font_size * 0.6
    }

    return font_sizes


def get_color_palette() -> list:
    """
    A function that returns the color palette.

    Returns:
        list: The color palette.
    """
    return COLOR_PALETTE


def cramers_v(x, y):
    """
    Calculates Cramer's V, a measure of association between two categorical variables.

    Parameters:
        x (array-like): The first categorical variable.
        y (array-like): The second categorical variable.

    Returns:
        float: The Cramer's V value, which ranges from 0 to 1. A value close to 1 indicates a strong association, while a value close to 0 indicates no association.

    Note:
        This function assumes that the input variables are categorical and have at least 2 categories.

    References:
        - Cramer, J. (1949). Measures of association between two categorical variables. Journal of the American Statistical Association, 44(250), 259-268.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)

    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def correlation_ratio(categories, measurements):
    """
    Calculate the correlation ratio between categorical variables and measurements.

    Args:
        categories (array-like): An array-like object containing the categorical variables.
        measurements (array-like): An array-like object containing the measurements.

    Returns:
        float: The correlation ratio, which ranges from 0 to 1. A value close to 1 indicates a strong association, while a value close to 0 indicates no association.

    Note:
        This function assumes that the input variables are categorical and have at least 2 categories.
    """
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(
            n_array,
            np.power(
                np.subtract(
                    y_avg_array,
                    y_total_avg),
                2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def draw_numerical_plots(
        df: pd.DataFrame, cols_list: list, grid_cols: int) -> None:
    """
    Draws histograms for the given DataFrame and columns.

    Args:
        df (DataFrame): Input DataFrame
        cols_list (list): List of columns to plot
        grid_cols (int): Number of columns in the grid

    Returns:
        None
    """
    # Grid size configuration
    cols = grid_cols
    rows = ceil(len(cols_list) / cols)
    fig_width = get_screen_width() / 100

    # Font configuration
    font_sizes = set_font_size()
    plt.rcParams.update(font_sizes)

    # Plotting
    _, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    axes = axes.flatten()

    for i, col in enumerate(cols_list):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        sns.histplot(df[col], color=color, ax=axes[i])
        axes[i].set_title(col)
        axes[i].grid(False)

    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def draw_categorical_plots(
        df: pd.DataFrame, cols_list: list, grid_cols: int) -> None:
    """
    Draws bar plots for the given DataFrame and columns.

    Args:
        df (DataFrame): Input DataFrame
        cols_list (list): List of columns to plot
        grid_cols (int): Number of columns in the grid

    Returns:
        None
    """
    # Grid size configuration
    cols = grid_cols
    rows = ceil(len(cols_list) / cols)
    fig_width = get_screen_width() / 100

    # Font configuration
    font_sizes = set_font_size()
    plt.rcParams.update(font_sizes)

    # Plotting
    _, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    axes = axes.flatten()

    for i, col in enumerate(cols_list):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        sns.countplot(df[col], color=color, ax=axes[i])
        axes[i].set_title(col)
        axes[i].grid(False)

    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def draw_predictor_target_plots(
        df: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Draws two plots to visualize the frequency counts and proportions of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                 figsize=(fig_width, fig_width / 5))

    # Chart 1: Frequencies
    sns.histplot(
        data=df,
        x=predictor,
        hue=target,
        multiple='stack',
        palette=COLOR_PALETTE,
        ax=ax1)
    ax1.set_title(
        f'Frequency Counts of {predictor.title()} by {target.title()}')
    ax1.set_xlabel(f'{predictor.title()}')
    ax1.set_ylabel('Count')

    # Chart 2: Proportions
    sns.histplot(
        data=df,
        x=predictor,
        hue=target,
        multiple='fill',
        discrete=True,
        palette=COLOR_PALETTE,
        ax=ax2)
    ax2.set_title(f'Proportion of {target.title()} by {predictor.title()}')
    ax2.set_xlabel(f'{predictor.title()}')
    ax2.set_ylabel('Proportion')
    plt.show()


def t_test(df: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Perform a t-test to compare the means of two groups based on a predictor variable in a DataFrame.

    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data.
    - predictor: str, the column name of the predictor variable.
    - target: str, the column name of the target variable.

    Returns:
    None
    """
    # Setup two groups
    positive_group = df[df[target] == 1][predictor]
    negative_group = df[df[target] == 0][predictor]

    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(positive_group, negative_group)

    # Print the results
    print(f"T-statistic: {t_statistic:.2f}")
    print(f"P-value: {p_value:.6f}")

    # Mean predictor values for both groups
    print(f"Mean {predictor} of stroke group: {positive_group.mean():.2f}")
    print(f"Mean {predictor} of non-stroke group: {negative_group.mean():.2f}")


def chi_squared_test(df: pd.DataFrame, predictor: str, target: str) -> None:
    """
    Performs a chi-squared test on the given DataFrame and returns the p-value.

    Args:
        df (DataFrame): Input DataFrame
        predictor (str): Name of the variable to perform the chi-squared test on
        target (str): Name of the variable to perform the chi-squared test on

    Returns:
        None
    """
    contingency_table = pd.crosstab(df[predictor], df[target])
    chi2, p, _, _ = chi2_contingency(contingency_table)

    print(f"Chi-squared Statistic: {chi2:.4f}")
    print(f"P-value: {p:.6f}")


def correlation_matrices(df: pd.DataFrame, numeric_cols: list,
                         categorical_cols: list) -> None:
    """
    Generates correlation matrices for numeric and categorical columns in the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        numeric_cols (list): List of column names containing numeric data
        categorical_cols (list): List of column names containing categorical data

    Returns:
        None
    """
    # Configuration
    fig_width = get_screen_width() / 100
    _, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, figsize=(fig_width, fig_width / 4))

    # Numeric correlation matrix - Spearman correlation
    corr_matrix = df[numeric_cols].corr(method='spearman')
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        square=True,
        linewidths=.5,
        ax=ax1)
    ax1.set_title('Spearman\'s Correlation Heatmap')

    # Categorical correlation matrix
    cat_correlations = pd.DataFrame(
        index=categorical_cols,
        columns=categorical_cols)
    for i in categorical_cols:
        for j in categorical_cols:
            cat_correlations.loc[i, j] = cramers_v(df[i], df[j])
    cat_correlations = cat_correlations.apply(pd.to_numeric, errors='coerce')
    mask = np.triu(np.ones_like(cat_correlations, dtype=bool))
    sns.heatmap(
        cat_correlations,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        square=True,
        ax=ax2)
    ax2.set_title('Cramer\'s V Correlation Heatmap')

    # Numeric-categorical correlation matrix
    mixed_correlations = pd.DataFrame(
        index=numeric_cols, columns=categorical_cols)
    for num in numeric_cols:
        for cat in categorical_cols:
            try:
                mixed_correlations.loc[num, cat] = correlation_ratio(
                    df[cat], df[num])
            except BaseException:
                mixed_correlations.loc[num, cat] = np.nan
    mixed_correlations = mixed_correlations.apply(
        pd.to_numeric, errors='coerce')
    sns.heatmap(
        mixed_correlations,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        ax=ax3)
    ax3.set_title('Correlation Ratio Heatmap')

    plt.show()


def visualize_performance(
        y_test: pd.Series, y_pred: pd.Series, y_pred_proba: pd.Series) -> None:
    """
    Visualizes the model performance

    Args:
        y_test (ndarray): True test labels
        y_pred (ndarray): Predicted labels
        y_pred_proba (ndarray): Predicted probabilities

    Returns:
        None
    """
    # 3 subplots
    fig_width = get_screen_width() / 100
    _, ax = plt.subplots(1, 3, figsize=(fig_width, fig_width / 3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    ax[0].set_title('Confusion Matrix')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ax[1].plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], 'k--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax[2].plot(recall, precision)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('Precision-Recall Curve')

    plt.tight_layout()
    plt.show()


def draw_boxplots(df: pd.DataFrame, target: str, predictors: list) -> None:
    """
    Draws boxplots for each predictor variable in the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        target (str): Name of the target variable
        predictors (list): List of predictor variable names

    Returns:
        None
    """
    # Plotting Configuration
    fig_width = get_screen_width() / 100
    cols = 3
    rows = int(len(predictors) / cols)
    _, ax = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    ax = ax.flatten()

    # Draw Boxplots
    for i, col in enumerate(predictors):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        sns.boxplot(x=df[target], y=df[col], ax=ax[i], color=color)
        plt.title(f'Boxplot of {col} by {target.title()}')
        ax[i].set_title(col)
        ax[i].grid(False)

    for j in range(i + 1, rows * cols):
        ax[j].axis('off')

    plt.tight_layout()
    plt.show()


def benchmark_models(pipeline: Pipeline, param_grids: dict, models: dict, X: pd.DataFrame,
                     y: pd.Series, cv: int, baseline_scores: list) -> None:
    """
    Benchmark multiple models using cross-validation and plot performance metrics.

    Parameters:
    - preprocessor: ColumnTransformer object for preprocessing data
    - param_grids: Dictionary of parameter grids for grid search
    - models: Dictionary of models to benchmark
    - X: Input features as a pandas DataFrame
    - y: Target variable as a pandas Series
    - cv: Number of cross-validation folds
    - baseline_scores: List of baseline scores

    Returns:
    - None
    """

    # Performance Metrics
    pos_recall = []
    pos_precision = []
    auc_scores = []

    # Add baseline scores to each of the lists
    pos_recall.append(baseline_scores[0])
    pos_precision.append(baseline_scores[1])
    auc_scores.append(baseline_scores[2])

    # Loop through models
    for model_name, model in models.items():

        # Skip Baseline model
        if model_name == 'Baseline':
            continue

        # Pipeline
        model_step = ('model', model)
        pipeline.steps.append(model_step)
        if 'feature_selector' in pipeline.named_steps:
            pipeline.set_params(feature_selector=SelectFromModel(model))

        # GridSearchCV
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=cv,
            scoring='recall',
            n_jobs=-1)
        grid_search.fit(X, y)

        # Best model
        best_model = grid_search.best_estimator_

        # Cross-Validation with best model
        y_pred = cross_val_predict(best_model, X, y, cv=cv)
        y_pred_proba = cross_val_predict(
            best_model, X, y, cv=cv, method='predict_proba')[:, 1]

        # Performance Metrics
        recall = recall_score(y, y_pred, pos_label=1)
        precision = precision_score(y, y_pred, pos_label=1)
        auc = roc_auc_score(y, y_pred_proba)

        # Append to performance metrics list
        pos_recall.append(recall)
        pos_precision.append(precision)
        auc_scores.append(auc)

        # Print best parameters and score
        if len(param_grids[model_name]) > 0:
            print(
                f"Best parameters for {model_name}: {grid_search.best_params_}")

        del pipeline.steps[-1]

    # Plotting Configuration
    model_names = list(models.keys())
    fig_width = get_screen_width() / 100
    cols = 3
    rows = 1
    _, ax = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    ax = ax.flatten()

    # Recall plot
    ax[0].bar(model_names, pos_recall)
    ax[0].set_title('Positive Recall Scores')
    ax[0].set_xlabel('Models')
    ax[0].set_ylabel('Recall Score')
    ax[0].set_ylim(0, 1)
    ax[0].set_xticklabels(model_names, rotation=45, ha='right')
    for i, v in enumerate(pos_recall):
        ax[0].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # Precision plot
    ax[1].bar(model_names, pos_precision)
    ax[1].set_title('Positive Precision Scores')
    ax[1].set_xlabel('Models')
    ax[1].set_ylabel('Precision Score')
    ax[1].set_ylim(0, 1)
    ax[1].set_xticklabels(model_names, rotation=45, ha='right')
    for i, v in enumerate(pos_precision):
        ax[1].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    # ROC-AUC plot
    ax[2].bar(model_names, auc_scores)
    ax[2].set_title('ROC-AUC Scores')
    ax[2].set_xlabel('Models')
    ax[2].set_ylabel('ROC-AUC Score')
    ax[2].set_ylim(0, 1)
    ax[2].set_xticklabels(model_names, rotation=45, ha='right')
    for i, v in enumerate(auc_scores):
        ax[2].text(i, v, f'{v:.4f}', ha='center', va='bottom')

    plt.tight_layout()


def draw_original_log_distribution(df: pd.DataFrame, feature: str):
    """
    Draws two plots to visualize the distributions of a feature variable and the log-transformed distribution.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature (str): The name of the feature to visualize.

    Returns:
        None
    """

    # Original distribution
    plt.subplot(121)
    df[feature].hist(color=COLOR_PALETTE[0], bins=30, grid=False)
    plt.title(f'Original {feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    # Log-transformed distribution
    plt.subplot(122)
    np.log1p(df[feature]).hist(color=COLOR_PALETTE[1], bins=30, grid=False)
    plt.title(f'Log-transformed {feature} Distribution')
    plt.xlabel(f'Log({feature})')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def bmi_category(bmi):
    """
    Determines the BMI category based on the BMI value provided.

    Parameters:
    bmi (float): The BMI value to categorize.

    Returns:
    str: The category of the provided BMI value ('Underweight', 'Normal', 'Overweight', 'Obese').
    """
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def age_group(age):
    """
    Determines the age group category based on the provided age.

    Parameters:
    age (int): The age to categorize.

    Returns:
    str: The age group category ('Under 18', '18-29', '30-44', '45-59', '60 and above').
    """
    if age < 18:
        return 'Under 18'
    elif 18 <= age < 30:
        return '18-29'
    elif 30 <= age < 45:
        return '30-44'
    elif 45 <= age < 60:
        return '45-59'
    else:
        return '60 and above'


def glucose_category(glucose):
    """
    Determines the glucose category based on the provided glucose value.

    Parameters:
    glucose (float): The blood glucose value to categorize.

    Returns:
    str: The category of the provided glucose value ('Normal', 'Prediabetes', 'Diabetes').
    """
    if glucose < 100:
        return 'Normal'
    elif 100 <= glucose < 126:
        return 'Prediabetes'
    else:
        return 'Diabetes'


def lifestyle_score(row):
    """
    Calculates a lifestyle score based on the provided row information containing 'smoking_status', 'bmi', and 'avg_glucose_level'.

    Parameters:
    row (dict): A dictionary containing keys for 'smoking_status', 'bmi', and 'avg_glucose_level'.

    Returns:
    int: The calculated lifestyle score based on the provided row information.
    """
    score = 0
    if row['smoking_status'] == 'never smoked':
        score += 2
    elif row['smoking_status'] == 'formerly smoked':
        score += 1
    if 18.5 <= row['bmi'] < 25:  # Normal BMI range
        score += 2
    if 70 <= row['avg_glucose_level'] <= 100:  # Normal fasting glucose range
        score += 2
    return score


def work_stress_proxy(row):
    """
    Calculates the stress level based on the attributes of the input row.

    Parameters:
    row (dict): A dictionary containing information about 'work_type', 'hypertension', 'heart_disease', 'bmi', and 'avg_glucose_level'.

    Returns:
    int: The calculated stress level based on the input row attributes.
    """
    stress = 0
    if row['work_type'] in ['Private', 'Self-employed']:
        stress += 1
    if row['hypertension'] == 1 or row['heart_disease'] == 1:
        stress += 1
    if row['bmi'] > 25:  # Overweight or obese
        stress += 1
    if row['avg_glucose_level'] > 100:  # Above normal
        stress += 1
    return stress


def add_new_features(X):
    """
    Adds new features to the input DataFrame by applying various transformations and calculations.

    Parameters:
    X (pandas.DataFrame): The input DataFrame containing the features to be transformed.

    Returns:
    pandas.DataFrame: The input DataFrame with additional features added.

    The function performs the following transformations and calculations:
    1. Grouping:
        - Adds a new column 'bmi_category' by applying the `bmi_category` function to the 'bmi' column.
        - Adds a new column 'age_group' by applying the `age_group` function to the 'age' column.
        - Adds a new column 'glucose_category' by applying the `glucose_category` function to the 'avg_glucose_level' column.
    2. Score calculation:
        - Adds a new column 'lifestyle_score' by applying the `lifestyle_score` function to each row of `X` along the rows axis.
        - Adds a new column 'work_stress_proxy' by applying the `work_stress_proxy` function to each row of `X` along the rows axis.
    3. Categorical to numerical / binary:
        - Creates a dictionary `smoking_risk` mapping the values of the 'smoking_status' column to numerical values.
        - Adds a new column 'smoking_risk' by mapping the values of the 'smoking_status' column using the `smoking_risk` dictionary.
    4. Numerical interactions:
        - Adds a new column 'age_bmi_interaction' by multiplying the 'age' and 'bmi' columns.
        - Adds a new column 'glucose_bmi_interaction' by multiplying the 'avg_glucose_level' and 'bmi' columns.
    5. Log transformation:
        - Adds a new column 'bmi_log' by taking the natural logarithm of the 'bmi' column.
        - Adds a new column 'avg_glucose_level_log' by taking the natural logarithm of the 'avg_glucose_level' column.
    """

    # Grouping
    X['bmi_category'] = X['bmi'].apply(bmi_category)
    X['age_group'] = X['age'].apply(age_group)
    X['glucose_category'] = X['avg_glucose_level'].apply(glucose_category)

    # Score calculation
    X['lifestyle_score'] = X.apply(lifestyle_score, axis=1)
    X['work_stress_proxy'] = X.apply(work_stress_proxy, axis=1)

    # Categorical to numerical / binary
    smoking_risk = {
        'never smoked': 0,
        'formerly smoked': 1,
        'smokes': 2,
        'Unknown': 0}
    X['smoking_risk'] = X['smoking_status'].map(smoking_risk)

    # Numerical interactions
    X['age_bmi_interaction'] = X['age'] * X['bmi']
    X['glucose_bmi_interaction'] = X['avg_glucose_level'] * X['bmi']

    # Log transformation
    X['bmi_log'] = np.log(X['bmi'])
    X['avg_glucose_level_log'] = np.log(X['avg_glucose_level'])

    return X


def remove_columns(X, columns_to_remove):
    """
    Removes specified columns from a pandas DataFrame.

    Parameters:
        X (pandas.DataFrame): The input DataFrame.
        columns_to_remove (list): A list of column names to remove.

    Returns:
        pandas.DataFrame: The input DataFrame with the specified columns removed.
    """
    return X.drop(columns=columns_to_remove, errors='ignore')
