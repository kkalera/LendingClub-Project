import logging
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    Normalizer,
    FunctionTransformer,
    MinMaxScaler,
)
from sklearn.impute import KNNImputer, SimpleImputer
import shap
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score,
)
from xgboost import XGBClassifier, XGBRegressor
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split

COLOR_MAIN = "#69b3a2"
COLOR_CONTRAST = "#B3697A"

DEPLOYMENT_MODELS= [
    {
        "model_path":"src/models/p1_best_lr_model.joblib",
        "sample_path": "src/data/p1_sample.json",
        "labels": ["Declined", "Approved"]},
    {
        "model_path":"src/models/p1_best_xgb_model.joblib",
        "sample_path": "src/data/p1_sample.json",
        "labels": ["Declined", "Approved"]},
    {
        "model_path":"src/models/p2_best_xgb_model.joblib",
        "sample_path": "src/data/p2_sample.json",
        "labels": ["A", "B", "C", "D", "E", "F", "G"]},
    {
        "model_path":"src/models/p3_best_sub_xgb_model.joblib",
        "sample_path": "src/data/p3_sub_sample.json",
        "labels": ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4", "C5",
                    "D1", "D2", "D3", "D4", "D5", "E1", "E2", "E3", "E4", "E5", "F1", "F2", "F3", "F4", "F5",
                    "G1", "G2", "G3", "G4", "G5"]},
    {
        "model_path":"src/models/p3_best_int_xgb_model.joblib",
        "sample_path": "src/data/p3_int_sample.json",
        "labels": None},
]

def color_palette_husl(n_colors):
    return sns.color_palette("husl", n_colors=n_colors)


def get_logger():
    """
    Get a logger instance with configured handlers and formatters.

    Returns:
        logging.Logger: The logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler("src/logs/my.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def check_employment_length(row):
    """
    Check if the employment duration is less than the loan duration.

    Args:
        row (dict): A dictionary containing the loan information.

    Returns:
        None
    """
    from datetime import datetime
    from datetime import timedelta
    import numpy as np

    if (
        type(row["emp_length"]) is float
        or type(row["last_pymnt_d"]) is float
        or type(row["issue_d"]) is float
    ):
        return

    last_pymnt_d = datetime.strptime(row["last_pymnt_d"], "%b-%Y")
    issue_d = datetime.strptime(row["issue_d"], "%b-%Y")
    employment_duration = int("".join(filter(str.isdigit, row["emp_length"])))
    if employment_duration * 365.24 < (last_pymnt_d - issue_d).days:
        print(
            f"Employment duration of {row['emp_length']} is less than the loan duration"
            f" of {(last_pymnt_d - issue_d).days} days"
        )


def get_cmap():
    """
    Returns a matplotlib colormap with a main color and a contrast color.

    Returns:
    matplotlib.colors.LinearSegmentedColormap: The matplotlib colormap.
    """
    norm = matplotlib.colors.Normalize(-1, 1)
    colors = [
        [norm(-1.0), COLOR_CONTRAST],
        [norm(0.0), "#ffffff"],
        [norm(1.0), COLOR_MAIN],
    ]
    return matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


def countplot(
    data,
    column_name: str,
    title: str = "Countplot",
    hue: str = None,
    ax=None,
    figsize=(10, 5),
    bar_labels: bool = False,
    bar_label_kind: str = "percentage",
    horizontal: bool = False,
):
    """
    Generate a countplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the countplot. Defaults to "Countplot".
        hue (str, optional): The column name to use for grouping the countplot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created. Defaults to None.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).
        bar_labels (bool, optional): Whether to add labels to the bars. Defaults to False.
        bar_label_kind (str, optional): The kind of labels to add to the bars. Can be "percentage" or "count". Defaults to "percentage".

    Returns:
        matplotlib.axes.Axes: The axis object containing the countplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=(10, 5)) if ax is None else (plt.gcf(), ax)

    if hue:
        if horizontal:
            sns.countplot(
                data=data,
                y=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=[COLOR_CONTRAST, COLOR_MAIN],
                hue=hue,
            )
        else:
            sns.countplot(
                data=data,
                x=column_name,
                ax=ax,
                color=COLOR_MAIN,
                palette=[COLOR_CONTRAST, COLOR_MAIN],
                hue=hue,
            )
    else:
        if horizontal:
            sns.countplot(data=data, y=column_name, ax=ax, color=COLOR_MAIN)
        else:
            sns.countplot(data=data, x=column_name, ax=ax, color=COLOR_MAIN)

    ## Add bar labels
    if bar_labels:
        for container in ax.containers:
            if bar_label_kind == "percentage":
                ax.bar_label(container, fmt=lambda x: f" {x / len(data):.1%}")
            else:
                ax.bar_label(container, fmt=lambda x: f" {x}")

    ## Add title
    ax.set_title(label=title, fontsize=16)
    return ax


def boxplot(
    data,
    column_name: str,
    title: str = "Boxplot",
    ax=None,
    figsize=(10, 5),
    y_lim: tuple = None,
    hue: str = None,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
):
    """
    Create a boxplot for a given column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to create the boxplot for.
        title (str, optional): The title of the boxplot. Defaults to "Boxplot".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the boxplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue is None:
        sns.boxplot(
            data=data,
            y=column_name,
            ax=ax,
            color=COLOR_MAIN,
        )
    else:
        sns.boxplot(
            data=data,
            y=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
        )
    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def histplot(
    data,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 5),
    kde: bool = False,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
    y_lim: tuple = None,
    bins="auto",
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.histplot(
            data=data,
            x=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
            kde=kde,
            bins=bins,
        )
    else:
        sns.histplot(
            data=data, x=column_name, ax=ax, color=COLOR_MAIN, kde=kde, bins=bins
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def kdeplot(
    data,
    column_name: str,
    hue: str = None,
    title: str = "Histogram",
    ax=None,
    figsize=(10, 4),
    palette=[COLOR_MAIN, COLOR_CONTRAST],
    y_lim: tuple = None,
    bw_adjust: float = 1,
):
    """
    Plot a histogram of a specified column in a pandas DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to plot.
        title (str, optional): The title of the histogram. Defaults to "Histogram".
        ax (matplotlib.axes.Axes, optional): The axis to plot on. If not provided, a new axis will be created.
        figsize (tuple, optional): The size of the figure. Defaults to (10, 5).

    Returns:
        matplotlib.axes.Axes: The axis object containing the histogram plot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    ## Create plot
    if hue:
        sns.kdeplot(
            data=data,
            x=column_name,
            ax=ax,
            palette=palette,
            hue=hue,
            warn_singular=False,
            bw_adjust=bw_adjust,
        )
    else:
        sns.kdeplot(
            data=data,
            x=column_name,
            ax=ax,
            color=COLOR_MAIN,
            bw_adjust=bw_adjust,
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    # Set Y axis limit
    if y_lim:
        ax.set_ylim(y_lim)

    return ax


def plot_distribution_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    bins="auto",
    hue=None,
    kde=False,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    if hue is None:
        histplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bins=bins,
            kde=kde,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
        )
    else:
        histplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bins=bins,
            hue=hue,
            kde=kde,
            palette=palette,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            hue=hue,
            palette=palette,
        )
    fig.suptitle(title, fontsize=16)
    return fig


def plot_kde_and_box(
    data,
    column_name: str,
    title: str = "Count and Boxplot",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    hue=None,
    palette=[COLOR_MAIN, COLOR_CONTRAST],
    bw_adjust: float = 1,
):
    """
    Plots the distribution and boxplot of a numerical column in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the numerical column to plot.
        title (str, optional): The title of the plot. Defaults to "Count and Boxplot".
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        figsize (tuple, optional): The figure size. Defaults to (10, 5).
        width_ratios (list, optional): The width ratios of the subplots. Defaults to [3, 1.25].
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(column_name, str)
    assert isinstance(title, str)
    # assert column_name in data.select_dtypes(include=np.number).columns

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(
        figsize=figsize, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    if hue is None:
        kdeplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            bw_adjust=bw_adjust,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
        )
    else:
        kdeplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[0],
            hue=hue,
            palette=palette,
            bw_adjust=bw_adjust,
        )
        boxplot(
            data=data,
            column_name=column_name,
            title="",
            ax=ax[1],
            hue=hue,
            palette=palette,
        )
    fig.suptitle(title, fontsize=16)
    return fig


def plot_distribution_and_ratio(
    data,
    ratio: pd.Series,
    column_name: str,
    hue: str,
    title: str = "Distribution and Ratio",
    ax=None,
    figsize=(10, 5),
    width_ratios=[3, 1.25],
    horizontal: bool = False,
    label_rotation: int = 0,
):
    """
    Plot the distribution and ratio of a categorical variable.

    Parameters:
    - data: The DataFrame containing the data.
    - ratio: The ratio of the categories.
    - column_name: The name of the categorical column.
    - hue: The column to use for grouping the data.
    - title: The title of the plot (default: "Distribution and Ratio").
    - ax: The matplotlib axes object to plot on (default: None).
    - figsize: The figure size (default: (10, 5)).
    - width_ratios: The width ratios of the subplots (default: [3, 1.25]).
    - horizontal: Whether to plot the bars horizontally (default: False).
    - label_rotation: The rotation angle of the tick labels (default: 0).
    """
    fig, ax = plt.subplots(
        figsize=figsize, nrows=1, ncols=2, gridspec_kw={"width_ratios": width_ratios}
    )
    countplot(
        data=data,
        column_name=column_name,
        hue=hue,
        title="Distribution",
        bar_labels=True,
        ax=ax.flatten()[0],
        horizontal=horizontal,
    )
    if horizontal:
        sns.barplot(
            y=ratio.index,
            x=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    else:
        sns.barplot(
            x=ratio.index,
            y=ratio.values,
            color=COLOR_MAIN,
            ax=ax.flatten()[1],
        )
    ax[1].set_title("Ratio")

    if label_rotation:
        if horizontal:
            for t1, t2 in zip(ax[0].get_yticklabels(), ax[1].get_yticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)
        else:
            for t1, t2 in zip(ax[0].get_xticklabels(), ax[1].get_xticklabels()):
                t1.set_rotation(45)
                t2.set_rotation(45)


def correlation_matrix(corr, title="Correlation Matrix"):
    """
    Plot a correlation matrix heatmap.

    Parameters:
    corr (numpy.ndarray): The correlation matrix.
    title (str): The title of the plot. Default is "Correlation Matrix".

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    cmap = get_cmap()

    sns.heatmap(corr, mask=mask, annot=True, cmap=cmap, vmin=-1, vmax=1, fmt=".2f")
    fig.suptitle(title, fontsize=16)


def calculate_outlier_range(data):
    """
    Calculate the lower and upper bounds for identifying outliers using the IQR method.

    Parameters:
    data (pandas.Series): The data for which to calculate the outlier range.

    Returns:
    tuple: A tuple containing the lower and upper bounds for identifying outliers.
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return (lower_bound, upper_bound)


def scatterplot(
    data,
    x,
    y,
    title="Scatterplot",
    ax=None,
    figsize=(5, 10),
    hue=None,
    size=None,
    sizes=None,
    palette=[COLOR_CONTRAST, COLOR_MAIN],
):
    """
    Plot a scatterplot of two numerical columns in a DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    x (str): The name of the column to plot on the x-axis.
    y (str): The name of the column to plot on the y-axis.
    title (str): The title of the plot. Default is "Scatterplot".
    ax (matplotlib.axes.Axes): The axis to plot on. If not provided, a new axis will be created.
    figsize (tuple): The size of the figure. Default is (10, 5).

    Returns:
    matplotlib.axes.Axes: The axis object containing the scatterplot.
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(x, str)
    assert isinstance(y, str)
    assert isinstance(title, str)

    sns.set_style("whitegrid")

    ## Create axis if not provided
    fig, ax = plt.subplots(1, 1, figsize=figsize) if ax is None else (plt.gcf(), ax)

    if hue is None:
        sns.scatterplot(data=data, x=x, y=y, ax=ax, color=COLOR_MAIN)
    else:
        sns.scatterplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            hue=hue,
            palette=palette,
            size=size,
            sizes=sizes,
        )

    ## Add title
    ax.set_title(label=title, fontsize=16)

    return ax


def get_correlations(data: pd.DataFrame):
    """
    Calculate the correlation matrix for the given DataFrame.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    pd.DataFrame: The correlation matrix of the input DataFrame.
    """
    df_corr = data.copy()
    categorical_columns = df_corr.select_dtypes(include=["category", "object"]).columns

    df_dummies = pd.get_dummies(df_corr[categorical_columns])
    df_corr = pd.concat([df_corr, df_dummies], axis=1)
    df_corr.drop(
        columns=categorical_columns,
        axis=1,
        inplace=True,
    )
    correlations = df_corr.corr()
    correlations.drop(columns=df_dummies.columns, inplace=True, axis=0)
    return correlations


def remove_emp_length_samples(data):
    import random

    random.seed(42)
    df = data.copy()
    df["emp_length"] = df["emp_length"].apply(lambda x: x.replace(" ", ""))
    df.reset_index(inplace=True)

    for value in df["emp_length"].unique():
        df_val = df.loc[df["emp_length"] == value]
        df_val_approved = df_val.loc[df_val["approved"] == True]
        df_val_declined = df_val.loc[df_val["approved"] == False]
        amount_to_drop = len(df_val_declined) - len(df_val_approved)

        if amount_to_drop > 0:
            drop_indices = df_val_declined.sample(n=amount_to_drop, random_state=42).index
            df.drop(index=drop_indices, inplace=True)
    df.drop(columns=["index"], inplace=True)
    return df


def part_1_plot_processed_comparison(original_data, processed_data):
    """
    Plot a comparison of the processed data with the original data.

    Parameters:
    - original_data (DataFrame): The original data.
    - processed_data (DataFrame): The processed data.

    Returns:
    - ax (Axes): The matplotlib axes object containing the plots.
    """
    fig, ax = plt.subplots(
        figsize=(10, 10),
        nrows=2,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 1.25], "wspace": 0.3, "hspace": 0.3},
    )
    countplot(
        original_data,
        "emp_length",
        title="Employment Length Before Processing",
        hue="approved",
        bar_labels=True,
        horizontal=True,
        bar_label_kind=None,
        ax=ax.flatten()[0],
    )

    countplot(
        original_data,
        "approved",
        title="Approved Before Processing",
        bar_labels=True,
        ax=ax.flatten()[1],
    )
    countplot(
        processed_data,
        "emp_length",
        title="Employment Length After Processing",
        hue="approved",
        bar_labels=True,
        horizontal=True,
        bar_label_kind=None,
        ax=ax.flatten()[2],
    )
    countplot(
        processed_data,
        "approved",
        title="Approved After Processing",
        bar_labels=True,
        ax=ax.flatten()[3],
    )

    ax.flatten()[0].set_xlim(0, ax.flatten()[0].get_xlim()[1] * 1.15)
    ax.flatten()[2].set_xlim(0, ax.flatten()[2].get_xlim()[1] * 1.15)

    return ax


def part_1_preprocess(data):
    """
    Preprocesses the given data by removing rows with dti > 100 and risk_score > 850 or < 600.

    Args:
        data (pandas.DataFrame): The input data.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    df = data.copy()
    df.drop(df.loc[df["dti"] > 100].index, inplace=True)
    df.drop(
        df.loc[((df["risk_score"] > 850) | (df["risk_score"] < 600))].index,
        inplace=True,
    )
    df = remove_emp_length_samples(df)
    df["emp_length"] = df["emp_length"].str.replace("<", "less_than_")
    return df


def part_2_preprocess(data):
    df = data.copy()
    df.drop(
        columns=[
            "last_fico_range_low",
            "last_fico_range_high",
            "sec_app_fico_range_high",
            "fico_range_high",
        ],
        axis=1,
        inplace=True,
    )

    df.dropna(subset=["grade_le"], inplace=True)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(-1)

    return df


def get_pipeline(
    model, numerical_columns=None, categorical_columns=None, binary_columns=None
):
    """
    Create a data preprocessing pipeline for a given model.

    Parameters:
    model (object): The machine learning model to be used in the pipeline.

    Returns:
    pipeline (object): The data preprocessing pipeline.

    """
    transformers = []

    # Do binary encoding for the binary categories
    if binary_columns is not None:
        binary_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("binary", OrdinalEncoder()),
            ]
        )
        transformers.append(("binary", binary_transformer, binary_columns))

    # Do one-hot encoding for the categorical categories
    if categorical_columns is not None:
        categorical_transformer = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="error")),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    # Do standard scaling for the numerical columns
    if numerical_columns is not None:
        numerical_transformer = Pipeline(
            steps=[
                ("impute", KNNImputer()),
                ("scale", StandardScaler()),
                ("normalize", Normalizer()),
            ]
        )
        transformers.append(("numerical", numerical_transformer, numerical_columns))

    # Create the preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
    )

    # Create the pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def get_lr_model(trial) -> LogisticRegression:
    """
    Based upon: https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451

    Instantiate a LogisticRegression model with the given hyperparameters.

    Parameters:
    - trial (optuna.trial): The optuna trial object containing the hyperparameters.

    Returns:
    - model (LogisticRegression): The instantiated LogisticRegression model.
    """
    model = LogisticRegression(
        penalty="l2",
        C=trial.suggest_float("C", 1e-10, 1e10, log=True),
        solver=trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        ),
        max_iter=trial.suggest_int("max_iter", 100, 1000),
        random_state=42,
    )
    return model


def get_xgb_classifier(trial, objective="binary:logistic") -> XGBClassifier:
    """
    Based upon: https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451

    Instantiate a XGBClassifier model with the given hyperparameters.

    Parameters:
    - trial (optuna.trial): The optuna trial object containing the hyperparameters.

    Returns:
    - model (XGBClassifier): The instantiated XGBClassifier model.
    """
    # Parameters found at : https://www.kaggle.com/code/kst6690/make-your-xgboost-model-awesome-with-optuna

    metric_list = ["logloss", "auc", "error"]
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
        "objective": objective,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.01),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1, step=0.01),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1, step=0.01),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
    }
    return XGBClassifier(**params)


def get_xgb_regressor(trial) -> XGBRegressor:
    """
    Based upon: https://medium.com/@walter_sperat/using-optuna-with-sklearn-the-right-way-part-1-6b4ad0ab2451

    Instantiate a XGBRegressor model with the given hyperparameters.

    Parameters:
    - trial (optuna.trial): The optuna trial object containing the hyperparameters.

    Returns:
    - model (XGBRegressor): The instantiated XGBRegressor model.
    """
    # Parameters found at : https://www.kaggle.com/code/kst6690/make-your-xgboost-model-awesome-with-optuna

    metric_list = ["logloss", "auc", "error"]
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "reg_alpha": trial.suggest_int("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_int("reg_lambda", 0, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 5),
        "gamma": trial.suggest_int("gamma", 0, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.5),
        "eval_metric": trial.suggest_categorical("eval_metric", metric_list),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1, step=0.01),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.1, 1, step=0.01),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1, step=0.01),
        "subsample": trial.suggest_float("subsample", 0.5, 1),
    }
    return XGBRegressor(**params)


def test_model(pred_train, pred_test, y_train, y_test, logger):
    """
    Evaluate the performance of a machine learning model on training and test data.

    Args:
        model: The trained machine learning model.
        x_train: The input features of the training data.
        x_test: The input features of the test data.
        y_train: The target labels of the training data.
        y_test: The target labels of the test data.
        logger: The logger object for logging the evaluation results.

    Returns:
        None
    """
    acc = accuracy_score(y_train, pred_train)
    prec = precision_score(y_train, pred_train, average="weighted", zero_division=0)
    rec = recall_score(y_train, pred_train, average="weighted")

    logger.info(f"Training scores:\n")
    logger.info(f"    - Accuracy: {acc:.3f}")
    logger.info(f"    - Precision: {prec:.3f}")
    logger.info(f"    - Recall: {rec:.3f}")

    acc = accuracy_score(y_test, pred_test)
    prec = precision_score(y_test, pred_test, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred_test, average="weighted")

    logger.info(f"Test scores:\n")
    logger.info(f"    - Accuracy: {acc:.3f}")
    logger.info(f"    - Precision: {prec:.3f}")
    logger.info(f"    - Recall: {rec:.3f}")


def test_regression_model(
    pred_train, pred_test, y_train, y_test, logger
):
    """
    Evaluate the performance of a regression machine learning model on training and test data.

    Args:
        model: The trained regression machine learning model.
        x_train: The input features of the training data.
        x_test: The input features of the test data.
        y_train: The target labels of the training data.
        y_test: The target labels of the test data.
        logger: The logger object for logging the evaluation results.

    Returns:
        None
    """
    mape = mean_absolute_percentage_error(y_train, pred_train)
    evs = explained_variance_score(y_train, pred_train)
    r2 = r2_score(y_train, pred_train)

    logger.info(f"Training scores:\n")
    logger.info(f"    - Mean Absolute Percentage Error: {mape:.3f}")
    logger.info(f"    - Explained Variance: {evs:.3f}")
    logger.info(f"    - R2 Score: {r2:.3f}\n")

    mape = mean_absolute_percentage_error(y_train, pred_train)
    evs = explained_variance_score(y_train, pred_train)
    r2 = r2_score(y_train, pred_train)

    logger.info(f"Test scores:\n")
    logger.info(f"    - Mean Absolute Percentage Error: {mape:.3f}")
    logger.info(f"    - Explained Variance: {evs:.3f}")
    logger.info(f"    - R2 Score: {r2:.3f}\n")


def plot_roc_and_confusion_matrix(model, X_test, y_test):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and the confusion matrix for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fig, ax = plt.subplots(
        figsize=(10, 5),
        nrows=1,
        ncols=2,
        gridspec_kw={"width_ratios": [3, 2], "wspace": 0.3},
    )
    plt.grid(False)
    plot_roc_curve(model, X_test, y_test, ax=ax[0])
    cm = confusion_matrix(y_test, model.predict(X_test), normalize="all")
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Declined", "Approved"]
    ).plot(include_values=True, cmap="Blues", ax=ax.flatten()[1])


def plot_roc_curve(model, x_test, y_test, ax=None):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    ax = plt.figure(figsize=(8, 6)) if ax is None else plt.sca(ax)
    plt.plot(fpr, tpr, color=COLOR_MAIN, lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color=COLOR_CONTRAST, lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")


def plot_threshold_score_curve(
    model, x_test, y_test, score_function, figsize=(10, 5), ax=None
):
    """
    Plots the score vs. threshold curve for a given model.

    Args:
        model: The trained model.
        x_test: The test data.
        y_test: The true labels for the test data.
        score_function: The scoring function to evaluate the predictions.
        figsize: The size of the figure (default: (10, 5)).
        ax: The matplotlib axes object to plot on (default: None).

    Returns:
        None
    """
    probabilities = model.predict_proba(x_test)[:, 1]
    thresholds = np.arange(0.0, 1.0, 0.01)
    scores = [
        score_function(y_test, probabilities > threshold) for threshold in thresholds
    ]

    idx = np.argmax(scores)
    best_threshold = thresholds[idx]

    fig, ax = plt.figure(figsize=figsize) if ax is None else plt.gcf(), ax
    sns.lineplot(x=thresholds, y=scores, color=COLOR_MAIN)
    plt.scatter(best_threshold, scores[idx], color=COLOR_CONTRAST)
    plt.text(
        best_threshold * 1.01,
        scores[idx] * 1.01,
        f"Best threshold: {best_threshold:.2f}",
    )
    plt.suptitle("Score vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.plot()


def predict_with_custom_threshold(model, data, threshold=0.5):
    """
    Predicts labels based on a custom threshold.

    :param model: The trained XGBoost model
    :param data: The input data for prediction
    :param threshold: The custom threshold for classification
    :return: Predicted labels
    """
    # Predict probabilities
    probabilities = model.predict_proba(data)[:, 1]

    # Apply custom threshold
    predictions = (probabilities >= threshold).astype(int)

    return predictions


def plot_confusion_matrix(pred, y_test, display_labels=None, ax=None):
    """
    Plots the confusion matrix for a given model.

    Parameters:
    - model: The trained model for which the ROC curve is plotted.
    - x_test: The input features for the test set.
    - y_test: The true labels for the test set.

    Returns:
    None
    """
    cm = confusion_matrix(y_test, pred, normalize="all")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels).plot(
        include_values=True, cmap="Blues"
    )
    plt.grid(False)


def plot_multiclass_roc(
    model,
    x_test,
    y_test,
    n_classes,
    class_labels=None,
    figsize=(10, 5),
    ax=None,
    legend_params={"loc": "lower right"},
):
    # Convert y_test to a NumPy array and reshape if it's a Pandas Series
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy().reshape(-1, 1)

    # One-hot encode y_test if it's not already in that format
    if y_test.ndim == 1 or y_test.shape[1] == 1:
        encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
        y_test = encoder.fit_transform(y_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Calculate ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], model.predict_proba(x_test)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves for each class
    plt.figure(figsize=figsize) if ax is None else plt.sca(ax)
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=2,
            label=(
                "ROC curve of class"
                f" {class_labels[i] if class_labels is not None else i } (area ="
                f" {roc_auc[i]:0.3f})"
            ),
        )

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(**legend_params)
    plt.show()


def plot_xgb_shap(
    model, x, feature_names, figsize=(10, 5), multi_class=False, classes_to_plot=None
):
    if issparse(x):
        x = pd.DataFrame.sparse.from_spmatrix(x, columns=feature_names)
    else:
        x = pd.DataFrame(x, columns=feature_names)

    explainer = shap.Explainer(model)
    shap_values = explainer(x)

    if not multi_class:
        shap.plots.beeswarm(shap_values, plot_size=figsize)
    else:
        for c in classes_to_plot:
            print(f"Plotting SHAP values for class {c}")
            shap.summary_plot(shap_values[:, :, c], x, plot_size=figsize)


def train_test_valid_split(
    data,
    label_column,
    stratify: str = None,
    test_size=0.2,
    valid_size=0.2,
    random_state=42,
):
    """
    Splits the data into training, validation and test sets.

    Parameters:
    - data: The input data.
    - label_column: The name of the label column.
    - stratify: The name of the column to use for stratification.
    - test_size: The size of the test set.
    - valid_size: The size of the validation set.
    - random_state: The random state for reproducibility.

    Returns:
    - train: The training set.
    - valid: The validation set.
    - test: The test set.
    """
    # Split into train and test set
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[stratify] if stratify else None,
    )

    # Split train set into train and validation set
    train, valid = train_test_split(
        train,
        test_size=valid_size,
        random_state=random_state,
        stratify=train[stratify] if stratify else None,
    )

    # Split into input features and target labels
    x_train = train.drop(columns=label_column)
    y_train = train[label_column]
    x_valid = valid.drop(columns=label_column)
    y_valid = valid[label_column]
    x_test = test.drop(columns=label_column)
    y_test = test[label_column]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "y_test": y_test,
    }


def xgb_classification_objective(trial, x_train, y_train, x_valid, y_valid):
    model = get_xgb_classifier(trial)
    pipeline = get_pipeline(
        model,
        numerical_columns=x_train.select_dtypes(include=np.number).columns,
    )
    pipeline.fit(x_train, y_train)
    return roc_auc_score(y_valid, pipeline.predict_proba(x_valid), multi_class="ovr")


def xgb_regression_objective(trial, x_train, y_train, x_valid, y_valid):
    model = get_xgb_regressor(trial)
    pipeline = get_pipeline(
        model,
        numerical_columns=x_train.select_dtypes(include=np.number).columns,
    )
    pipeline.fit(x_train, y_train)
    return mean_absolute_percentage_error(y_valid, pipeline.predict(x_valid))


def plot_error_distribution(
    y_true, y_pred, title="Absolute Error Distribution", bins=10, ax=None
):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["abs_error"] = abs(df["y_true"] - df["y_pred"])
    df["bin"] = pd.cut(df["y_true"], bins=bins)
    df["bin_mid"] = [(x.right + x.left) / 2 for x in df["bin"]]

    fig, ax = plt.subplots(figsize=(15, 7.5)) if ax is None else (plt.gcf(), ax)

    sns.lineplot(df.groupby("bin_mid")["abs_error"].mean(), ax=ax, linewidth=2)
    ax.set_title(title)
