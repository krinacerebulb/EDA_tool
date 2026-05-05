"""Chart builders. Each function returns a matplotlib Figure so the caller
can either pass it to st.pyplot or embed it in an HTML report."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
    """Histogram + KDE for a numeric column."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[column].dropna(), bins=bins, kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return fig


def boxplot(df: pd.DataFrame, column: str) -> plt.Figure:
    """Boxplot for a single numeric column to visualise outliers."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.boxplot(x=df[column].dropna(), ax=ax, color="#55A868")
    ax.set_title(f"Boxplot of {column}")
    ax.set_xlabel(column)
    return fig


def bar_chart(df: pd.DataFrame, column: str, top_n: int = 10) -> plt.Figure:
    """Bar chart of the top-N most frequent values in a categorical column."""
    counts = df[column].dropna().astype(str).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#C44E52")
    ax.set_title(f"Top {len(counts)} values in {column}")
    ax.set_xlabel("Count")
    ax.set_ylabel(column)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure | None:
    """Correlation heatmap of numeric columns. Returns None if <2 numeric cols."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return None

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr) * 0.6), max(5, len(corr) * 0.5)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Heatmap")
    return fig
