import pandas as pd
from dython.nominal import correlation_ratio, cramers_v 
 
def cramers_v_corr(pd_df, columns):
    """
    Calculates the Cramér's V correlation for all combinations of categorical variables in a given dataset.

    Cramér's V measures the association between two categorical variables and ranges from 0 (no association) to 1 (perfect association).

    Parameters:
    -----------
    pd_df : pd.DataFrame
        The Pandas DataFrame containing the categorical variables.
    columns : list
        A list of categorical column names for which Cramér's V correlation will be computed.

    Returns:
    --------
    pd.DataFrame
        A correlation matrix where each entry represents the Cramér's V value between two categorical variables.
        The diagonal values are set to 1.0 (perfect association with itself).
    """
    cramers_v_table = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                cramers_v_table.loc[col1, col2] = 1.0  # Perfect correlation with itself
            else:
                cramers_v_table.loc[col1, col2] = cramers_v(pd_df[col1], pd_df[col2])

    return cramers_v_table.astype(float)

def correlation_ratio_corr(pd_df, num_cols, cat_cols):
    """
    Calculates the Correlation Ratio for categorical vs numerical variable relationships in a dataset.

    The Correlation Ratio (η²) measures the strength of the relationship between a categorical variable and a numerical variable.
    It ranges from 0 (no correlation) to 1 (strong correlation).

    Parameters:
    -----------
    pd_df : pd.DataFrame
        The Pandas DataFrame containing the data.
    num_cols : list
        A list of numerical column names.
    cat_cols : list
        A list of categorical column names.

    Returns:
    --------
    pd.DataFrame
        A table where each entry represents the Correlation Ratio (η²) between a categorical and a numerical variable.
    """
    cr_table = pd.DataFrame(index=cat_cols, columns=num_cols)

    for col1 in cat_cols:
        for col2 in num_cols:
            cr_table.loc[col1, col2] = correlation_ratio(pd_df[col1], pd_df[col2])

    return cr_table.astype("float")

