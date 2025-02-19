import pandas as pd
from dython.nominal import correlation_ratio, cramers_v 
 
def cramers_v_corr(pd_df, columns):
    cramers_v_table = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                cramers_v_table.loc[col1, col2] = 1.0  # Perfect correlation with itself
            else:
                cramers_v_table.loc[col1, col2] = cramers_v(pd_df[col1], pd_df[col2])

    return cramers_v_table.astype(float)

def correlation_ratio_corr(pd_df, num_cols, cat_cols):

    cr_table = pd.DataFrame(index=cat_cols, columns=num_cols)

    for col1 in cat_cols:
        for col2 in num_cols:
            cr_table.loc[col1, col2] = correlation_ratio(pd_df[col1], pd_df[col2])

    return cr_table.astype("float")

