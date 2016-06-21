from sklearn.feature_extraction import DictVectorizer
import pandas

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

# Code to convert a dataframe with some columns we want converted to categorical data into format that sci-kit learn plays nicely with

# 


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer


# Goals: 
# - Want to convert some categorical from a dataframe to a format that sklearn
# - Want to take advantage of sparse representations for the categorical transformations (especially when we have lots of dummmy values)
# - Want to standardize numeric variables (transform data such that columns have mean 0 and variance 1). 


# This function gives a rough start towards this goal. Things could be made better but that is for a future date. 

def convert_df_to_X_data(df, categorical_variables, numeric_variables):

    class DataFrame_Col_Selector(BaseEstimator, TransformerMixin):
        def __init__(self, cols):
            self.cols = cols
        def fit(self,x,y=None):
            return self
        def transform(self, df):
            return df[self.cols]

    class DataFrame_Categorical_Converter(BaseEstimator, TransformerMixin):
        def fit(self, x,y=None):
            return self
        def transform(self, df):
            series_list = []
            for c in df.columns:
                temp_series = df[c].astype(str)
                temp_series.name = c + '_str'
                series_list.append(temp_series)
            temp_data = pandas.concat(series_list, axis=1)
            # Convert data to dict to pass to sklearn DictVectorizer
            dict_data = temp_data.to_dict(orient='records')
            return dict_data
        
 
        
    pipeline = FeatureUnion([
            ('categorical_pipeline',Pipeline([
                ('col_selector', DataFrame_Col_Selector(categorical_variables)),
                ('dict_converter', DataFrame_Categorical_Converter()),
                ('dict_vectorizer', DictVectorizer())
            ])),
            ('numeric_pipeline',Pipeline([
                        ('col_selector', DataFrame_Col_Selector(numeric_variables)),
                        ('standard_scaler',StandardScaler(with_mean=True))
                    ]))
        ])


    return pipeline.fit_transform(df), pipeline



def convert_df_to_X_data(data_frame, categorical_predictor_variables, dv=None):
    
    # First limit to only variables we want to convert
    temp_data = data_frame[categorical_predictor_variables]
    
    # Convert all columns to strings 
    temp_data = temp_data.apply(lambda x: x.astype(str), axis=0)


    # Convert data to dict to pass to sklearn DictVectorizer
    dict_data = temp_data.to_dict(orient='records')
    
    if dv is None:
        dv = DictVectorizer()
        converted_data = dv.fit_transform(dict_data)
    else:
        converted_data = dv.fit(dict_data)
    return converted_data, dv