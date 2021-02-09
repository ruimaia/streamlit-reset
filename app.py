import pandas as pd
import streamlit as st 
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
import itertools
from datetime import datetime
import SessionState


def fetch_data(uploaded_file):
    """Reads csv file to pandas DataFrame"""

    file_df=pd.read_csv(uploaded_file, sep=',')
    
    return file_df


def create_param_grid(growth, daily_seasonality, weekly_seasonality, yearly_seasonality):
    """Create hyperparameter grid for cross validation""" 
    
    param_grid = {  
                    'growth': [growth],
                    'daily_seasonality': [daily_seasonality],
                    'weekly_seasonality': [weekly_seasonality],
                    'yearly_seasonality': [yearly_seasonality],
                    'changepoint_prior_scale': [0.5, 1, 5, 10],
                    'seasonality_prior_scale': [5, 10, 15, 20],
                }
    
    return param_grid

def preprocess_data(df, dayfirst):
    """Preprocess the Dataframe extracted from the uploaded file"""

    try:
        df.loc[:, 'ds'] = pd.to_datetime(df['ds'], dayfirst=dayfirst)
    except ValueError:
        st.error("Failed to convert column y to datetime")

    try:
        df.loc[:, 'y'] = pd.to_numeric(df['y'])
    except ValueError:
        st.error("Failed to convert column y to numeric type")

    df = df.sort_values('ds', ascending=True)

    return df



def main():
    """Forecasting with FBProphet App"""

    # Config
    st.set_page_config(
        page_title="Forecast App",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )  
    
    # Title
    st.title("Forecast App")

    session_state = SessionState.get(dayfirst = True, upload_file_flag = False, params={}, cv_df=pd.DataFrame())

    uploaded_file = st.file_uploader("Upload the training data", type=['csv'])
    if st.checkbox('Day first', value = session_state.dayfirst):
        session_state.dayfirst = True
    else:
        session_state.dayfirst = False

    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_df = fetch_data(uploaded_file)
        df = file_df.copy()
        df = preprocess_data(df, session_state.dayfirst)
        session_state.upload_file_flag = True
    else:
        session_state.upload_file_flag = False

    if session_state.upload_file_flag:

        # Sidebar
        # User Defined Hyperparameters
        st.sidebar.header("User Defined Hyperparameters")
        # Growth SelectBox
        growth = st.sidebar.selectbox("Growth", ["linear", "logistic"])
        if growth == 'logistic':
            cap = st.sidebar.number_input(label="Cap")
            floor = st.sidebar.number_input(label="Floor")
            df['cap'] = cap
            df['floor'] = floor

        # Seasonalities
        daily_seasonality = st.sidebar.selectbox("Daily seasonality", [False, True, "auto"])
        weekly_seasonality = st.sidebar.selectbox("Weekly seasonality", [False, True, "auto"])
        yearly_seasonality = st.sidebar.selectbox("Yearly seasonality", [True, False, "auto"])

        # Uncertainty intervals
        #uncertainty_interval = st.sidebar.number_input(label="Uncertainty interval", value=0.8, min_value=0.0, max_value=1.0)

        # Expander test
        st.write(
            """
            ####
            Validate the model by selecting an appropriate **horizon** and **error metric**.\n
            The optimal value for the **changepoint_prior_scale** and the **seasonality_prior_scale** hyperparameters will be automatically selected with respect to the selected horizon and error metric.
            """)
        left_column, center_column, right_column = st.beta_columns(3)
        with left_column:
            horizon_units = st.radio('Horizon units', ("Days", "Hours", "Months")).lower()
        with center_column:
            horizon_periods = st.number_input(label="Number of periods to validate the model", value=400, min_value=1)
            horizon_str = str(horizon_periods) + ' ' + horizon_units
        with right_column:
            error_metric = st.radio('Error metric', ("RMSE", "MSE", "MAPE")).lower()
        
        # Trigger cross validation process
        if st.button("Validate"):
            # Generate hyperparameter grid
            param_grid = create_param_grid(growth, daily_seasonality, weekly_seasonality, yearly_seasonality)
                
            # Generate all combinations of parameters
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            
            errors = []  # Store the errors for each params here

            # Progress bar
            progress = 0 
            bar = st.progress(progress)

            # Use cross validation to evaluate all parameters
            for params in all_params:

                m = Prophet(**params).fit(df)  # Fit model with given params

                try:
                    df_cv = cross_validation(m, horizon=horizon_str, period=horizon_str) # cutoff period set to horizon
                    
                except ValueError:
                    if yearly_seasonality:
                        st.warning("Warning: Make sure that yearly seasonality is set to False/auto if you have less than a year worth of training data")
                    if weekly_seasonality:
                        st.warning("Warning: Make sure that weekly seasonality is set to False/auto if you have less than a week worth of training data")
                    if daily_seasonality:
                        st.warning("Warning: Make sure that daily seasonality is set to False/auto if you have less than a day worth of training data")
                    st.warning("Warning: For validation purposes, the dataset provided should cover a time window at least 4 times the specified horizon. Make sure this condition is verified.")
                    break
            
                else:                    
                    try:
                        df_p = performance_metrics(df_cv, rolling_window=1)
                        errors.append(df_p[error_metric].values[0])
                    except KeyError:
                        if (error_metric == 'mape') and (df['y'].min()<1*10**(-8)): # Fbprophet skips MAPE when y < 1e-8
                            st.error("Possible error: Validation data contains y close to 0. Cannot compute MAPE.")
                            break
                    
                    progress = progress+(1/len(all_params))
                    bar.progress(progress)
            
            # Visualize results and best hyperparameters
            try:    
                # Create DataFrame with the cross validation results
                cross_val_results_df = pd.DataFrame(all_params)
                cross_val_results_df[error_metric] = errors
            except ValueError:
                st.error("Error: cross validation failed and it was not possible to get the respective results.")
            else: 
                # Sort results from best to worse
                cross_val_results_df = cross_val_results_df.sort_values(error_metric, ascending=True).reset_index(drop=True)

                # (Optimal) parameters that minimize the specified error metric
                optimal_params = all_params[errors.index(min(errors))]
                session_state.params = optimal_params
                session_state.cv_df = cross_val_results_df
        
        if session_state.params and not session_state.cv_df.empty:
            st.write("Optimal Hyperparameters: ", session_state.params)


if __name__ == '__main__':
    main()