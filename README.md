# streamlit-reset
Strange and subtle refresh Streamlit bugs

Streamlit app sometimes resets during fbprophet's cross-validation process.

To try to reproduce this strange behavior, follow these steps:
  1. Run the app.py file with streamlit: streamlit run app.py 
  2. Upload the vdf_stock.csv file using the file_uploader leaving the "Day first" checkbox selected.
  3. Press the Validate button (you may leave the default settings unchange as I have noticed this behavior happens more frequently when Yearly seasonality is set to True). The app should run for at least a few iterations of the corss-validation process and then, all of a sudden, the text on the top-right corner of the screen will change from RUNNING... to CONNECTING... and the app will reset to its initial state (where there is only the file_uplaoder feature). Bear in mind that this does not always happen a sometimes the application will be able to run smoothly to the end of the cross-validation process, hence the difficulty in understading the root cause.
