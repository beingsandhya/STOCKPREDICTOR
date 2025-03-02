
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


MODEL_PATH = "stock_prediction_model.h5"
model = load_model(MODEL_PATH)


st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")


stock_symbol = st.sidebar.text_input("Enter Stock Ticker:", "AAPL").upper()
start_date = st.sidebar.date_input("📅 Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("📅 End Date", pd.to_datetime("today"))

@st.cache_data
def get_stock_data(stock_symbol, start_date, end_date):
    stock = yf.download(stock_symbol, start=start_date, end=end_date)
    if stock.empty:
        return None
    stock.index = pd.to_datetime(stock.index)  
    return stock


def prepare_data(data, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[['Close']])
    X = []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

if st.sidebar.button("🔍 Predict Prices"):
    with st.spinner("Fetching data and making predictions..."):
        
        data = get_stock_data(stock_symbol, start_date, end_date)
        if data is None:
            st.error("❌ Invalid Stock Symbol or No Data Available!")
        else:
           
            X, scaler = prepare_data(data)

            
            predicted_prices = model.predict(X[-30:])
            predicted_prices = scaler.inverse_transform(predicted_prices)

            
            future_dates = pd.date_range(start=data.index[-1], periods=30, freq="B")
            predicted_df = pd.DataFrame({"Date": future_dates, "Predicted": predicted_prices.flatten()})
            predicted_df.set_index("Date", inplace=True)

            
            fig = go.Figure()

            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Actual Prices"
            ))

           
            data["50_MA"] = data["Close"].rolling(window=50).mean()
            data["200_MA"] = data["Close"].rolling(window=200).mean()
            fig.add_trace(go.Scatter(
                x=data.index, y=data["50_MA"], mode="lines", name="50-Day MA", line=dict(color="blue", width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data.index, y=data["200_MA"], mode="lines", name="200-Day MA", line=dict(color="purple", width=2)
            ))

           
            fig.add_trace(go.Scatter(
                x=predicted_df.index, y=predicted_df["Predicted"], mode="lines",
                name="Predicted Prices", line=dict(color="red", width=3, dash='dot')
            ))

            
            fig.update_layout(
                title=f"📊 {stock_symbol} Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Stock Price (USD)",
                xaxis_rangeslider_visible=True,
                template="plotly_dark",
                legend=dict(x=0, y=1, traceorder="normal", font=dict(size=12)),
                hovermode="x unified"
            )

          
            st.plotly_chart(fig)





















































































































           


           
           
           

           
           
           
           
           
           

           
           














































































            
            
            
            
            
            
            
            
            

            
            

            
            
            
            

            
            

            

        
        

        
        
        










































































