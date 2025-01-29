# from install_requirements import install_requirements
# install_requirements("requirements.txt")
import numpy as np
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 1. Initializing a Flask app instance called "app": 
app = Flask(__name__)

# 2. Defining a function to fetch data
def fetch_data(tickers, start_date="2023-01-01", end_date="2025-01-01"):
    data = {}
    for ticker in tickers: #Iterate over each ticker and download stock data using yfinance
        # Fetch the data for the ticker
        stock_data = yf.download(ticker, start=start_date, end=end_date).reset_index()
        
        # Ensure data is valid and not empty -- extracts closing price and date
        if not stock_data.empty:
            data[ticker] = stock_data["Close"][ticker].to_numpy()
            data['Date'] = stock_data["Date"].to_numpy()
        else:
            print(f"No data found for {ticker}. Skipping.")

    # Convert to a pandas df
    if data:
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        raise ValueError("No valid data fetched. Please check the tickers and try again.")


# 3. Portfolio simulation function
def simulate_portfolio(data, weights):
    """
    Simulate portfolio performance based on historical data and weights.
    """
    weights = np.array(weights)
    
    # Check if 0<=weights<=1
    if any([weight<0 for weight in weights]):
        raise ValueError("⚠️ Portfolio weights must be between 0 and 1. Please adjust your inputs.")
    
    # Check if weights sum to 1 and raise ValueError if not
    if np.sum(weights) != 1:
        raise ValueError("⚠️ Portfolio weights must sum to 1. Please adjust the input weight values.")

    # Daily returns are used to assess the performance of individual stocks and their combined portfolio.
    # Ensure the Date column is the index for time-series calculations.
    # Then, calculate the percentage change between consecutive rows, giving daily returns.
    # Remove rows with NaN values (first row will have no previous data for percentage change).
    daily_returns = data.set_index("Date").pct_change().dropna() 

    # Portfolio return is the aggregate of individual stock returns, adjusted by their contribution (weights).
    # Multiply each stock's daily return by its respective weight then sum the weighted returns for each day to get the portfolio's daily return.
    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)

    # Track the portfolio's value growth over time 
    # Convert % returns to growth factors (e.g. a 2% return becomes 1.02).
    # Compute the cumulative product, showing how the portfolio's value evolves day by day.
    portfolio_cumulative_return = (1 + portfolio_daily_returns).cumprod() 

    # Convert average daily return to an annualized metric (trading days in a year ~252)
    portfolio_annualized_return = portfolio_daily_returns.mean() * 252

    # Convert daily volatility (std) to an annualized metric -- Higher volatility indicates higher risk
    portfolio_annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)

    # How much return do we get for each unit of risk? A higher Sharpe ratio indicates better risk-adjusted performance.
    sharpe_ratio = portfolio_annualized_return / portfolio_annualized_volatility

    stats = {
        "Cumulative Return": portfolio_cumulative_return.iloc[-1] - 1, #The final cumulative return (subtracting 1 converts growth factor to percentage gain).
        "Annualized Return": portfolio_annualized_return,
        "Annualized Volatility": portfolio_annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
    }

    return portfolio_cumulative_return, stats

# Route: Home Page
# homepage to serve as the client's starting point for entering stock tickers, weights, and date ranges for analysis.
@app.route("/", methods=["GET", "POST"])  # GET loads form, POST handles user input
def index():
    tickers, weights, stock_chart = [], [], None
    start_date, end_date = "2023-01-01", "2025-01-01"
    error_message = None  # Initialize error message variable

    if request.method == "POST":  # Process user input
        try:
            tickers = request.form.get("tickers").split(",")
            weights = request.form.get("weights").split(",")
            start_date = request.form.get("start_date", start_date)
            end_date = request.form.get("end_date", end_date)  # Use default if blank

            tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
            weights = [float(weight.strip()) for weight in weights if weight.strip()]

            # Check if 0<=weights<=1
            if any([weight<0 for weight in weights]):
                raise ValueError("⚠️ Portfolio weights must be between 0 and 1. Please adjust your inputs.")

            # Check if weights sum to 1
            if sum(weights) != 1:
                raise ValueError("⚠️ Portfolio weights must sum to 1. Please adjust your inputs.")

            # Ensure valid tickers and weights
            if tickers and weights and len(tickers) == len(weights):
                df = fetch_data(tickers, start_date, end_date)

                fig = make_subplots(rows=1, cols=1)  # Initialize subplots
                for ticker in tickers:
                    fig.add_trace(go.Scatter(x=df["Date"], y=df[ticker], mode="lines", name=ticker))

                fig.update_layout(
                    title="Stock Price Trends",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    xaxis=dict(type="date", title="Date"),  # Set x-axis as date format
                    template="plotly_dark",
                )
                stock_chart = fig.to_html(full_html=False)  # Convert to HTML

        except ValueError as e:
            error_message = str(e)  # Capture error message to display on the page

    # Render index.html with stock chart & error message
    return render_template(
        "index.html",
        stock_chart=stock_chart,
        tickers=",".join(tickers),
        weights=",".join(map(str, weights)),
        start_date=start_date,
        end_date=end_date,
        error_message=error_message  # Pass error message to template
    )


# Route: Correlation Heatmap
@app.route("/correlation", methods=["GET"]) # Load page (only displays data unlike POST (form))
def correlation():
    tickers = request.args.get("tickers", "").split(",")
    start_date = request.args.get("start_date", "2023-01-01")
    end_date = request.args.get("end_date", "2025-01-01")

    if not tickers or tickers == [""]:
        return redirect(url_for("index")) # If no tickers are provided, redirects the user to the homepage (index).

    df = fetch_data(tickers, start_date, end_date)

    # Compute correlation matrix
    correlation_matrix = df.set_index("Date").corr() # Uses Date as index (time-series) and computes correlation between stocks
    # Measures how stocks move together:
    # 1.0 = Perfect correlation (stocks move identically).
    # 0.0 = No correlation (stocks are independent).
    # -1.0 = Perfect negative correlation (stocks move in opposite directions).

    # Generate heatmap using Plotly
    fig = px.imshow( 
        correlation_matrix,
        text_auto=True,
        title="Stock Correlation Heatmap",
        color_continuous_scale="Blues",
    )
    # Convert the interactive heatmap into an HTML string that can be embedded in a webpage.
    heatmap = fig.to_html(full_html=False)
    # Allows the user to view the correlation heatmap in the browser.
    return render_template("correlation.html", heatmap=heatmap, tickers=",".join(tickers), start_date=start_date, end_date=end_date)

# Route: Portfolio Returns
@app.route("/portfolio_returns", methods=["GET"])
def portfolio_returns():
    tickers = request.args.get("tickers", "").split(",")
    weights = request.args.get("weights", "").split(",")
    start_date = request.args.get("start_date", "2023-01-01")
    end_date = request.args.get("end_date", "2025-01-01")

    if not tickers or tickers == [""] or not weights or weights == [""]:
        return redirect(url_for("index"))

    weights = [float(w) for w in weights]
    df = fetch_data(tickers, start_date, end_date)

    try:
        # Ensure data is properly aligned
        portfolio_cumulative_return, stats = simulate_portfolio(df, weights)

        # Convert cumulative return to percentage
        cumulative_return_percent = round(portfolio_cumulative_return * 100, 2)

        # Plot cumulative return
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Date"], y=cumulative_return_percent,
            mode="lines", name="Portfolio Cumulative Return"
        ))

        fig.update_layout(
            title="Portfolio Cumulative Return Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            xaxis=dict(type="date", title="Date"),
            template="plotly_dark",
        )

        return render_template(
            "portfolio_returns.html",
            cumulative_return=cumulative_return_percent.iloc[-1],  
            annualized_return=round(stats["Annualized Return"] * 100, 2),
            annualized_volatility=round(stats["Annualized Volatility"] * 100, 2),
            sharpe_ratio=round(stats["Sharpe Ratio"], 2),
            portfolio_chart=fig.to_html(full_html=False),
            tickers=",".join(tickers),
            weights=",".join(map(str, weights)),
            start_date=start_date,
            end_date=end_date,
            error_message=None,  # No error
        )

    except ValueError as e:
        # Handle the error by displaying it on the webpage
        return render_template(
            "portfolio_returns.html",
            error_message=str(e),  # Pass the error message
            tickers=",".join(tickers),
            weights=",".join(map(str, weights)),
            start_date=start_date,
            end_date=end_date,
        )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)