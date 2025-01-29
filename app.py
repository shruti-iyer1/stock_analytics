# from install_requirements import install_requirements
# install_requirements("requirements.txt")
import numpy as np
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from portfolio_utils import simulate_portfolio

app = Flask(__name__)

# Helper function to fetch data
def fetch_data(tickers, start_date="2023-01-01", end_date="2025-01-01"):
    data = {}
    for ticker in tickers:
        # Fetch the data for the ticker
        stock_data = yf.download(ticker, start=start_date, end=end_date).reset_index()
        
        # Ensure data is valid and not empty
        if not stock_data.empty:
            data[ticker] = stock_data["Close"][ticker].to_numpy()
            data['Date'] = stock_data["Date"].to_numpy()
        else:
            print(f"No data found for {ticker}. Skipping.")

    # Convert to a pandas DataFrame
    if data:
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    else:
        raise ValueError("No valid data fetched. Please check the tickers and try again.")


# Portfolio simulation function
def simulate_portfolio(data, weights):
    """
    Simulate portfolio performance based on historical data and weights.
    """
    weights = np.array(weights)
    weights /= weights.sum()

    daily_returns = data.set_index("Date").pct_change().dropna()

    portfolio_daily_returns = (daily_returns * weights).sum(axis=1)
    portfolio_cumulative_return = (1 + portfolio_daily_returns).cumprod()
    portfolio_annualized_return = portfolio_daily_returns.mean() * 252
    portfolio_annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252)
    sharpe_ratio = portfolio_annualized_return / portfolio_annualized_volatility

    stats = {
        "Cumulative Return": portfolio_cumulative_return.iloc[-1] - 1,
        "Annualized Return": portfolio_annualized_return,
        "Annualized Volatility": portfolio_annualized_volatility,
        "Sharpe Ratio": sharpe_ratio,
    }

    return portfolio_cumulative_return, stats

# Route: Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    tickers, weights, stock_chart = [], [], None
    start_date, end_date = "2023-01-01", "2025-01-01"

    if request.method == "POST":
        tickers = request.form.get("tickers").split(",")
        weights = request.form.get("weights").split(",")
        start_date = request.form.get("start_date", start_date)
        end_date = request.form.get("end_date", end_date)

        tickers = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
        weights = [float(weight.strip()) for weight in weights if weight.strip()]

        if tickers and weights and len(tickers) == len(weights):
            df = fetch_data(tickers, start_date, end_date)

            fig = make_subplots(rows=1, cols=1)
            for ticker in tickers:
                fig.add_trace(go.Scatter(x=df["Date"], y=df[ticker], mode="lines", name=ticker))

            fig.update_layout(
                title="Stock Price Trends",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis=dict(type="date", title="Date"),
                template="plotly_dark",
            )
            stock_chart = fig.to_html(full_html=False)

    return render_template("index.html", stock_chart=stock_chart, tickers=",".join(tickers), weights=",".join(map(str, weights)), start_date=start_date, end_date=end_date)

# Route: Correlation Heatmap
@app.route("/correlation", methods=["GET"])
def correlation():
    tickers = request.args.get("tickers", "").split(",")
    start_date = request.args.get("start_date", "2023-01-01")
    end_date = request.args.get("end_date", "2025-01-01")

    if not tickers or tickers == [""]:
        return redirect(url_for("index"))

    df = fetch_data(tickers, start_date, end_date)

    # Compute correlation matrix
    correlation_matrix = df.set_index("Date").corr()

    # Generate heatmap using Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        title="Stock Correlation Heatmap",
        color_continuous_scale="Blues",
    )
    heatmap = fig.to_html(full_html=False)

    return render_template("correlation.html", heatmap=heatmap, tickers=",".join(tickers), start_date=start_date, end_date=end_date)

# Route: Portfolio Summary
@app.route("/portfolio_summary", methods=["GET"])
def portfolio_summary():
    tickers = request.args.get("tickers", "").split(",")
    start_date = request.args.get("start_date", "2023-01-01")
    end_date = request.args.get("end_date", "2025-01-01")

    if not tickers or tickers == [""]:
        return redirect(url_for("index"))

    df = fetch_data(tickers, start_date, end_date)

    total_return = (df.set_index("Date").iloc[-1] / df.set_index("Date").iloc[0] - 1).mean() * 100
    avg_daily_return = df.set_index("Date").pct_change().mean().mean() * 100
    risk = df.set_index("Date").pct_change().std().mean() * 100

    return render_template(
        "portfolio_summary.html",
        total_return=round(total_return, 2),
        avg_daily_return=round(avg_daily_return, 2),
        risk=round(risk, 2),
        tickers=",".join(tickers),
        start_date=start_date,
        end_date=end_date,
    )

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

    portfolio_cumulative_return, stats = simulate_portfolio(df, weights)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=portfolio_cumulative_return, mode="lines", name="Portfolio Return"))

    fig.update_layout(
        title="Portfolio Cumulative Return Over Time",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        xaxis=dict(type="date", title="Date"),
        template="plotly_dark",
    )

    return render_template(
        "portfolio_returns.html",
        cumulative_return=round(stats["Cumulative Return"] * 100, 2),
        annualized_return=round(stats["Annualized Return"] * 100, 2),
        annualized_volatility=round(stats["Annualized Volatility"] * 100, 2),
        sharpe_ratio=round(stats["Sharpe Ratio"], 2),
        portfolio_chart=fig.to_html(full_html=False),
        tickers=",".join(tickers),
        weights=",".join(map(str, weights)),
        start_date=start_date,
        end_date=end_date,
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)