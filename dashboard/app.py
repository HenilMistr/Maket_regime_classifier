import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']



available_files = [f for f in os.listdir() if f.endswith("_regime_classified_data.csv")]
available_tickers = [f.replace("_regime_classified_data.csv", "") for f in available_files]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Market Regime Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("📊 Market Regime Classifier Dashboard", className="text-center text-primary mb-4"),
        
        html.Div([
            html.Label("Select Ticker:", className="font-weight-bold"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in available_tickers],
                value=available_tickers[0] if available_tickers else None,
                className="mb-4"
            )
        ], className="container"),

        html.Div([
            html.Div([
                dcc.Graph(id='price-chart')
            ], className="card shadow-sm p-3 mb-4 bg-white rounded"),

            html.Div([
                dcc.Graph(id='regime-distribution')
            ], className="card shadow-sm p-3 mb-4 bg-white rounded")
        ], className="container")
    ])
], style={'backgroundColor': '#f8f9fa', 'paddingBottom': '30px'})

@app.callback(
    [Output('price-chart', 'figure'),
     Output('regime-distribution', 'figure')],
    [Input('ticker-dropdown', 'value')]
)
def update_dashboard(ticker):
    if ticker is None:
        return {}, {}

    file_name = f"{ticker}_regime_classified_data.csv"
    if not os.path.exists(file_name):
        return {}, {}

    df = pd.read_csv(file_name, parse_dates=['Date'])

    price_fig = px.line(df, x='Date', y='Close', color='Regime',
                        title=f"{ticker} Price & Regimes",
                        color_discrete_map={'Bull': 'green', 'Bear': 'red', 'Sideways': 'gray'})
    price_fig.update_layout(template="plotly_white")

    regime_fig = px.histogram(df, x='Regime', title=f"{ticker} Regime Distribution")
    regime_fig.update_layout(template="plotly_white")

    return price_fig, regime_fig

if __name__ == '__main__':
    app.run(debug=True)
