import random
from datetime import datetime
import base64
import plotly.graph_objects as go
import streamlit as st

def generate_random_date():
    start_date = datetime(2004, 1, 1)
    end_date = datetime.now()
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime("%m/%Y")

def generate_random_assets():
    return f"${random.uniform(10, 100):.1f} m"

def display_recent_interactions(client_name, client_demographics_df):
    if client_name.strip() != "":
        recent_interactions = client_demographics_df[client_demographics_df['Client'] == client_name]
        if not recent_interactions.empty:
            return ', '.join(recent_interactions['Recent Interactions'].astype(str).tolist())
        else:
            return "No recent interactions data available"
    else:
        return "No client selected"

def create_download_link(val, filename):
    b64 = base64.b64encode(val).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}.pdf">Download file</a>'

def plot_growth_of_10000(monthly_returns_df, selected_strategy, benchmark):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_returns_df.index,
        y=(monthly_returns_df[selected_strategy].cumsum() + 1) * 10000,
        mode='lines',
        name=f'{selected_strategy} Fund'
    ))
    
    if benchmark != "N/A":
        fig.add_trace(go.Scatter(
            x=monthly_returns_df.index,
            y=(monthly_returns_df[benchmark].cumsum() + 1) * 10000,
            mode='lines',
            name='Benchmark'
        ))

    fig.update_layout(
        title=f"Growth of $10K - {selected_strategy} Fund",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        legend_title="Legend",
        template="plotly_dark"
    )
    
    return fig

