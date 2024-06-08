
from app.baseline import compute_long_term_baseline, read_and_format
import plotly.graph_objs as go
import plotly.io as pio
from config import config

df = read_and_format('MUO')
baseline = compute_long_term_baseline('MUO', df.index.min(), df.index.max(), df)

# trace_df = go.Scatter(x=df.index, y=df['X'], mode='lines', name='Original Data')
# trace_baseline = go.Scatter(x=baseline.index, y=baseline['X'], mode='lines', name='Baseline')

# # Create the layout for the plot
# layout = go.Layout(
#     title='Data, Baseline, and Difference',
#     xaxis=dict(title='Time'),
#     yaxis=dict(title='Value')
# )
# fig = go.Figure(data=[trace_df, trace_baseline], layout=layout)
# pio.show(fig)