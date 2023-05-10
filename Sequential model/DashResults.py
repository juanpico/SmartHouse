from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from LoadResults import loadResults

app = Dash(__name__)

INSTANCES = pd.read_csv('Data/instances/instances.csv', index_col=0).to_numpy().reshape(14)


results, decisions = loadResults()


app.layout = html.Div([
    html.H2('Resultados'),

    html.Div(
        id = "parametros",
        children = [
    
            html.Div(
                children = [
                    html.P("Ingrese instancia:"),
                    dcc.Dropdown(
                    id="dropdown_instancia",
                    options=results.instance.unique(),
                    value=results.instance.unique()[0],
                    clearable=False,
                    )
                ],
                style={"width": '15%', "display": "inline-block"}
            ),

            html.Div(
                children=[
                    html.P("Ingrese la política:"),
                    dcc.Dropdown(
                        id="dropdown_politica",
                        options=results.policy.unique(),
                        value=results.policy.unique()[0],
                        clearable=False,
                    )
                ],
                style={"width": '15%', "display": "inline-block"}
            ),

            html.Div(
                children = [
                    html.P("Ingrese fecha:"),
                    dcc.DatePickerSingle(
                        id='datepicker',
                        min_date_allowed=min(results.date),
                        max_date_allowed=max(results.date),
                        initial_visible_month=min(results.date),
                        date=min(results.date)
                    )
                ],
                style={"width": '15%', "display": "inline-block"}
            ),

            html.Div(
                children = [
                    html.P("Ingrese el valor de theta:"),
                    dcc.Dropdown(
                        id="dropdown_theta",
                        options=results.theta.unique(),
                        value=results.theta.unique()[0],
                        clearable=False,
                    )
                ],
                style={"width": '15%', "display": "inline-block"}
            )
            
        ],
       # style=dict(display='inline-block')
    ),
    
    html.Div(
        id = "graphs1",
        children = [
            html.Div(
                children = [
                   # dcc.Graph(id="graph_price"),
                    dcc.Graph(id="graph_grid")
                    
                ],
                #style={"width": '49%', "display": "inline-block"}
            ),

        ]
    ),

     html.Div(
        id = "graphs2",
        children = [

            html.Div(
                children = [
                    dcc.Graph(id="graph_cost"),
                ],
                style={"width": '49%', "display": "inline-block"}
            ),

            html.Div(
                children = [
                    dcc.Graph(id="graph_cost2"),
                ],
                style={"width": '49%', "display": "inline-block"}
            )

        ]
    ),

    html.Div(
        id = "graphs3",
        children = [

            html.Div(
                children = [
                    dcc.Graph(id="graph_price"),
                ],
                style={"width": '49%', "display": "inline-block"}
            ),

            html.Div(
                children = [
                    dcc.Graph(id="graph_battery"),
                ],
                style={"width": '49%', "display": "inline-block"}
            ),
            dcc.Graph(id="graph_appliances")

        ]
    )
    

])


@app.callback(
    Output("graph_price", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_price_chart(instance, policy, date, theta):

    fig = go.Figure()
    df = decisions[(decisions["instance"] == instance)&(decisions.date==date)&(decisions.theta==theta)&(decisions.policy==policy)]

    fig.add_trace(go.Scatter(x=df.time, y=df.p_t, mode='lines+markers', name='Energy buying price'))
    fig.update_layout(title_text="Energy price")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

@app.callback(
    Output("graph_cost", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_cost_chart(instance, policy, date, theta):

    df = results[(results["instance"] == instance)&(results.date==date)&(results.theta==theta)&(results.policy==policy)]
    
    df = df[['time','electricity cost', 'discomfort index', 'objective']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.time, y=df['electricity cost'], mode='lines+markers', name='Electricity cost'))
    fig.add_trace(go.Scatter(x=df.time, y=df['discomfort index'], mode='lines+markers', name='Discomfort index'))
    fig.add_trace(go.Scatter(x=df.time, y=df['objective'], mode='lines+markers', name='Objective'))
    fig.update_layout(title_text="Costs")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))

    return fig

@app.callback(
    Output("graph_cost2", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_cost2_chart(instance, policy, date, theta):

    df = results[(results["instance"] == instance)&(results.date==date)&(results.theta==theta)&(results.policy==policy)]
    
    df = df[['electricity cost', 'discomfort index', 'objective']]
    df = df.sum()
    
    fig = go.Figure(go.Bar(x=df.index, y=df, text=df, textposition='auto'))

    return fig


@app.callback(
    Output("graph_grid", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_grid_chart(instance, policy, date, theta):

    fig = go.Figure()
    df = decisions[(decisions["instance"] == instance)&(decisions.date==date)&(decisions.theta==theta)&(decisions.policy==policy)]

    fig.add_trace(go.Scatter(x=df.time, y=df.b_t, mode='lines+markers', name='Energy bought'))
    fig.add_trace(go.Scatter(x=df.time, y=df.s_t, mode='lines+markers', name='Energy sold'))
    fig.add_trace(go.Scatter(x=df.time, y=df.g_t, mode='lines+markers', name='Energy generated'))
    fig.add_trace(go.Scatter(x=df.time, y=df.l_t, mode='lines+markers', name='Energy demanded (nonshiftable)'))
    fig.update_layout(title_text="Grid decisions")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig


@app.callback(
    Output("graph_battery", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_battery_chart(instance, policy, date, theta):

    fig = go.Figure()
    df = decisions[(decisions["instance"] == instance)&(decisions.date==date)&(decisions.theta==theta)&(decisions.policy==policy)]

    fig.add_trace(go.Scatter(x=df.time, y=df.e_t, mode='lines+markers', name='Energy stored'))
    fig.add_trace(go.Scatter(x=df.time, y=df.h_t, mode='lines+markers', name='Energy transferred'))
    fig.update_layout(title_text="Battery decisions")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    return fig

@app.callback(
    Output("graph_appliances", "figure"), 
    [Input("dropdown_instancia", "value"),
     Input("dropdown_politica", "value"),
     Input("datepicker", "date"),
     Input("dropdown_theta", "value")])
def update_appliance_heatmap(instance, policy, date, theta):

    
    df = decisions[(decisions["instance"] == instance)&(decisions.date==date)&(decisions.theta==theta)&(decisions.policy==policy)].copy()
    df = df.dropna(axis=1)

    col_appliances = [col for col in df.columns if col[:2]=="a_"]
    appliances = [col[2:] for col in col_appliances]
    times = df['time'].str.slice(start=-8, stop=-3).tolist()

    a = df[col_appliances].to_numpy().T

    df2 = df[col_appliances+['time']].astype(str)

    df2.index = df2['time']
    df2 = df2.drop(columns=['time'])

    fig = px.imshow(df2.T, color_continuous_scale="viridis")
    data = [
        dict(
            x=times,
            y=appliances,
            z=a,
            type="heatmap",
            name="",
            #hovertemplate=hovertemplate,
            showscale=False,
            colorscale=[[0, "#caf3ff"], [1, "#2c82ff"]],
        )
    ]

    layout = dict(
        margin=dict(l=70, b=50, t=50, r=50),
        modebar={"orientation": "v"},
        font=dict(family="Open Sans"),
        #annotations=annotations,
        #shapes=shapes,
        xaxis=dict(
            side="top",
            ticks="",
            ticklen=2,
            tickfont=dict(family="sans-serif"),
            tickcolor="#ffffff",
        ),
        yaxis=dict(
            side="left", ticks="", tickfont=dict(family="sans-serif"), ticksuffix=" "
        ),
        hovermode="closest",
        showlegend=False,
    )

    #return fig
    return {"data": data, "layout": layout}




app.run_server(debug=True)