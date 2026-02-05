# ============================================================================
# 6. Business Intelligence Dashboard - Interactive Python Dash Application
# ============================================================================

"""
E-Commerce Business Intelligence Dashboard
-------------------------------------------
This interactive dashboard tracks:
1. Revenue trends by category, seller, and region
2. Delivery performance metrics
3. Customer satisfaction scores
4. Lead conversion rates

Requirements:
pip install dash plotly pandas numpy --break-system-packages

Usage:
python dashboard.py
Then open browser to: http://127.0.0.1:8050/
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Note: Make sure you have the dataframes loaded from the main analysis
# This section assumes you've run the main analysis and have the cleaned data

def prepare_dashboard_data():
    """
    Prepare aggregated data for dashboard visualizations.
    Replace this with your actual data loading logic.
    """
    
    # 1. Revenue Data by Category
    revenue_by_category = (
        order_items_clean
        .merge(products_clean[['product_id', 'product_category_name']], on='product_id')
        .merge(product_cat_name_clean, on='product_category_name')
        .groupby('product_category_name_english')
        .agg(total_revenue=('price', 'sum'), order_count=('order_id', 'count'))
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )
    
    # 2. Revenue by Region
    revenue_by_region = (
        order_items_clean[['order_id', 'price']]
        .merge(orders_clean[['order_id', 'customer_id']], on='order_id')
        .merge(customers_clean[['customer_id', 'customer_state']], on='customer_id')
        .groupby('customer_state')
        .agg(total_revenue=('price', 'sum'), order_count=('order_id', 'count'))
        .reset_index()
        .sort_values('total_revenue', ascending=False)
    )
    
    # 3. Revenue by Seller (Top 20)
    revenue_by_seller = (
        order_items_clean
        .groupby('seller_id')
        .agg(total_revenue=('price', 'sum'), order_count=('order_id', 'count'))
        .reset_index()
        .sort_values('total_revenue', ascending=False)
        .head(20)
    )
    
    # 4. Delivery Performance Metrics
    delivery_metrics = orders_clean[['delivery_delay', 'delay_status']].copy()
    
    # 5. Customer Satisfaction Scores
    satisfaction_data = (
        order_reviews_clean
        .merge(orders_clean[['order_id', 'order_purchase_timestamp']], on='order_id')
    )
    satisfaction_data['order_month'] = satisfaction_data['order_purchase_timestamp'].dt.to_period('M')
    
    satisfaction_over_time = (
        satisfaction_data.groupby('order_month')['review_score']
        .agg(['mean', 'count'])
        .reset_index()
    )
    satisfaction_over_time['order_month'] = satisfaction_over_time['order_month'].astype(str)
    
    # 6. Lead Conversion Metrics
    lead_metrics = {
        'total_qualified_leads': len(qualified_leads_clean),
        'total_closed_leads': len(closed_leads_clean),
        'conversion_rate': (len(closed_leads_clean) / len(qualified_leads_clean) * 100) if len(qualified_leads_clean) > 0 else 0
    }
    
    # Lead conversion by business segment
    lead_conversion_by_segment = (
        qualified_leads_clean
        .merge(closed_leads_clean[['mql_id', 'business_segment', 'won_date']], 
               on='mql_id', how='left')
        .groupby('business_segment')
        .agg(
            total_leads=('mql_id', 'count'),
            converted_leads=('won_date', lambda x: x.notna().sum())
        )
        .reset_index()
    )
    lead_conversion_by_segment['conversion_rate'] = (
        lead_conversion_by_segment['converted_leads'] / 
        lead_conversion_by_segment['total_leads'] * 100
    ).fillna(0)
    
    return {
        'revenue_by_category': revenue_by_category,
        'revenue_by_region': revenue_by_region,
        'revenue_by_seller': revenue_by_seller,
        'delivery_metrics': delivery_metrics,
        'satisfaction_over_time': satisfaction_over_time,
        'lead_metrics': lead_metrics,
        'lead_conversion_by_segment': lead_conversion_by_segment
    }

# Prepare data
print("Preparing dashboard data...")
dashboard_data = prepare_dashboard_data()
print("✓ Data preparation complete")

# ============================================================================
# INITIALIZE DASH APP
# ============================================================================

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "E-Commerce Business Intelligence Dashboard"

# Define color scheme
colors = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'primary': '#3498db',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'card': '#ffffff'
}

# ============================================================================
# DASHBOARD LAYOUT
# ============================================================================

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
    # Header
    html.Div([
        html.H1('📊 E-Commerce Business Intelligence Dashboard',
                style={
                    'textAlign': 'center',
                    'color': colors['text'],
                    'padding': '20px',
                    'backgroundColor': colors['card'],
                    'marginBottom': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),
    ]),
    
    # KPI Cards Row
    html.Div([
        html.Div([
            html.Div([
                html.H4('💰 Total Revenue', style={'color': colors['text'], 'marginBottom': '10px'}),
                html.H2(f"R$ {dashboard_data['revenue_by_category']['total_revenue'].sum():,.2f}",
                       style={'color': colors['success'], 'margin': '0'}),
                html.P(f"{dashboard_data['revenue_by_category']['order_count'].sum():,} orders",
                      style={'color': '#7f8c8d', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            })
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '0 1%'}),
        
        html.Div([
            html.Div([
                html.H4('📦 Delivery Performance', style={'color': colors['text'], 'marginBottom': '10px'}),
                html.H2(f"{(dashboard_data['delivery_metrics']['delay_status'] == 'Early').mean() * 100:.1f}%",
                       style={'color': colors['success'], 'margin': '0'}),
                html.P('Early Deliveries',
                      style={'color': '#7f8c8d', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            })
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '0 1%'}),
        
        html.Div([
            html.Div([
                html.H4('⭐ Avg Satisfaction', style={'color': colors['text'], 'marginBottom': '10px'}),
                html.H2(f"{dashboard_data['satisfaction_over_time']['mean'].mean():.2f}/5.0",
                       style={'color': colors['warning'], 'margin': '0'}),
                html.P(f"{len(order_reviews_clean):,} reviews",
                      style={'color': '#7f8c8d', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            })
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '0 1%'}),
        
        html.Div([
            html.Div([
                html.H4('🎯 Lead Conversion', style={'color': colors['text'], 'marginBottom': '10px'}),
                html.H2(f"{dashboard_data['lead_metrics']['conversion_rate']:.1f}%",
                       style={'color': colors['danger'], 'margin': '0'}),
                html.P(f"{dashboard_data['lead_metrics']['total_closed_leads']} / {dashboard_data['lead_metrics']['total_qualified_leads']} leads",
                      style={'color': '#7f8c8d', 'margin': '5px 0 0 0'})
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'textAlign': 'center'
            })
        ], style={'width': '23%', 'display': 'inline-block', 'margin': '0 1%'}),
        
    ], style={'marginBottom': '20px'}),
    
    # Tab Navigation
    dcc.Tabs(id='dashboard-tabs', value='tab-revenue', children=[
        dcc.Tab(label='💰 Revenue Analytics', value='tab-revenue',
               style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='📦 Delivery Performance', value='tab-delivery',
               style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='⭐ Customer Satisfaction', value='tab-satisfaction',
               style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Tab(label='🎯 Lead Conversion', value='tab-leads',
               style={'fontWeight': 'bold', 'fontSize': '14px'}),
    ]),
    
    # Tab Content
    html.Div(id='tabs-content', style={'padding': '20px'})
])

# ============================================================================
# CALLBACKS FOR INTERACTIVE CONTENT
# ============================================================================

@app.callback(
    Output('tabs-content', 'children'),
    Input('dashboard-tabs', 'value')
)
def render_tab_content(active_tab):
    """Render content based on selected tab"""
    
    if active_tab == 'tab-revenue':
        return html.Div([
            # Revenue by Category
            html.Div([
                html.H3('Revenue by Product Category (Top 15)', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=px.bar(
                        dashboard_data['revenue_by_category'].head(15),
                        x='product_category_name_english',
                        y='total_revenue',
                        title='',
                        labels={'product_category_name_english': 'Category', 
                               'total_revenue': 'Total Revenue (R$)'},
                        color='total_revenue',
                        color_continuous_scale='Blues'
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis_tickangle=-45,
                        height=400
                    )
                )
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }),
            
            # Revenue by Region and Seller (side by side)
            html.Div([
                html.Div([
                    html.H3('Revenue by State (Top 10)', 
                           style={'color': colors['text'], 'marginBottom': '15px'}),
                    dcc.Graph(
                        figure=px.bar(
                            dashboard_data['revenue_by_region'].head(10),
                            x='customer_state',
                            y='total_revenue',
                            title='',
                            labels={'customer_state': 'State', 
                                   'total_revenue': 'Total Revenue (R$)'},
                            color='total_revenue',
                            color_continuous_scale='Greens'
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400
                        )
                    )
                ], style={
                    'width': '48%',
                    'display': 'inline-block',
                    'backgroundColor': colors['card'],
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'verticalAlign': 'top'
                }),
                
                html.Div([
                    html.H3('Revenue by Top Sellers', 
                           style={'color': colors['text'], 'marginBottom': '15px'}),
                    dcc.Graph(
                        figure=px.bar(
                            dashboard_data['revenue_by_seller'].head(10),
                            x=dashboard_data['revenue_by_seller'].head(10)['seller_id'].str[:15] + '...',
                            y='total_revenue',
                            title='',
                            labels={'x': 'Seller ID', 'total_revenue': 'Total Revenue (R$)'},
                            color='total_revenue',
                            color_continuous_scale='Reds'
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis_tickangle=-45,
                            height=400
                        )
                    )
                ], style={
                    'width': '48%',
                    'display': 'inline-block',
                    'backgroundColor': colors['card'],
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginLeft': '4%',
                    'verticalAlign': 'top'
                })
            ])
        ])
    
    elif active_tab == 'tab-delivery':
        # Calculate delivery metrics
        delay_distribution = dashboard_data['delivery_metrics']['delay_status'].value_counts()
        
        # Delivery delay histogram data
        delay_data = dashboard_data['delivery_metrics']['delivery_delay'].clip(-30, 30)
        
        return html.Div([
            # Delivery Status Pie Chart
            html.Div([
                html.H3('Delivery Status Distribution', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=go.Figure(data=[go.Pie(
                        labels=delay_distribution.index,
                        values=delay_distribution.values,
                        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
                        hole=0.4
                    )]).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'verticalAlign': 'top'
            }),
            
            # Delivery Delay Distribution
            html.Div([
                html.H3('Delivery Delay Distribution', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=px.histogram(
                        delay_data,
                        nbins=50,
                        title='',
                        labels={'value': 'Delivery Delay (days)', 'count': 'Number of Orders'},
                        color_discrete_sequence=['#3498db']
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400,
                        showlegend=False
                    )
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginLeft': '4%',
                'verticalAlign': 'top'
            }),
            
            # Delivery Metrics Table
            html.Div([
                html.H3('Key Delivery Metrics', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dash_table.DataTable(
                    data=[
                        {'Metric': 'Average Delivery Delay', 
                         'Value': f"{dashboard_data['delivery_metrics']['delivery_delay'].mean():.1f} days"},
                        {'Metric': 'Median Delivery Delay', 
                         'Value': f"{dashboard_data['delivery_metrics']['delivery_delay'].median():.1f} days"},
                        {'Metric': 'Early Deliveries', 
                         'Value': f"{(dashboard_data['delivery_metrics']['delay_status'] == 'Early').sum():,} ({(dashboard_data['delivery_metrics']['delay_status'] == 'Early').mean()*100:.1f}%)"},
                        {'Metric': 'On-Time Deliveries', 
                         'Value': f"{(dashboard_data['delivery_metrics']['delay_status'] == 'On Time').sum():,} ({(dashboard_data['delivery_metrics']['delay_status'] == 'On Time').mean()*100:.1f}%)"},
                        {'Metric': 'Late Deliveries', 
                         'Value': f"{(dashboard_data['delivery_metrics']['delay_status'] == 'Late').sum():,} ({(dashboard_data['delivery_metrics']['delay_status'] == 'Late').mean()*100:.1f}%)"},
                    ],
                    columns=[{'name': 'Metric', 'id': 'Metric'}, 
                            {'name': 'Value', 'id': 'Value'}],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': colors['primary'], 
                                 'color': 'white', 
                                 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'white'},
                    style_table={'overflowX': 'auto'}
                )
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginTop': '20px'
            })
        ])
    
    elif active_tab == 'tab-satisfaction':
        # Review score distribution
        review_distribution = order_reviews_clean['review_score'].value_counts().sort_index()
        
        return html.Div([
            # Satisfaction Over Time
            html.Div([
                html.H3('Average Customer Satisfaction Over Time', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=go.Figure([
                        go.Scatter(
                            x=dashboard_data['satisfaction_over_time']['order_month'],
                            y=dashboard_data['satisfaction_over_time']['mean'],
                            mode='lines+markers',
                            name='Avg Rating',
                            line=dict(color=colors['warning'], width=3),
                            marker=dict(size=8)
                        )
                    ]).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis_title='Month',
                        yaxis_title='Average Review Score',
                        yaxis_range=[0, 5],
                        height=400,
                        hovermode='x unified'
                    )
                )
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }),
            
            # Review Score Distribution and Volume
            html.Div([
                html.Div([
                    html.H3('Review Score Distribution', 
                           style={'color': colors['text'], 'marginBottom': '15px'}),
                    dcc.Graph(
                        figure=px.bar(
                            x=review_distribution.index,
                            y=review_distribution.values,
                            title='',
                            labels={'x': 'Review Score', 'y': 'Number of Reviews'},
                            color=review_distribution.values,
                            color_continuous_scale='RdYlGn'
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            height=400,
                            showlegend=False
                        )
                    )
                ], style={
                    'width': '48%',
                    'display': 'inline-block',
                    'backgroundColor': colors['card'],
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'verticalAlign': 'top'
                }),
                
                html.Div([
                    html.H3('Review Volume Over Time', 
                           style={'color': colors['text'], 'marginBottom': '15px'}),
                    dcc.Graph(
                        figure=px.bar(
                            dashboard_data['satisfaction_over_time'],
                            x='order_month',
                            y='count',
                            title='',
                            labels={'order_month': 'Month', 'count': 'Number of Reviews'},
                            color='count',
                            color_continuous_scale='Blues'
                        ).update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            xaxis_tickangle=-45,
                            height=400
                        )
                    )
                ], style={
                    'width': '48%',
                    'display': 'inline-block',
                    'backgroundColor': colors['card'],
                    'padding': '20px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginLeft': '4%',
                    'verticalAlign': 'top'
                })
            ])
        ])
    
    elif active_tab == 'tab-leads':
        return html.Div([
            # Lead Conversion Funnel
            html.Div([
                html.H3('Lead Conversion Funnel', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=go.Figure(go.Funnel(
                        y=['Qualified Leads', 'Closed Deals'],
                        x=[dashboard_data['lead_metrics']['total_qualified_leads'],
                           dashboard_data['lead_metrics']['total_closed_leads']],
                        textinfo='value+percent previous',
                        marker=dict(color=['#3498db', '#2ecc71'])
                    )).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginBottom': '20px',
                'verticalAlign': 'top'
            }),
            
            # Conversion Rate by Business Segment
            html.Div([
                html.H3('Conversion Rate by Business Segment (Top 10)', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dcc.Graph(
                    figure=px.bar(
                        dashboard_data['lead_conversion_by_segment'].nlargest(10, 'conversion_rate'),
                        x='business_segment',
                        y='conversion_rate',
                        title='',
                        labels={'business_segment': 'Business Segment', 
                               'conversion_rate': 'Conversion Rate (%)'},
                        color='conversion_rate',
                        color_continuous_scale='Oranges'
                    ).update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        xaxis_tickangle=-45,
                        height=400
                    )
                )
            ], style={
                'width': '48%',
                'display': 'inline-block',
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginLeft': '4%',
                'verticalAlign': 'top'
            }),
            
            # Lead Metrics Table
            html.Div([
                html.H3('Lead Conversion Metrics', 
                       style={'color': colors['text'], 'marginBottom': '15px'}),
                dash_table.DataTable(
                    data=[
                        {'Metric': 'Total Qualified Leads', 
                         'Value': f"{dashboard_data['lead_metrics']['total_qualified_leads']:,}"},
                        {'Metric': 'Total Closed Deals', 
                         'Value': f"{dashboard_data['lead_metrics']['total_closed_leads']:,}"},
                        {'Metric': 'Overall Conversion Rate', 
                         'Value': f"{dashboard_data['lead_metrics']['conversion_rate']:.2f}%"},
                        {'Metric': 'Lead Leakage', 
                         'Value': f"{dashboard_data['lead_metrics']['total_qualified_leads'] - dashboard_data['lead_metrics']['total_closed_leads']:,} leads ({100-dashboard_data['lead_metrics']['conversion_rate']:.1f}%)"},
                    ],
                    columns=[{'name': 'Metric', 'id': 'Metric'}, 
                            {'name': 'Value', 'id': 'Value'}],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': colors['primary'], 
                                 'color': 'white', 
                                 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'white'},
                    style_table={'overflowX': 'auto'}
                )
            ], style={
                'backgroundColor': colors['card'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'marginTop': '20px'
            })
        ])

# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("STARTING E-COMMERCE BUSINESS INTELLIGENCE DASHBOARD")
    print("="*70)
    print("\n📊 Dashboard Features:")
    print("   ✓ Revenue Analytics (by category, region, seller)")
    print("   ✓ Delivery Performance Metrics")
    print("   ✓ Customer Satisfaction Tracking")
    print("   ✓ Lead Conversion Funnel")
    print("\n🌐 Access the dashboard at: http://127.0.0.1:8050/")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*70 + "\n")
    
    app.run_server(debug=True, host='127.0.0.1', port=8050)
