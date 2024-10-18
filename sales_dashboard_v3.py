# %%
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# %%
# Setup the page
st.set_page_config(page_title="Sales Dashboard", layout="wide")
st.title("Sales Performance Dashboard")

manually_signed_dealers = {
    7: 73,  # July 2024
    8: 71,  # August 2024
    9: 69   # September 2024
}

def calculate_recent_activation_metrics(df, activation_month, activation_year, manually_signed_dealers):
    # Filter for dealers activated in the specified month
    activated_dealers = df[(df['Dealer Sign up Date'].dt.month == activation_month) & 
                           (df['Dealer Sign up Date'].dt.year == activation_year)]
    
    # Get unique dealer count
    #num_dealers_signed = activated_dealers['Dealer Name'].nunique()
    num_dealers_signed = manually_signed_dealers
    
    # Get dealers with contracts
    dealers_with_contracts = activated_dealers[activated_dealers['Sale Date'] >= activated_dealers['Dealer Sign up Date']]['Dealer Name'].nunique()
    
    # Calculate total sales since signup
    total_vsc = activated_dealers[activated_dealers['Parent Product Type'] == 'VSC']['Sale Date'].count()
    total_ancillary = activated_dealers[activated_dealers['Parent Product Type'] == 'Ancillary']['Sale Date'].count()
    
    # Calculate sales for specific months
    months_to_calculate = [activation_month, activation_month + 1, activation_month + 2, activation_month + 3]
    years_to_calculate = [activation_year] * 4
    
    # Adjust years if months go beyond December
    for i in range(len(months_to_calculate)):
        if months_to_calculate[i] > 12:
            months_to_calculate[i] -= 12
            years_to_calculate[i] += 1
    
    monthly_sales = {}
    for month, year in zip(months_to_calculate, years_to_calculate):
        month_sales = activated_dealers[(activated_dealers['Sale Date'].dt.month == month) & 
                                        (activated_dealers['Sale Date'].dt.year == year)]
        vsc_sales = month_sales[month_sales['Parent Product Type'] == 'VSC']['Sale Date'].count()
        ancillary_sales = month_sales[month_sales['Parent Product Type'] == 'Ancillary']['Sale Date'].count()
        monthly_sales[f"{year}-{month:02d}"] = {'VSC': vsc_sales, 'Ancillary': ancillary_sales}
    
    return {
        'Dealers Signed': num_dealers_signed,
        'Dealers with Contracts': dealers_with_contracts,
        'Total VSC Sales': total_vsc,
        'Total Ancillary Sales': total_ancillary,
        'Monthly Sales': monthly_sales
    }


# %%

# Function to load data
@st.cache_data  # This caches the data to make it load faster
def load_data(file):
    file_extension = file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        df = pd.read_csv(file)
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None
    
    # Convert date columns to datetime
    df['Sale Date'] = pd.to_datetime(df['Sale Date'])
    df['Dealer Sign up Date'] = pd.to_datetime(df['Dealer Sign up Date'])
    return df

# File uploader
uploaded_file = st.file_uploader("Upload your sales data (Excel/CSV)", type=['xlsx', 'csv'])

if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file)
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_sales = len(df)
        delta_pct = ((len(df[df['Sale Date'].dt.year == 2024]) / len(df[df['Sale Date'].dt.year == 2023])) - 1) * 100
        st.metric("Total Sales", total_sales, delta=f"{delta_pct:.1f}%")
    
    with col2:
        active_dealers = df['Dealer Name'].nunique()
        st.metric("Active Dealers", active_dealers)
    
    with col3:
        # Calculate average time to first sale for new dealers
        min_sale_date = df['Sale Date'].min()
        today = pd.Timestamp.today()
        new_dealers_mask = (df['Dealer Sign up Date'] >= min_sale_date) & (df['Dealer Sign up Date'] <= today)
        new_dealers_df = df[new_dealers_mask].copy()
        
        dealer_metrics = new_dealers_df.groupby('Dealer Name').agg({
            'Sale Date': 'min',
            'Dealer Sign up Date': 'first'
        }).reset_index()
        
        avg_time_to_first_sale = (dealer_metrics['Sale Date'] - dealer_metrics['Dealer Sign up Date']).mean().days
        st.metric("New Dealer Avg Time to First Sale", f"{avg_time_to_first_sale:.1f} days")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sales Trends", "Geographic Distribution", "Product Analysis", "Dealer Performance","New Dealer Performance"])
    
    with tab1:
        # Monthly sales trend
        monthly_sales = df.resample('M', on='Sale Date').size().reset_index()
        monthly_sales.columns = ['Date', 'Sales']
        
        fig_trend = px.line(monthly_sales, x='Date', y='Sales', 
                           title='Monthly Sales Trend')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Year over Year comparison
        yoy_sales = df.pivot_table(
            index=df['Sale Date'].dt.month,
            columns=df['Sale Date'].dt.year,
            values='Dealer Name',
            aggfunc='count'
        ).fillna(0)
        
        fig_yoy = px.line(yoy_sales, title='Year over Year Comparison')
        st.plotly_chart(fig_yoy, use_container_width=True)

    with tab2:
        # State-wise sales heat map
        state_sales = df['Dealer State'].value_counts().reset_index()
        state_sales.columns = ['state', 'sales']
        
        fig_map = px.choropleth(state_sales,
                               locations='state',
                               locationmode="USA-states",
                               scope="usa",
                               color='sales',
                               color_continuous_scale="Viridis",
                               title="Sales by State")
        st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        # Product type distribution
        col1, col2 = st.columns(2)
        
        with col1:
            product_sales = df['Product Type'].value_counts()
            fig_product = px.pie(values=product_sales.values, 
                               names=product_sales.index, 
                               title='Sales by Product Type')
            st.plotly_chart(fig_product, use_container_width=True)
        
        with col2:
            parent_product_sales = df['Parent Product Type'].value_counts()
            fig_parent = px.pie(values=parent_product_sales.values, 
                              names=parent_product_sales.index, 
                              title='Sales by Parent Product Type')
            st.plotly_chart(fig_parent, use_container_width=True)

    with tab4:
        # Top dealers
        top_dealers = df.groupby('Dealer Name').size().sort_values(ascending=False).head(10)
        fig_dealers = px.bar(x=top_dealers.index, y=top_dealers.values,
                           title='Top 10 Dealers by Sales Volume')
        st.plotly_chart(fig_dealers, use_container_width=True)

        # Show top dealers table
        st.subheader("Top Dealers Details")
        top_dealers_df = df.groupby('Dealer Name').agg({
            'Sale Date': ['count', 'min', 'max'],
            'Dealer State': 'first'
        }).sort_values(('Sale Date', 'count'), ascending=False).head(10)
        
        top_dealers_df.columns = ['Total Sales', 'First Sale', 'Last Sale', 'State']
        st.dataframe(top_dealers_df)
        
    with tab5:
        
        # New Dealer Analysis Section
        st.subheader("New Dealer (Jan 2023 - Current) Performance Analysis")

        # Get date range
        min_sale_date = df['Sale Date'].min()
        today = pd.Timestamp.today()
        
        # Filter for new dealers
        new_dealers_mask = (df['Dealer Sign up Date'] >= min_sale_date) & (df['Dealer Sign up Date'] <= today)
        new_dealers_df = df[new_dealers_mask].copy()
        
        # Monthly sales trend chart for new dealers
        st.subheader("New Dealer Monthly Sales Trend by Product Type")
        
        # Product type filter
        product_type = st.selectbox(
            "Select Product Type",
            ["VSC", "Ancillary"],
            key="new_dealer_product_filter"
        )
        
        # Calculate months since activation for each sale
        def calculate_months_since_activation(row):
            return ((row['Sale Date'].year - row['Dealer Sign up Date'].year) * 12 + 
                    row['Sale Date'].month - row['Dealer Sign up Date'].month)
        
        new_dealers_df['Months Since Activation'] = new_dealers_df.apply(calculate_months_since_activation, axis=1)
        
        # Filter for selected product type
        product_filtered_df = new_dealers_df[new_dealers_df['Parent Product Type'] == product_type]
        
        # Create monthly trend
        monthly_trend = product_filtered_df.groupby(['Dealer Name', 'Months Since Activation']).size().reset_index(name='Sales')
        
        # Calculate average sales by month since activation
        avg_monthly_trend = monthly_trend.groupby('Months Since Activation')['Sales'].mean().reset_index()
        
        # Create the trend chart
        fig = px.line(avg_monthly_trend, 
                    x='Months Since Activation', 
                    y='Sales',
                    title=f'Average {product_type} Sales by Months Since Dealer Activation')
        
        fig.update_layout(
            xaxis_title="Months Since Dealer Activation",
            yaxis_title="Average Number of Sales",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total New Dealers", len(new_dealers_df['Dealer Name'].unique()))
            
        with col2:
            active_new_dealers = len(new_dealers_df[
                new_dealers_df['Sale Date'] >= (today - pd.Timedelta(days=30))
            ]['Dealer Name'].unique())
            st.metric("Active New Dealers (Last 30 Days)", active_new_dealers)
            
        with col3:
            avg_sales_per_dealer = new_dealers_df.groupby('Dealer Name').size().mean()
            st.metric("Avg Sales per New Dealer", f"{avg_sales_per_dealer:.1f}")
        
        # Create summary DataFrame for new dealers
        dealer_metrics = new_dealers_df.groupby('Dealer Name').agg({
            'Sale Date': ['count', 'min'],
            'Dealer Sign up Date': 'first',
            'Dealer State': 'first'
        }).reset_index()
        
        # Rename columns for clarity
        dealer_metrics.columns = ['Dealer Name', 'Total Sales', 'First Sale Date', 'Activation Date', 'State']
        
        # Calculate product type specific sales
        vsc_sales = new_dealers_df[new_dealers_df['Parent Product Type'] == 'VSC'].groupby('Dealer Name')['Sale Date'].count()
        ancillary_sales = new_dealers_df[new_dealers_df['Parent Product Type'] == 'Ancillary'].groupby('Dealer Name')['Sale Date'].count()
        
        dealer_metrics['VSC Sales'] = dealer_metrics['Dealer Name'].map(vsc_sales).fillna(0)
        dealer_metrics['Ancillary Sales'] = dealer_metrics['Dealer Name'].map(ancillary_sales).fillna(0)
        
        # Reorder columns
        dealer_metrics = dealer_metrics[[
            'Dealer Name', 'Total Sales', 'VSC Sales', 'Ancillary Sales', 
            'Activation Date', 'First Sale Date', 'State'
        ]]
        # Display the summary table
        
        st.subheader("New Dealer Details")
        st.dataframe(dealer_metrics.style.format({
            'Total Sales': '{:,.0f}',
            'VSC Sales': '{:,.0f}',
            'Ancillary Sales': '{:,.0f}',
            'Activation Date': '{:%m/%d/%Y}',
            'First Sale Date': '{:%m/%d/%Y}'
        }))

        st.subheader("Recent Dealer (Q3 2024) Activations Analysis")

        # Calculate metrics for July, August, and September 2024
        months_to_analyze = [7, 8, 9]  # July, August, September
        year_to_analyze = 2024

        for month in months_to_analyze:
            metrics = calculate_recent_activation_metrics(df, month, year_to_analyze, manually_signed_dealers[month])
            month_name = pd.to_datetime(f"{year_to_analyze}-{month}-01").strftime("%B")
            
            st.write(f"### Dealers Activated {month_name} {year_to_analyze}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Dealers Signed", metrics['Dealers Signed'])
            col2.metric("Dealers with Contracts", metrics['Dealers with Contracts'])
            col3.metric("Total VSC Sales", metrics['Total VSC Sales'])
            col4.metric("Total Ancillary Sales", metrics['Total Ancillary Sales'])

            st.write("Monthly Sales Breakdown:")
            monthly_data = []
            for date, sales in metrics['Monthly Sales'].items():
                monthly_data.append({
                    'Month': pd.to_datetime(date).strftime("%B %Y"),
                    'VSC Sales': sales['VSC'],
                    'Ancillary Sales': sales['Ancillary']
                })
            
            monthly_df = pd.DataFrame(monthly_data)
            st.table(monthly_df.style.format({
                'VSC Sales': '{:,.0f}',
                'Ancillary Sales': '{:,.0f}'
            }))
            
            # Add a bar chart for visual representation
            fig = px.bar(monthly_df, x='Month', y=['VSC Sales', 'Ancillary Sales'], 
                        title=f"Monthly Sales for Dealers Activated in {month_name} {year_to_analyze}")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")  # Add a separator between months

        # Additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total New Dealers", len(new_dealers_df['Dealer Name'].unique()))
            
        with col2:
            active_new_dealers = len(new_dealers_df[
                new_dealers_df['Sale Date'] >= (today - pd.Timedelta(days=30))
            ]['Dealer Name'].unique())
            st.metric("Active New Dealers (Last 30 Days)", active_new_dealers)
            
        with col3:
            avg_sales_per_dealer = new_dealers_df.groupby('Dealer Name').size().mean()
            st.metric("Avg Sales per New Dealer", f"{avg_sales_per_dealer:.1f}")
        
        # Create summary DataFrame for new dealers
        dealer_metrics = new_dealers_df.groupby('Dealer Name').agg({
            'Sale Date': ['count', 'min'],
            'Dealer Sign up Date': 'first',
            'Dealer State': 'first'
        }).reset_index()
        
        # Rename columns for clarity
        dealer_metrics.columns = ['Dealer Name', 'Total Sales', 'First Sale Date', 'Activation Date', 'State']
        
        # Calculate product type specific sales
        vsc_sales = new_dealers_df[new_dealers_df['Parent Product Type'] == 'VSC'].groupby('Dealer Name')['Sale Date'].count()
        ancillary_sales = new_dealers_df[new_dealers_df['Parent Product Type'] == 'Ancillary'].groupby('Dealer Name')['Sale Date'].count()
        
        dealer_metrics['VSC Sales'] = dealer_metrics['Dealer Name'].map(vsc_sales).fillna(0)
        dealer_metrics['Ancillary Sales'] = dealer_metrics['Dealer Name'].map(ancillary_sales).fillna(0)
        
        # Reorder columns
        dealer_metrics = dealer_metrics[[
            'Dealer Name', 'Total Sales', 'VSC Sales', 'Ancillary Sales', 
            'Activation Date', 'First Sale Date', 'State'
        ]]
        
        

else:
    st.info("Please upload your sales data file to begin analysis")

# Add filters in the sidebar
with st.sidebar:
    st.header("Filters")
    st.info("Upload a file to enable filters")


