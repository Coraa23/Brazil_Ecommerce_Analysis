# based on Kaggle Brazilian E-commerce Public Dataset by Olist

# Part 1: Data Overview

# import libraries
import pandas as pd
from itertools import combinations
from collections import defaultdict
from src.Visualization.plot import plot_histogram, plot_pairplot
import src.DataPreprocessing.DataPreprocessor
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
# display all the columns
# summarize column names and datatype in each csv files.

file_names = {'customers': 'olist_customers_dataset',
             'geolocation': 'olist_geolocation_dataset',
             'order_items': 'olist_order_items_dataset',
             'order_payments':'olist_order_payments_dataset',
             'order_reviews': 'olist_order_reviews_dataset',
             'orders': 'olist_orders_dataset',
             'products': 'olist_products_dataset',
             'sellers': 'olist_sellers_dataset',
             'product_category': 'product_category_name_translation'
            }
for name in file_names.keys():
    locals()[name] = pd.read_csv('/Users/a21997/Desktop/ESSEC/04-Introduction to Python/Group-15/Brazil_Ecommerce_Dataset/'
                                 +file_names[name]+'.csv')
dataset_name = file_names.keys()
print(dataset_name)
print(type(dataset_name))

# create a table listing all the column names from the different tables in dataset
df_column_name = pd.DataFrame([globals()[i].columns for i in dataset_name], index=dataset_name).T
# print(df_column_name)



# # define all datasets
# dataset_list = ['']
# define a function to return the datatype and null value percentage in each table
def dataset_info(df, df_name):
    df1 = pd.DataFrame([df.nunique(), df.dtypes, df.isna().sum()*100/len(df)],
                       index=['unique_value','datatype','nullvalue_p%']).T
    df1.index.name = 'column_names'
    df1.reset_index(level='column_names',inplace=True)
    return df1.assign(table = df_name)[['table'] + df1.columns.tolist()]

# go through all datasets and put their info into one table
df_dataset_info = pd.DataFrame()
for dataset in dataset_name:
    df2 = dataset_info(globals()[dataset], dataset)
    df_dataset_info  = df_dataset_info.append(df2)

df_dataset_info.reset_index(drop=True, inplace=True)
print(df_dataset_info)

# from above analysis we can find that most of the columns don't have null values except for review comment in order reviews


# construct order dataset and add additional columns
orders_df = orders[['order_id', 'customer_id', 'order_status', 'order_purchase_timestamp', 'order_delivered_carrier_date',
                    'order_delivered_customer_date', 'order_estimated_delivery_date', 'order_approved_at']]

# transfer timestamp data into different columns for analysis
orders_df = orders_df.assign(
    purchase_date = pd.to_datetime(orders_df['order_purchase_timestamp']).dt.date,
    purchase_month=pd.to_datetime(orders_df['order_purchase_timestamp']).dt.month,
    purchase_year=pd.to_datetime(orders_df['order_purchase_timestamp']).dt.year,
    # purchase_month_year= pd.to_datetime(orders_df['order_purchase_timestamp']).dt.strftime('%b-%y'),
    purchase_month_year=pd.to_datetime(orders_df['order_purchase_timestamp']).dt.to_period('M'),
    purchase_day=pd.to_datetime(orders_df['order_purchase_timestamp']).dt.day_name(),
    purchase_hour=pd.to_datetime(orders_df['order_purchase_timestamp']).dt.hour,
    )



# joining different tables/datasets to conduct analysis
detail_df= (((order_items.merge(orders_df, how="left",on='order_id'))
             .merge(products, how="left",on='product_id'))
            .merge(product_category, how='left', on='product_category_name'))\
            .merge(customers, how="left", on="customer_id")
# Only keep delivered order to conduct analysis
detail_df = detail_df[detail_df['order_status']=='delivered']

# select useful columns into new dataframe
order_analysis_df = detail_df[['order_id', 'product_id','price', 'order_status', 'purchase_date', 'purchase_year',
                               'purchase_month','purchase_day','product_category_name_english',
                               'customer_unique_id', 'customer_state', 'order_delivered_customer_date',
                               'order_estimated_delivery_date','order_delivered_carrier_date','shipping_limit_date',
                               'seller_id', 'purchase_month_year', 'order_approved_at']]
print(order_analysis_df.info())
# Part 2: Order Analysis

# Part 2.1: Revenue Analysis
#
# # create pivot table to conduct revenue analysis from detailed dataframe
# order_month_pivot = order_analysis_df.pivot_table(values=['order_id', 'price'],
#                                                   index=['purchase_month_year'],
#                                                   aggfunc={'order_id': 'nunique', 'price': 'sum'})
# order_month_pivot = order_month_pivot.sort_index(ascending=[1, 1, 1])

# Group by month and count the number of orders
monthly_orders = order_analysis_df.groupby('purchase_month_year').agg({'order_id':'nunique', 'price':'sum'}).reset_index()

# Plotting with Seaborn
# present data to line chart:
width = .45
fig = plt.figure()
ax1 = monthly_orders['price'].plot(kind='bar', figsize=(20,7), width = width)
ax2 = monthly_orders['order_id'].plot(secondary_y=True, color='#007FD1')
ax1.set(ylabel='revenue ($R1000)')
ax1.set(title="Trend of Revenue ($R1000) & Orders by Month", xlabel="purchased month")
ax1.title.set_size(18)
ax1.xaxis.label.set_size(14)
ax1.yaxis.label.set_size(14)
ax1.xaxis.set_tick_params(labelsize=13)
ax1.yaxis.set_tick_params(labelsize=13)
ax2.yaxis.label.set_size(14)
ax2.yaxis.set_tick_params(labelsize=13)
ax2.set_ylabel('no of orders', rotation=-90, labelpad=20)
fig.legend(loc='upper right', fontsize=14)

# draw peak point and save graph
from datetime import datetime
peak = 'Peak of revenue & orders'
ax2.annotate(peak, xy=(13, 7289+50),
             xytext=(13, 7289 + 300),fontsize=15, color='red',
             arrowprops=dict(facecolor='#FC5190',shrink=0.05),
             horizontalalignment='left', verticalalignment='top')
plt.show()
# plt.savefig('Trend of Revenue ($R1000) & Orders by Month', dpi=400, bbox_inches='tight')

# Part 3: Product Analysis
# Analyze the top 10 product categories and their percentage
category_counts = order_analysis_df['product_category_name_english'].value_counts().head(10)
total_orders = len(order_analysis_df)
category_percentage = (category_counts / total_orders) * 100


# Trend analysis - Count the number of orders per month for each category
trend_analysis = order_analysis_df.groupby(['product_category_name_english', 'purchase_month_year']).size().unstack().T
trend_analysis = trend_analysis[category_counts.index]  # Filter for only top 10 categories

# Displaying the results
print("Top 10 Categories and Counts:")
print(category_counts)
print("\nPercentage of Each Category:")
print(category_percentage)
print("\nTrend Analysis (sample data):")
print(trend_analysis.head())  # Can modify as needed to display more data


# Grouping products by order and creating a list of categories purchased together
pairs = order_analysis_df.groupby('order_id')['product_category_name_english'].apply(list)

# Filtering out orders with missing or undefined category names
pairs = pairs[pairs.apply(lambda x: all(isinstance(item, str) for item in x))]

# Creating a dictionary to count co-occurrences of categories
co_occurrence = defaultdict(int)

for product_list in pairs:
    # Create all possible combinations of 2 products within each order
    for combo in combinations(product_list, 2):
        if combo[0] != combo[1]:  # Ensure they are different categories
            sorted_combo = tuple(sorted(combo))  # Sorting the combination
            co_occurrence[sorted_combo] += 1

# Convert the co_occurrence dictionary to a DataFrame for analysis
co_occurrence_df = pd.DataFrame(list(co_occurrence.items()), columns=['Category_Pair', 'Co-occurrence_Count'])
co_occurrence_df.sort_values(by='Co-occurrence_Count', ascending=False, inplace=True)

# Displaying the top 10 category pairs
print("Top 10 Category Pairs based on Co-occurrence:")
print(co_occurrence_df.head(10))

# Part 4: Delivery Analysis

# Add new columns to express delivery dates
order_analysis_df = order_analysis_df.assign(
    order_approved_date=pd.to_datetime(order_analysis_df['order_approved_at']).dt.date,
    order_delivered_carrier_date=pd.to_datetime(order_analysis_df['order_delivered_carrier_date']).dt.date,
    order_delivered_customer_date=pd.to_datetime(order_analysis_df['order_delivered_customer_date']).dt.date,
    order_estimated_delivery_date=pd.to_datetime(order_analysis_df['order_estimated_delivery_date']).dt.date,
)
# Calculate delivery days by stages
order_analysis_df = order_analysis_df.assign(
    total_delivery_days=(order_analysis_df['order_delivered_customer_date']-order_analysis_df['purchase_date']).dt.days,
    purchase_approved_days=(order_analysis_df['order_approved_date']-order_analysis_df['purchase_date']).dt.days,
    approved_carrier_days=(order_analysis_df['order_delivered_carrier_date'] - order_analysis_df['order_approved_date']).dt.days,
    carrier_customer_days=(order_analysis_df['order_delivered_customer_date'] - order_analysis_df['order_delivered_carrier_date']).dt.days,
    delivery_days_delays=(order_analysis_df['order_delivered_customer_date']-order_analysis_df['order_estimated_delivery_date']).dt.days
)

average_delivery_days = order_analysis_df['total_delivery_days'].mean()
average_purchase_approved_days = order_analysis_df['purchase_approved_days'].mean()
average_approved_carrier_days = order_analysis_df['approved_carrier_days'].mean()
average_carrier_customer_days = order_analysis_df['carrier_customer_days'].mean()

print(f'The average delivery days is : {average_delivery_days:.2f}')
print(f'The average order approval days is : {average_purchase_approved_days:.2f}')
print(f'The average order-to-carrier days is : {average_approved_carrier_days:.2f}')
print(f'The average carrier-to-customer days is : {average_carrier_customer_days:.2f}')

# Draw the histogram of total_delivery_days, using visualization library
plot_histogram(order_analysis_df, column='total_delivery_days', bins = 30, color = 'skyblue')
# sns.histplot(order_analysis_df['total_delivery_days'], kde=False, color='skyblue', bins=30)
# plt.title(f'Histogram of total_delivery_days')
# plt.xlabel('total_delivery_days')
# plt.ylabel('Days')
# plt.show()

# Draw the histogram of delivery_days_delays, using visualization library
plot_histogram(order_analysis_df, column='delivery_days_delays', bins = 30, color = 'skyblue')
# sns.histplot(order_analysis_df['delivery_days_delays'], kde=False, color='skyblue', bins=30)
# plt.title(f'Histogram of delivery_days_delays')
# plt.xlabel('delivery_days_delays')
# plt.ylabel('Days')
# plt.show()

# Part 5: Review Analysis




