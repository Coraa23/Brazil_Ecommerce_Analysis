# visualization/plots.py
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(data, column, bins=30, color='skyblue'):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column], kde=False, color=color, bins=bins)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_pairplot(data, hue=None):
    plt.figure(figsize=(10, 8))
    sns.pairplot(data, hue=hue)
    plt.title('Pairplot of Variables')
    plt.show()


