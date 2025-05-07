import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_all_model_metrics(data_dict, figsize=(18, 10), save_path=None):
    """
    Plot both performance metrics (Accuracy, Precision, Recall, F1) and 
    time metrics (Train_Time, Infer_Time) for models in separate subplots.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing model data with keys:
        'model_name', 'Accuracy', 'Precision', 'Recall', 'F1', 'Train_Time', 'Infer_Time'.
    figsize : tuple
        Figure size as (width, height) in inches.
    save_path : str, optional
        If provided, saves the figure to the specified path.
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. Performance metrics chart (Accuracy, Precision, Recall, F1)
    ax1 = plt.subplot(2, 1, 1)
    performance_df = df.melt(id_vars=['model_name'], 
                             value_vars=['Accuracy', 'Precision', 'Recall', 'F1'],
                             var_name='Metric', value_name='Value')
    
    sns.barplot(x='model_name', y='Value', hue='Metric', data=performance_df, ax=ax1,
                palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    ax1.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xlabel('')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_ylim([0, 1])  # Performance metrics usually between 0-1
    ax1.legend(title='Metric', loc='lower right')
    
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.2f', padding=3)
    
    # 2. Time metrics chart (Train_Time and Infer_Time)
    ax2 = plt.subplot(2, 1, 2)
    
    # Bar width and positions
    bar_width = 0.35
    x = np.arange(len(df['model_name']))
    
    # Train_Time bars
    train_bars = ax2.bar(x - bar_width/2, df['Train_Time'], bar_width, 
                        label='Training Time', color='#3498db', alpha=0.8)
    
    # Infer_Time with secondary y-axis
    ax2_twin = ax2.twinx()
    infer_bars = ax2_twin.bar(x + bar_width/2, df['Infer_Time'], bar_width, 
                         label='Inference Time', color='#e74c3c', alpha=0.8)
    
    # Graph styling
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Training Time (seconds)', color='#3498db', fontsize=12)
    ax2_twin.set_ylabel('Inference Time (seconds)', color='#e74c3c', fontsize=12)
    
    # X-axis labels
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['model_name'])
    
    # Color settings
    ax2.tick_params(axis='y', colors='#3498db')
    ax2_twin.tick_params(axis='y', colors='#e74c3c')
    
    # Combine legend elements from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Subplot title
    ax2.set_title('Model Training and Inference Times', fontsize=14, fontweight='bold')
    
    # Add value labels to bars
    for bar in train_bars:
        height = bar.get_height()
        ax2.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in infer_bars:
        height = bar.get_height()
        ax2_twin.annotate(f'{height}',
                      xy=(bar.get_x() + bar.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')
    
    # Main title
    plt.suptitle('Model Performance and Time Metrics Comparison', fontsize=16, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax2_twin)


def plot_performance_metrics(data_dict, figsize=(12, 6), save_path=None):
    """
    Plot only performance metrics (Accuracy, Precision, Recall, F1) for different models.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing model data with keys:
        'model_name', 'Accuracy', 'Precision', 'Recall', 'F1'.
    figsize : tuple
        Figure size as (width, height) in inches.
    save_path : str, optional
        If provided, saves the figure to the specified path.
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transform data for seaborn
    performance_df = df.melt(id_vars=['model_name'], 
                          value_vars=['Accuracy', 'Precision', 'Recall', 'F1'],
                          var_name='Metric', value_name='Value')
    
    # Create plot
    sns.barplot(x='model_name', y='Value', hue='Metric', data=performance_df, ax=ax,
             palette=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Styling
    ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_ylim([0, 1])  # Performance metrics usually between 0-1
    ax.legend(title='Metric')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax
