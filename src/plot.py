import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def plot_client_metrics(clients_acc: dict, save_path: str = 'client_metrics.png'):
    """
    Creates a grid chart (heatmap) where columns are clients and rows are metrics.
    Highlights the highest Test Accuracy with a red bounding box.

    Args:
        clients_acc (dict): Dictionary mapping client IDs to a list of metrics.
                            Format: {client_id: [train_acc, val_acc, test_acc]}
        save_path (str): The filepath to save the generated image.
    """
    # 1. Convert dictionary to a Pandas DataFrame
    # By passing the dict directly, the keys (Client IDs) become the columns!
    metrics = ['Train Accuracy', 'Val Accuracy', 'Test Accuracy']
    df = pd.DataFrame(clients_acc, index=metrics)
    
    # 2. Find the client with the highest Test Accuracy
    best_client_id = df.loc['Test Accuracy'].idxmax()
    best_test_acc = df.loc['Test Accuracy'].max()
    
    # Get the integer column index for plotting the highlight box
    best_col_idx = list(df.columns).index(best_client_id)
    
    # 3. Setup the plot dimensions based on number of clients
    # Scales width automatically so the cells don't get squished if you have many clients
    fig_width = max(8, len(df.columns) * 0.8)
    plt.figure(figsize=(fig_width, 4))
    
    # 4. Draw the Heatmap (acting as a colored table)
    # annot=True writes the actual numbers in the boxes
    # fmt=".2%" formats the decimals into percentages (e.g., 0.854 -> 85.40%)
    ax = sns.heatmap(
        df, 
        annot=True, 
        fmt=".2%", 
        cmap="Blues",       # Darker blue = higher accuracy
        cbar=False,         # Hide the color bar legend (unnecessary here)
        linewidths=1, 
        linecolor='white'
    )
    
    # 5. Highlight the Highest Test Accuracy
    # Matplotlib draws rectangles from the bottom-left corner of the target cell.
    # The 'Test Accuracy' row is at index 2 (Train=0, Val=1, Test=2)
    highlight_box = patches.Rectangle(
        (best_col_idx, 2),  # (x, y) coordinates
        width=1, 
        height=1, 
        fill=False, 
        edgecolor='red',    # Highlight color
        lw=4,               # Line width (thickness)
        zorder=10           # Ensure it draws on top of the heatmap
    )
    ax.add_patch(highlight_box)
    
    # 6. Formatting and Titles
    plt.title(f'Baseline Local Training Accuracies\n(Best Test Acc: Client {best_client_id} with {best_test_acc:.2%})', 
              fontsize=14, pad=15)
    plt.xlabel('Client ID', fontsize=12, labelpad=10)
    plt.ylabel('Evaluation Metric', fontsize=12, labelpad=10)
    
    # Ensure everything fits cleanly
    plt.tight_layout()
    
    # Save and display
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visualizer] Plot successfully saved to {save_path}")
    plt.show()