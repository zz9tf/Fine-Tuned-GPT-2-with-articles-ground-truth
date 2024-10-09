import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
from custom.io import load_nodes_jsonl

# ------------------ Data Loading Functions ------------------

def load_embeddings(file_path):
    """Load embeddings from a file if it exists."""
    if os.path.exists(file_path):
        print("Loading embeddings from file...")
        return np.load(file_path)
    else:
        print(f"File {file_path} not found.")
        sys.exit()

def load_nodes(file_path):
    """Load nodes using a custom function from JSONL file."""
    return load_nodes_jsonl(file_path)

# ------------------ TSNE and Data Preparation ------------------

def get_tsne_result(embeddings, n_components=2):
    """Generate TSNE result from embeddings."""
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(embeddings)
    return tsne_result

def create_tsne_dataframe(tsne_result, nodes):
    """Create a DataFrame with TSNE result and corresponding labels."""
    labels = [node.metadata['level'] for node in nodes]
    return pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': labels})

# ------------------ Plotting Functions ------------------

def plot_tsne_result(tsne_result_df, unique_labels, palette, nodes_file_name):
    """Plot TSNE result using a scatter plot with subplots for individual labels."""
    # Create a figure and a GridSpec layout
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 3, width_ratios=[2, 1, 1])

    # Create the large plot on the left (spanning both rows)
    ax_main = fig.add_subplot(gs[:, 0])
    sns.scatterplot(data=tsne_result_df, x='tsne_1', y='tsne_2', hue='label', palette=palette, ax=ax_main, alpha=0.5)
    ax_main.set_title('All Data')

    # Get x and y axis limits from the main plot
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    # Create the 2x2 grid on the right for individual label plots
    for i, label in enumerate(unique_labels[:4]):
        ax = fig.add_subplot(gs[i // 2, i % 2 + 1])
        label_data = tsne_result_df[tsne_result_df['label'] == label]
        sns.scatterplot(data=label_data, x='tsne_1', y='tsne_2', color=palette[i], ax=ax, alpha=0.8)

        # Set x and y limits to match the main plot
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_title(f'Label {label}')
        ax.legend([label])

    # Adjust layout and save the image
    plt.tight_layout()
    plt.savefig(f"tsne_result_{nodes_file_name}.png")
    plt.close()

# ------------------ Main Execution ------------------

def main():
    pid_num = 13
    # Load embeddings
    embeddings_file_path = f"../1_get_embedding_value/contexts/embeddings_pid_{pid_num}.npy"
    embeddings = load_embeddings(embeddings_file_path)

    # Load nodes
    cache_dir = os.path.abspath('../../.save')
    nodes_file_name = f"gpt-4o-batch-all-p_2_parser_ManuallyHierarchicalNodeParser_8165_gpu_V100_nodeNum_200_pid_{pid_num}.jsonl"
    nodes_file_path = os.path.join(cache_dir, nodes_file_name)
    nodes = load_nodes(nodes_file_path)

    # Generate TSNE result
    tsne_result = get_tsne_result(embeddings, 2)

    # Create a DataFrame with TSNE result and labels
    tsne_result_df = create_tsne_dataframe(tsne_result, nodes)

    # Define unique labels and color palette
    unique_labels = tsne_result_df['label'].unique()
    palette = sns.color_palette('Set1', len(unique_labels))

    # Plot and save the TSNE result
    plot_tsne_result(tsne_result_df, unique_labels, palette, nodes_file_name)

if __name__ == "__main__":
    main()
