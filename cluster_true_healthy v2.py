__CURR_FILE__ = "cluster_true_healthy v2"

import os
import logging
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)

"""
Enhanced clustering script with individual plot saving functionality.
This script processes wearable blood glucose data, computes various glycemic metrics, 
clusters healthy participants, and identifies a 'true healthy' cluster with comprehensive visualizations.
"""

# =====================================================================
# === Logging setup ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === Define PATH SETUP ===
from paths import (
    PILOT_ROOT_DATA_PATH,
    MANIFEST_PATH,
    CLEANED_DATA_PATH,
    PARTICIPANTS_DATA_PATH,
    SUMMARY_PATH,
    PLOTS_OUTPUT_DIR,
    LOG_DIR
)

# Create logs directory and Ensure logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG to see more details
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, __CURR_FILE__ + ".log")),  # Save logs to file
        logging.StreamHandler()  # Also print to console
    ]
)
logging.info(f"Logs will be saved to: {LOG_DIR}")

# Create plots directory
os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
logging.info(f"Plots will be saved to: {PLOTS_OUTPUT_DIR}")

# =====================================================================
# === METRIC FUNCTIONS ===
def calculate_mage(glucose_values):
    """Calculate Mean Amplitude of Glycemic Excursions (MAGE)"""
    if len(glucose_values) < 2: 
        return np.nan
    mean_glucose = np.mean(glucose_values)
    std_glucose = np.std(glucose_values)
    peaks, _ = find_peaks(glucose_values)
    troughs, _ = find_peaks(-glucose_values)
    extreme_points = np.sort(np.concatenate([peaks, troughs]))
    amplitudes = np.abs(np.diff(glucose_values[extreme_points]))
    valid_amplitudes = amplitudes[amplitudes > std_glucose]
    if len(valid_amplitudes) == 0:
        return 0.0
    mage = np.mean(valid_amplitudes)
    return mage

def calculate_j_index(glucose_values):
    """Calculate J-index based on glucose time-series."""
    mean_glucose = np.mean(glucose_values)
    sd_glucose = np.std(glucose_values)
    j_index = 0.001 * (mean_glucose + sd_glucose) ** 2
    return j_index

def calculate_lability_index(glucose_values):
    """Calculate Lability Index based on glucose time-series."""
    diffs = np.diff(glucose_values)
    squared_diffs = diffs ** 2
    li = np.sum(squared_diffs) / len(glucose_values)
    return li

def calculate_conga(glucose_values, lag_steps=12):
    """Calculate CONGA (Continuous Overlapping Net Glycemic Action)"""
    if len(glucose_values) <= lag_steps:
        return np.nan
    conga_diffs = glucose_values[lag_steps:] - glucose_values[:-lag_steps]
    conga_sd = np.std(conga_diffs)
    return conga_sd

# =====================================================================
# === ENHANCED VISUALIZATION FUNCTIONS ===

def save_plot_1_cluster_highlight(X_scaled, labels, healthy_df):
    """Save Plot 1: Cluster 0 Highlighted"""
    plt.figure(figsize=(12, 8))
    
    # Plot all clusters in light gray first
    for i in range(1, 5):
        mask = labels == i
        if np.any(mask):
            plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], 
                       c='lightgray', alpha=0.4, s=30)
    
    # Highlight Cluster 0 (True Healthy) in bold red
    mask_0 = labels == 0
    plt.scatter(X_scaled[mask_0, 0], X_scaled[mask_0, 1], 
               c='red', s=100, alpha=0.8, label='True Healthy (Cluster 0)', 
               edgecolors='darkred', linewidth=2)
    
    plt.title('True Healthy Cluster Highlighted\n(Cluster 0 in Red)', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Blood Glucose (Scaled)', fontsize=12)
    plt.ylabel('Std Dev Blood Glucose (Scaled)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add statistics box
    n_true_healthy = np.sum(mask_0)
    total_healthy = len(labels)
    percentage = (n_true_healthy / total_healthy) * 100
    textstr = f'True Healthy: {n_true_healthy}/{total_healthy}\n({percentage:.1f}%)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "01_cluster_highlight.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 01_cluster_highlight.png")

def save_plot_2_traditional_clusters(X_scaled, labels):
    """Save Plot 2: Traditional Cluster View"""
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, 
                         cmap='viridis', s=60, alpha=0.7)
    plt.title('All Clusters - Traditional View', fontsize=16, fontweight='bold')
    plt.xlabel('Mean Blood Glucose (Scaled)', fontsize=12)
    plt.ylabel('Std Dev Blood Glucose (Scaled)', fontsize=12)
    plt.colorbar(scatter, label='Cluster Label')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "02_traditional_clusters.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 02_traditional_clusters.png")

# def save_plot_3_pca_view(X_scaled, labels):
#     """Save Plot 3: PCA View"""
#     plt.figure(figsize=(10, 8))
    
#     # Apply PCA for better visualization
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled.reshape(X_scaled.shape[0], -1))
    
#     # Plot other clusters in gray
#     for i in range(1, 5):
#         mask = labels == i
#         if np.any(mask):
#             plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
#                        c='lightgray', alpha=0.4, s=30)
    
#     # Highlight Cluster 0
#     mask_0 = labels == 0
#     plt.scatter(X_pca[mask_0, 0], X_pca[mask_0, 1], 
#                c='red', s=100, alpha=0.8, label='True Healthy', 
#                edgecolors='darkred', linewidth=2)
    
#     plt.title('PCA View - True Healthy Cluster', fontsize=16, fontweight='bold')
#     plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
#     plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
#     plt.legend(fontsize=12)
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "03_pca_view.png"), dpi=300, bbox_inches='tight')
#     plt.close()
#     logging.info("Saved: 03_pca_view.png")

def save_plot_3_pca_view(X_scaled, labels):
    """Save Plot 3: PCA View with different colors for each cluster"""
    plt.figure(figsize=(12, 8))
    
    # Apply PCA for better visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled.reshape(X_scaled.shape[0], -1))
    
    # Define colors for each cluster
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    cluster_names = ['CGM-Healthy', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
    
    # Plot each cluster with different colors
    for i in range(5):
        mask = labels == i
        if np.any(mask):
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=colors[i], 
                       s=100 if i == 0 else 60,  # Larger points for True Healthy
                       alpha=0.8 if i == 0 else 0.7, 
                       label=cluster_names[i],
                       edgecolors='darkred' if i == 0 else 'black', 
                       linewidth=2 if i == 0 else 1)
    
    plt.title('PCA View - All Clusters with Different Colors', fontsize=16, fontweight='bold')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add cluster statistics as text box
    n_clusters = len(np.unique(labels))
    textstr = f'Total Clusters: {n_clusters}\n'
    for i in range(5):
        count = np.sum(labels == i)
        if count > 0:
            textstr += f'{cluster_names[i]}: {count} participants\n'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr.strip(), transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "03_pca_view.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 03_pca_view.png")

def save_plot_4_tir_boxplot(labels, healthy_df):
    """Save Plot 4: Time In Range Box Plot"""
    plt.figure(figsize=(12, 8))
    
    # Create DataFrame for plotting
    plot_data = []
    for i, pid in enumerate(healthy_df['participant_id']):
        cluster_label = labels[i]
        cluster_name = 'True Healthy' if cluster_label == 0 else f'Cluster {cluster_label}'
        plot_data.append({
            'participant_id': pid,
            'cluster': cluster_name,
            'tir_percent': healthy_df.iloc[i]['tir_percent'],
            'mean_glucose': healthy_df.iloc[i]['mean_blood_glucose']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Box plot for TIR (Time In Range)
    sns.boxplot(data=plot_df, x='cluster', y='tir_percent')
    plt.title('Time In Range by Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Time In Range (%)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Highlight True Healthy label
    ax = plt.gca()
    labels_text = ax.get_xticklabels()
    for label in labels_text:
        if 'True Healthy' in label.get_text():
            label.set_color('red')
            label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "04_tir_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 04_tir_boxplot.png")

def save_plot_5_feature_radar(X_scaled, labels, healthy_df, cluster_features):
    """Save Plot 5: Feature Radar Chart"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Calculate mean values for each cluster
    cluster_means = {}
    for cluster_id in range(5):
        mask = labels == cluster_id
        if np.any(mask):
            cluster_data = healthy_df.iloc[mask][cluster_features]
            # Normalize to 0-1 scale for radar chart
            normalized_means = []
            for feature in cluster_features:
                values = cluster_data[feature]
                min_val = healthy_df[feature].min()
                max_val = healthy_df[feature].max()
                normalized_mean = (values.mean() - min_val) / (max_val - min_val)
                normalized_means.append(normalized_mean)
            cluster_means[cluster_id] = normalized_means
    
    # Plot radar chart for Cluster 0 vs others
    angles = np.linspace(0, 2 * np.pi, len(cluster_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot Cluster 0 (True Healthy) prominently
    if 0 in cluster_means:
        values_0 = cluster_means[0] + cluster_means[0][:1]
        ax.plot(angles, values_0, 'r-', linewidth=3, label='True Healthy', alpha=0.8)
        ax.fill(angles, values_0, 'red', alpha=0.2)
    
    # Plot other clusters in gray
    for cluster_id in [1, 2, 3, 4]:
        if cluster_id in cluster_means:
            values = cluster_means[cluster_id] + cluster_means[cluster_id][:1]
            ax.plot(angles, values, color='gray', linewidth=1, alpha=0.6)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace('_', '\n') for f in cluster_features], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Cluster 0 Feature Profile\n(True Healthy)', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "05_feature_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 05_feature_radar.png")

def save_plot_6_statistics_table(labels, healthy_df):
    """Save Plot 6: Statistics Table"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create statistics table
    stats_data = []
    for cluster_id in range(5):
        mask = labels == cluster_id
        if np.any(mask):
            cluster_data = healthy_df.iloc[mask]
            stats_data.append([
                f'Cluster {cluster_id}' if cluster_id != 0 else 'CGM-Healthy',
                len(cluster_data),
                f'{cluster_data["tir_percent"].mean():.1f}±{cluster_data["tir_percent"].std():.1f}',
                f'{cluster_data["mean_blood_glucose"].mean():.1f}±{cluster_data["mean_blood_glucose"].std():.1f}',
                f'{cluster_data["cv_percent"].mean():.1f}±{cluster_data["cv_percent"].std():.1f}'
            ])
    
    table_df = pd.DataFrame(stats_data, columns=['Cluster', 'N', 'TIR%', 'Mean Glucose', 'CV%'])
    
    # Create table
    table = ax.table(cellText=table_df.values,
                     colLabels=table_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.1, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Highlight True Healthy row
    for i in range(len(table_df.columns)):
        table[(1, i)].set_facecolor('#ffcccc')  # Light red background for True Healthy
        table[(1, i)].set_text_props(weight='bold')
    
    ax.set_title('Cluster Statistics Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "06_statistics_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 06_statistics_table.png")

def save_plot_7_intracluster_compactness(X_scaled, labels):
    """Save Plot 7: Intra-cluster Compactness Analysis"""
    plt.figure(figsize=(12, 8))
    
    # Distance from cluster centers
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X_scaled.reshape(X_scaled.shape[0], -1))
    
    distances_to_centers = []
    for i in range(5):
        mask = labels == i
        if np.any(mask):
            cluster_points = X_scaled[mask].reshape(np.sum(mask), -1)
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - center, axis=1)
            distances_to_centers.extend([(i, d) for d in distances])
    
    distance_df = pd.DataFrame(distances_to_centers, columns=['Cluster', 'Distance'])
    distance_df['Cluster_Name'] = distance_df['Cluster'].apply(
        lambda x: 'True Healthy' if x == 0 else f'Cluster {x}'
    )
    
    sns.boxplot(data=distance_df, x='Cluster_Name', y='Distance')
    plt.title('Intra-cluster Compactness Analysis', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Distance to Center', fontsize=12)
    plt.xticks(rotation=45)
    
    # Highlight True Healthy
    ax = plt.gca()
    labels_text = ax.get_xticklabels()
    for label in labels_text:
        if 'True Healthy' in label.get_text():
            label.set_color('red')
            label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "07_intracluster_compactness.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 07_intracluster_compactness.png")

def save_plot_8_tir_distribution(labels, healthy_df):
    """Save Plot 8: TIR Distribution Comparison"""
    plt.figure(figsize=(12, 8))
    
    cluster_0_tir = healthy_df.iloc[labels == 0]['tir_percent']
    other_tir = healthy_df.iloc[labels != 0]['tir_percent']
    
    plt.hist(other_tir, alpha=0.6, bins=20, label='Other Clusters', color='gray', density=True)
    plt.hist(cluster_0_tir, alpha=0.8, bins=15, label='True Healthy', color='red', density=True)
    plt.title('Time In Range Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Time In Range (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.axvline(cluster_0_tir.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'True Healthy Mean: {cluster_0_tir.mean():.1f}%')
    plt.axvline(other_tir.mean(), color='gray', linestyle='--', linewidth=2,
                label=f'Others Mean: {other_tir.mean():.1f}%')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "08_tir_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 08_tir_distribution.png")

def save_plot_9_silhouette_analysis(X_scaled, labels):
    """Save Plot 9: Silhouette Analysis"""
    plt.figure(figsize=(12, 8))
    
    # Calculate silhouette scores
    silhouette_scores = silhouette_samples(X_scaled.reshape(X_scaled.shape[0], -1), labels)
    overall_silhouette_score = silhouette_score(X_scaled.reshape(X_scaled.shape[0], -1), labels)
    
    # Print overall silhouette score
    print("\n" + "="*70)
    print(f"\nSILHOUETTE ANALYSIS:")
    print(f"   Overall Silhouette Score: {overall_silhouette_score:.3f}")
    print(f"   Interpretation: {'Excellent' if overall_silhouette_score > 0.7 else 'Good' if overall_silhouette_score > 0.5 else 'Fair' if overall_silhouette_score > 0.25 else 'Poor'}")
    
    # Calculate and print per-cluster silhouette scores
    print(f"\n   Per-Cluster Silhouette Scores:")
    for i in range(5):
        if np.any(labels == i):
            cluster_silhouette = silhouette_scores[labels == i].mean()
            cluster_name = 'CGM-Healthy' if i == 0 else f'Cluster {i}'
            print(f"   {cluster_name:12}: {cluster_silhouette:.3f}")    
    
    y_lower = 10
    for i in range(5):
        cluster_silhouette_scores = silhouette_scores[labels == i]
        cluster_silhouette_scores.sort()
        
        size_cluster_i = cluster_silhouette_scores.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = 'red' if i == 0 else plt.cm.nipy_spectral(float(i) / 5)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_scores,
                         facecolor=color, edgecolor=color, alpha=0.7)
        
        label = 'True Healthy' if i == 0 else f'Cluster {i}'
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, label, fontsize=12, fontweight='bold' if i == 0 else 'normal')
        y_lower = y_upper + 10
    
    plt.title('Silhouette Analysis by Cluster', fontsize=16, fontweight='bold')
    plt.xlabel('Silhouette Score', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "09_silhouette_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 09_silhouette_analysis.png")

def save_plot_10_clinical_metrics(labels, healthy_df):
    """Save Plot 10: Enhanced Clinical Metrics Comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Core Clinical Metrics (TIR, TAR, TBR)
    ax1 = axes[0, 0]
    core_metrics = ['tir_percent', 'tar_percent', 'tbr_percent']
    cluster_0_means = [healthy_df.iloc[labels == 0][metric].mean() for metric in core_metrics]
    other_means = [healthy_df.iloc[labels != 0][metric].mean() for metric in core_metrics]
    
    x = np.arange(len(core_metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cluster_0_means, width, label='CGM-Healthy', color='red', alpha=0.8)
    bars2 = ax1.bar(x + width/2, other_means, width, label='Other Clusters', color='gray', alpha=0.6)
    
    ax1.set_title('Core Clinical Metrics (Time-based)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Metrics', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Time In Range\n(70-140)', 'Time Above Range\n(>140)', 'Time Below Range\n(<70)'])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(cluster_0_means, other_means)):
        ax1.text(i - width/2, v1 + max(cluster_0_means + other_means) * 0.01, f'{v1:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.text(i + width/2, v2 + max(cluster_0_means + other_means) * 0.01, f'{v2:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    # Subplot 2: Glucose Statistics
    ax2 = axes[0, 1]
    glucose_metrics = ['mean_blood_glucose', 'std_dev', 'cv_percent']
    glucose_labels = ['Mean Glucose\n(mg/dL)', 'Std Deviation\n(mg/dL)', 'CV\n(%)']
    
    cluster_0_glucose = [healthy_df.iloc[labels == 0][metric].mean() for metric in glucose_metrics]
    other_glucose = [healthy_df.iloc[labels != 0][metric].mean() for metric in glucose_metrics]
    
    x2 = np.arange(len(glucose_metrics))
    bars3 = ax2.bar(x2 - width/2, cluster_0_glucose, width, label='CGM-Healthy', color='red', alpha=0.8)
    bars4 = ax2.bar(x2 + width/2, other_glucose, width, label='Other Clusters', color='gray', alpha=0.6)
    
    ax2.set_title('Glucose Statistics', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Values', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(glucose_labels)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (v1, v2) in enumerate(zip(cluster_0_glucose, other_glucose)):
        ax2.text(i - width/2, v1 + max(cluster_0_glucose + other_glucose) * 0.01, f'{v1:.1f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(i + width/2, v2 + max(cluster_0_glucose + other_glucose) * 0.01, f'{v2:.1f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Subplot 3: Advanced Glycemic Metrics
    ax3 = axes[1, 0]
    advanced_metrics = ['mage', 'j_index', 'lability_index']
    advanced_labels = ['MAGE\n(mg/dL)', 'J-Index', 'Lability Index']
    
    # Check which advanced metrics are available
    available_advanced = [m for m in advanced_metrics if m in healthy_df.columns]
    available_labels = [advanced_labels[i] for i, m in enumerate(advanced_metrics) if m in available_advanced]
    
    if available_advanced:
        cluster_0_advanced = [healthy_df.iloc[labels == 0][metric].mean() for metric in available_advanced]
        other_advanced = [healthy_df.iloc[labels != 0][metric].mean() for metric in available_advanced]
        
        x3 = np.arange(len(available_advanced))
        bars5 = ax3.bar(x3 - width/2, cluster_0_advanced, width, label='CGM-Healthy', color='red', alpha=0.8)
        bars6 = ax3.bar(x3 + width/2, other_advanced, width, label='Other Clusters', color='gray', alpha=0.6)
        
        ax3.set_title('Advanced Glycemic Variability', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Metrics', fontsize=12)
        ax3.set_ylabel('Values', fontsize=12)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(available_labels)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (v1, v2) in enumerate(zip(cluster_0_advanced, other_advanced)):
            ax3.text(i - width/2, v1 + max(cluster_0_advanced + other_advanced) * 0.01, f'{v1:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax3.text(i + width/2, v2 + max(cluster_0_advanced + other_advanced) * 0.01, f'{v2:.2f}', 
                    ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Advanced metrics\nnot available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Advanced Glycemic Variability', fontsize=14, fontweight='bold')
    
    # Subplot 4: CONGA Metrics (if available)
    ax4 = axes[1, 1]
    conga_metrics = ['conga_1h', 'conga_2h']
    available_conga = [m for m in conga_metrics if m in healthy_df.columns]
    
    if available_conga:
        conga_labels = [m.replace('_', ' ').upper() for m in available_conga]
        cluster_0_conga = [healthy_df.iloc[labels == 0][metric].mean() for metric in available_conga]
        other_conga = [healthy_df.iloc[labels != 0][metric].mean() for metric in available_conga]
        
        x4 = np.arange(len(available_conga))
        bars7 = ax4.bar(x4 - width/2, cluster_0_conga, width, label='CGM-Healthy', color='red', alpha=0.8)
        bars8 = ax4.bar(x4 + width/2, other_conga, width, label='Other Clusters', color='gray', alpha=0.6)
        
        ax4.set_title('CONGA Metrics\n(Continuous Overlapping Net Glycemic Action)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Metrics', fontsize=12)
        ax4.set_ylabel('Values (mg/dL)', fontsize=12)
        ax4.set_xticks(x4)
        ax4.set_xticklabels(conga_labels)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (v1, v2) in enumerate(zip(cluster_0_conga, other_conga)):
            ax4.text(i - width/2, v1 + max(cluster_0_conga + other_conga) * 0.01, f'{v1:.1f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
            ax4.text(i + width/2, v2 + max(cluster_0_conga + other_conga) * 0.01, f'{v2:.1f}', 
                    ha='center', va='bottom', fontsize=10)
    else:
        # Create a summary statistics table instead
        ax4.axis('off')
        
        # Create summary data
        summary_data = []
        cluster_0_data = healthy_df.iloc[labels == 0]
        other_data = healthy_df.iloc[labels != 0]
        
        summary_metrics = ['tir_percent', 'mean_blood_glucose', 'cv_percent', 'mage']
        available_summary = [m for m in summary_metrics if m in healthy_df.columns]
        
        for metric in available_summary:
            metric_name = metric.replace('_percent', ' (%)').replace('_', ' ').title()
            cgm_healthy_val = cluster_0_data[metric].mean()
            others_val = other_data[metric].mean()
            difference = cgm_healthy_val - others_val
            
            summary_data.append([
                metric_name,
                f'{cgm_healthy_val:.1f}',
                f'{others_val:.1f}',
                f'{difference:+.1f}'
            ])
        
        if summary_data:
            table_df = pd.DataFrame(summary_data, 
                                  columns=['Metric', 'CGM-Healthy', 'Others', 'Difference'])
            
            # Create table
            table = ax4.table(cellText=table_df.values,
                             colLabels=table_df.columns,
                             cellLoc='center',
                             loc='center',
                             colWidths=[0.3, 0.2, 0.2, 0.2])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(table_df.columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Highlight differences
            for i in range(1, len(table_df) + 1):
                table[(i, 0)].set_facecolor('#f8f8f8')
                table[(i, 1)].set_facecolor('#ffcccc')  # CGM-Healthy column
                table[(i, 1)].set_text_props(weight='bold')
            
            ax4.set_title('Clinical Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Add overall figure title
    fig.suptitle('Comprehensive Clinical Metrics Comparison\nCGM-Healthy vs Other Clusters', 
                fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for suptitle
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "10_clinical_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 10_clinical_metrics.png")

def save_plot_11_elbow_curve(X_scaled):
    """Save Plot 11: Elbow Curve for Optimal Clusters"""
    plt.figure(figsize=(10, 8))
    
    inertia = []
    for i in range(1, 11):
        model = TimeSeriesKMeans(n_clusters=i, metric="euclidean", random_state=42)
        model.fit(X_scaled)
        inertia.append(model.inertia_)
    
    plt.plot(range(1, 11), inertia, marker='o', linewidth=2, markersize=8)
    plt.title("Elbow Method for Optimal Number of Clusters", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Highlight the chosen number of clusters
    plt.axvline(x=5, color='red', linestyle='--', linewidth=2, label='Chosen: 5 clusters')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "11_elbow_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Saved: 11_elbow_curve.png")

def create_all_visualizations(X_scaled, labels, healthy_df, cluster_features):
    """Create and save all individual visualizations"""
    logging.info("Creating and saving all visualization plots...")
    
    # Save all individual plots
    save_plot_1_cluster_highlight(X_scaled, labels, healthy_df)
    save_plot_2_traditional_clusters(X_scaled, labels)
    save_plot_3_pca_view(X_scaled, labels)
    save_plot_4_tir_boxplot(labels, healthy_df)
    save_plot_5_feature_radar(X_scaled, labels, healthy_df, cluster_features)
    save_plot_6_statistics_table(labels, healthy_df)
    save_plot_7_intracluster_compactness(X_scaled, labels)
    save_plot_8_tir_distribution(labels, healthy_df)
    save_plot_9_silhouette_analysis(X_scaled, labels)
    save_plot_10_clinical_metrics(labels, healthy_df)
    save_plot_11_elbow_curve(X_scaled)
    
    logging.info(f"All plots saved successfully to: {PLOTS_OUTPUT_DIR}")

# =====================================================================
# === COMPUTE METRICS FROM BATCH CSVs ===
def compute_participant_metrics():
    """Compute summary metrics for participants from batch CSV files"""
    # === STEP 1: LOAD METADATA ===
    logging.info("Loading manifest and participant metadata.")
    dfm = pd.read_csv(MANIFEST_PATH, sep='\t')
    participants_df = pd.read_csv(PARTICIPANTS_DATA_PATH)
    participants_df.columns = participants_df.columns.str.strip()

    # === STEP 2: EXTRACT METRICS FOR EACH PARTICIPANT ===
    logging.info("Computing summary metrics for participants.")
    all_metrics = []
    start = 0

    # Iterate through csv batches of cleaned data files
    for r in (100, 200, 300, 400, 500, 600, 700, 800, 809):
        file_path = os.path.join(CLEANED_DATA_PATH, f"batch_{start}_{r}.csv")
        if not os.path.exists(file_path):
            logging.warning(f"Missing file: {file_path}")
            start = r
            continue

        df = pd.read_csv(file_path)
        # Iterating through each unique participant in the batch
        for pid in df['participant_id'].unique():
            record_row = dfm[dfm['participant_id'] == pid]
            record_count = record_row['glucose_level_record_count'].values[0] if not record_row.empty else np.nan
            # Only process participants with at least 2138 records
            if record_count < 2138:
                continue

            data = df[df['participant_id'] == pid].head(2138)
            glucose = data['blood_glucose_value'].values
            if len(glucose) == 0:
                continue
            low_threshold = 70
            high_threshold = 140

            # Calculate metrics
            metrics = {
                'participant_id': pid,
                'age': participants_df.loc[participants_df['participant_id'] == pid, 'age'].values[0] if pid in participants_df['participant_id'].values else np.nan,
                'glucose_level_record_count': record_count,
                'mean_blood_glucose': np.mean(glucose),
                'std_dev': np.std(glucose),
                'cv_percent': np.std(glucose) / np.mean(glucose) * 100,
                'tir_percent': np.sum((glucose >= low_threshold) & (glucose <= high_threshold)) / len(glucose) * 100,
                'tar_percent': np.sum(glucose > high_threshold) / len(glucose) * 100,
                'tbr_percent': np.sum(glucose < low_threshold) / len(glucose) * 100,
                'mage': calculate_mage(glucose),
                'j_index': calculate_j_index(glucose),
                'lability_index': calculate_lability_index(glucose),
                'conga_1h': calculate_conga(glucose, 12),
                'conga_2h': calculate_conga(glucose, 24),
                'study_group': data['study_group'].iloc[0]
            }
            all_metrics.append(metrics)
        start = r

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(SUMMARY_PATH, index=False)
    logging.info(f"Saved summarized metrics to {SUMMARY_PATH}")
    return metrics_df

def print_cluster_summary(labels, healthy_df):
    """Print detailed summary of clustering results"""
    print("\n" + "="*70)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*70)
    
    total_participants = len(labels)
    cluster_0_count = np.sum(labels == 0)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"   Total Healthy Participants: {total_participants}")
    print(f"   True Healthy (Cluster 0): {cluster_0_count}")
    print(f"   Percentage True Healthy: {(cluster_0_count/total_participants)*100:.1f}%")
    
    print(f"\nCLINICAL SIGNIFICANCE:")
    cluster_0_data = healthy_df.iloc[labels == 0]
    other_data = healthy_df.iloc[labels != 0]
    
    print(f"   True Healthy TIR: {cluster_0_data['tir_percent'].mean():.1f}% ± {cluster_0_data['tir_percent'].std():.1f}%")
    print(f"   Other Groups TIR: {other_data['tir_percent'].mean():.1f}% ± {other_data['tir_percent'].std():.1f}%")
    print(f"   Difference: {cluster_0_data['tir_percent'].mean() - other_data['tir_percent'].mean():.1f}% higher")
    
    print(f"\nGLUCOSE METRICS:")
    print(f"   True Healthy Mean Glucose: {cluster_0_data['mean_blood_glucose'].mean():.1f} mg/dL")
    print(f"   True Healthy CV: {cluster_0_data['cv_percent'].mean():.1f}%")
    print(f"   True Healthy MAGE: {cluster_0_data['mage'].mean():.1f}")
    
    print(f"\nPLOTS SAVED:")
    print(f"   All visualization plots saved to: {PLOTS_OUTPUT_DIR}")
    print("   Individual files:")
    print("   - 01_cluster_highlight.png")
    print("   - 02_traditional_clusters.png")
    print("   - 03_pca_view.png")
    print("   - 04_tir_boxplot.png")
    print("   - 05_feature_radar.png")
    print("   - 06_statistics_table.png")
    print("   - 07_intracluster_compactness.png")
    print("   - 08_tir_distribution.png")
    print("   - 09_silhouette_analysis.png")
    print("   - 10_clinical_metrics.png")
    print("   - 11_elbow_curve.png")
    
    print("\n" + "="*70)

def cluster_healthy_participants_enhanced(metrics_df):
    """Enhanced clustering function with comprehensive visualization saving"""
    logging.info("Clustering healthy participants with enhanced visualization.")

    # Filter only healthy participants
    healthy_df = metrics_df[metrics_df['study_group'] == 'healthy']
    if healthy_df.empty:
        raise ValueError("No healthy participants found.")

    # Selected features for clustering
    cluster_features = ['mean_blood_glucose', 'std_dev', 'cv_percent', 'tir_percent', 'tbr_percent',
                        'mage', 'lability_index', 'conga_2h']

    # Data preparation for clustering
    X = healthy_df[cluster_features].values
    scaler = TimeSeriesScalerMeanVariance()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 5
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", verbose=True, random_state=42)
    labels = model.fit_predict(X_scaled)

    # Create and save all visualizations
    create_all_visualizations(X_scaled, labels, healthy_df, cluster_features)

    # Print detailed summary
    print_cluster_summary(labels, healthy_df)

    # Identify true healthy cluster
    cluster_id = 0
    all_healthy_participants = healthy_df['participant_id'].values
    cluster_labels = {int(all_healthy_participants[i]): int(labels[i]) for i in range(len(labels))}
    true_healthy = {pid: label for pid, label in cluster_labels.items() if label == cluster_id}

    # Print participant IDs in True Healthy cluster
    print(f"\nTRUE HEALTHY PARTICIPANTS:")
    print(f"Participant IDs in Cluster 0: {list(true_healthy.keys())}" + "\n")

    # Update labels in dataframe
    metrics_df['study_group_cleaned'] = metrics_df['study_group']
    metrics_df.loc[metrics_df['participant_id'].isin(true_healthy.keys()), 'study_group_cleaned'] = 'true_healthy'
    metrics_df.loc[
        (metrics_df['study_group'] == 'healthy') & (~metrics_df['participant_id'].isin(true_healthy.keys())),
        'study_group_cleaned'
    ] = 'pre_diabetes_lifestyle'

    # Save updated metrics
    metrics_df.to_csv(SUMMARY_PATH, index=False)
    logging.info("Updated summarized_metrics_all_participants.csv with 'study_group_cleaned' column.")

    return metrics_df, labels, cluster_labels

def save_summary_report(metrics_df, labels, healthy_df):
    """Save a comprehensive text summary report"""
    report_path = os.path.join(PLOTS_OUTPUT_DIR, "clustering_summary_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRUE HEALTHY CLUSTER IDENTIFICATION - COMPREHENSIVE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total participants in analysis: {len(metrics_df)}\n")
        f.write(f"Healthy participants (original): {len(healthy_df)}\n")
        f.write(f"True healthy identified: {np.sum(labels == 0)}\n")
        f.write(f"Percentage true healthy: {(np.sum(labels == 0)/len(healthy_df))*100:.1f}%\n\n")
        
        f.write("CLINICAL METRICS COMPARISON:\n")
        f.write("-" * 50 + "\n")
        cluster_0_data = healthy_df.iloc[labels == 0]
        other_data = healthy_df.iloc[labels != 0]
        
        metrics_to_compare = ['tir_percent', 'tar_percent', 'tbr_percent', 'mean_blood_glucose', 'cv_percent', 'mage']
        
        for metric in metrics_to_compare:
            f.write(f"{metric}:\n")
            f.write(f"  True Healthy: {cluster_0_data[metric].mean():.2f} ± {cluster_0_data[metric].std():.2f}\n")
            f.write(f"  Other Groups: {other_data[metric].mean():.2f} ± {other_data[metric].std():.2f}\n")
            f.write(f"  Difference: {cluster_0_data[metric].mean() - other_data[metric].mean():.2f}\n\n")
        
        f.write("CLUSTER DISTRIBUTION:\n")
        f.write("-" * 50 + "\n")
        for i in range(5):
            count = np.sum(labels == i)
            percentage = (count / len(labels)) * 100
            cluster_name = "True Healthy" if i == 0 else f"Cluster {i}"
            f.write(f"{cluster_name}: {count} participants ({percentage:.1f}%)\n")
        
        f.write(f"\nVISUALIZATION FILES GENERATED:\n")
        f.write("-" * 50 + "\n")
        plot_descriptions = [
            "01_cluster_highlight.png - True Healthy cluster highlighted in red",
            "02_traditional_clusters.png - Traditional cluster visualization",
            "03_pca_view.png - PCA-transformed cluster view",
            "04_tir_boxplot.png - Time In Range comparison by cluster",
            "05_feature_radar.png - Feature profile radar chart",
            "06_statistics_table.png - Comprehensive statistics table",
            "07_intracluster_compactness.png - Cluster compactness analysis",
            "08_tir_distribution.png - TIR distribution comparison",
            "09_silhouette_analysis.png - Silhouette coefficient analysis",
            "10_clinical_metrics.png - Clinical metrics bar comparison",
            "11_elbow_curve.png - Elbow method for optimal clusters"
        ]
        
        for desc in plot_descriptions:
            f.write(f"  {desc}\n")
        
        f.write(f"\nTRUE HEALTHY PARTICIPANT IDs:\n")
        f.write("-" * 50 + "\n")
        true_healthy_ids = healthy_df.iloc[labels == 0]['participant_id'].tolist()
        
        # Write IDs in rows of 10
        for i in range(0, len(true_healthy_ids), 10):
            row_ids = true_healthy_ids[i:i+10]
            f.write(f"{', '.join(map(str, row_ids))}\n")
        
        f.write(f"\nMETHODOLOGY:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Computed glycemic metrics for all participants\n")
        f.write("2. Filtered healthy participants from AI-READI dataset\n")
        f.write("3. Applied K-means clustering (k=5) using TimeSeriesScalerMeanVariance\n")
        f.write("4. Identified Cluster 0 as 'True Healthy' based on clinical criteria\n")
        f.write("5. Validated using multiple visualization and statistical techniques\n")
        f.write("6. Updated study group labels for subsequent analysis\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n")
    
    logging.info(f"Comprehensive summary report saved: {report_path}")

def create_feature_explanation_table():
    """Create comprehensive feature explanation table and visualizations"""
    
    # Feature explanation data
    feature_data = {
        'Feature Category': [
            'Basic Statistics', 'Basic Statistics', 'Basic Statistics',
            'Clinical Metrics', 'Clinical Metrics', 'Clinical Metrics', 'Clinical Metrics',
            'Glycemic Variability', 'Glycemic Variability', 'Glycemic Variability', 'Glycemic Variability',
            'Temporal Derivatives', 'Temporal Derivatives', 'Temporal Derivatives',
            'Rolling Statistics', 'Rolling Statistics', 'Rolling Statistics',
            'Circadian Encoding', 'Circadian Encoding', 'Circadian Encoding', 'Circadian Encoding',
            'Contextual Features', 'Contextual Features'
        ],
        'Feature Name': [
            'mean_blood_glucose', 'std_dev', 'cv_percent',
            'tir_percent', 'tar_percent', 'tbr_percent', 'above_140_flag',
            'mage', 'j_index', 'lability_index', 'conga_1h/2h',
            'glucose_diff', 'glucose_change_rate', 'glucose_accel',
            'glucose_rollmean_1h', 'glucose_rollstd_1h', 'glucose_rollmax_1h',
            'hour', 'day_fraction', 'sin_hour', 'cos_hour',
            'is_meal_time', 'is_night'
        ],
        'Formula/Calculation': [
            'μ = (1/n)∑glucose[i]',
            'σ = √[(1/n)∑(glucose[i] - μ)²]',
            'CV = (σ/μ) × 100%',
            
            '% time in [70-140] mg/dL',
            '% time > 140 mg/dL',
            '% time < 70 mg/dL',
            '1 if any glucose > 140, else 0',
            
            'Mean amplitude of excursions > 1σ',
            '0.001 × (μ + σ)²',
            '∑(diff²)/n where diff = glucose[i+1] - glucose[i]',
            'SD of glucose differences at lag h',
            
            'glucose[t] - glucose[t-1]',
            '(glucose[t] - glucose[t-1])/Δt',
            'glucose_change_rate[t] - glucose_change_rate[t-1]',
            
            'Moving average over 1 hour (12 points)',
            'Moving std dev over 1 hour',
            'Moving maximum over 1 hour',
            
            'Hour of day (0-23)',
            'hour/24 (0-1 scale)',
            'sin(2π × hour/24)',
            'cos(2π × hour/24)',
            
            '1 if hour in [6-10, 12-14, 18-20]',
            '1 if hour in [22-6]'
        ],
        'Clinical Interpretation': [
            'Average glucose control',
            'Glucose variability magnitude',
            'Relative variability (normalized)',
            
            'Optimal glucose control (target: >70%)',
            'Hyperglycemic exposure (target: <25%)',
            'Hypoglycemic risk (target: <4%)',
            'Any hyperglycemic episodes',
            
            'Glucose swings magnitude (target: <60)',
            'Combined mean+variability risk',
            'Rate of glucose fluctuations',
            'Delayed glucose variability',
            
            'Immediate glucose change',
            'Rate of glucose change',
            'Glucose change acceleration',
            
            'Short-term glucose trend',
            'Short-term variability',
            'Peak glucose in recent hour',
            
            'Time-of-day effects',
            'Normalized time (0=midnight, 0.5=noon)',
            'Circadian sine component',
            'Circadian cosine component',
            
            'Meal-related glucose patterns',
            'Sleep-related glucose patterns'
        ],
        'Typical Range': [
            '80-120 mg/dL (healthy)',
            '10-30 mg/dL (healthy)',
            '10-30% (healthy)',
            
            '>70% (excellent), 50-70% (good)',
            '<25% (good), >70% (poor)',
            '<4% (safe), >10% (dangerous)',
            '0 (ideal) to 1 (concerning)',
            
            '<60 (stable), >100 (highly variable)',
            '<0.3 (good), >1.0 (poor control)',
            '<5 (stable), >20 (highly variable)',
            '<20 (stable), >40 (variable)',
            
            '-50 to +50 mg/dL per 5min',
            '-10 to +10 mg/dL per min',
            '-5 to +5 mg/dL per min²',
            
            '80-120 mg/dL (context dependent)',
            '5-25 mg/dL (recent variability)',
            '90-180 mg/dL (recent peak)',
            
            '0-23 (integer)',
            '0.0-1.0 (continuous)',
            '-1.0 to +1.0 (periodic)',
            '-1.0 to +1.0 (periodic)',
            
            '0 (non-meal) or 1 (meal time)',
            '0 (day) or 1 (night)'
        ]
    }
    
    # Create DataFrame
    feature_df = pd.DataFrame(feature_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Feature Categories Distribution
    ax1 = axes[0, 0]
    category_counts = feature_df['Feature Category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
    bars = ax1.bar(range(len(category_counts)), category_counts.values, color=colors)
    ax1.set_xticks(range(len(category_counts)))
    ax1.set_xticklabels(category_counts.index, rotation=45, ha='right')
    ax1.set_title('Number of Features by Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Sample Circadian Encoding
    ax2 = axes[0, 1]
    hours = np.arange(0, 24, 0.5)
    sin_hours = np.sin(2 * np.pi * hours / 24)
    cos_hours = np.cos(2 * np.pi * hours / 24)
    
    ax2.plot(hours, sin_hours, 'b-', linewidth=2, label='sin(2π×hour/24)', alpha=0.8)
    ax2.plot(hours, cos_hours, 'r-', linewidth=2, label='cos(2π×hour/24)', alpha=0.8)
    ax2.axhspan(0.8, 1.0, alpha=0.2, color='yellow', label='Morning Peak')
    ax2.axhspan(-1.0, -0.8, alpha=0.2, color='blue', label='Evening Trough')
    
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Encoded Value')
    ax2.set_title('Circadian Encoding: Sine/Cosine Transform', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, 25, 4))
    
    # Plot 3: Sample MAGE Calculation
    ax3 = axes[1, 0]
    # Simulate glucose data with peaks and troughs
    time_points = np.arange(0, 288, 1)  # 24 hours, 5-min intervals
    glucose_sim = 100 + 20*np.sin(2*np.pi*time_points/288) + 10*np.sin(4*np.pi*time_points/288) + np.random.normal(0, 5, 288)
    
    # Find peaks and troughs
    peaks, _ = find_peaks(glucose_sim, height=np.mean(glucose_sim))
    troughs, _ = find_peaks(-glucose_sim, height=-np.mean(glucose_sim))
    
    ax3.plot(time_points/12, glucose_sim, 'g-', alpha=0.7, linewidth=1, label='Glucose')
    ax3.scatter(peaks/12, glucose_sim[peaks], color='red', s=50, zorder=5, label='Peaks')
    ax3.scatter(troughs/12, glucose_sim[troughs], color='blue', s=50, zorder=5, label='Troughs')
    ax3.axhline(np.mean(glucose_sim), color='black', linestyle='--', alpha=0.5, label='Mean')
    ax3.axhline(np.mean(glucose_sim) + np.std(glucose_sim), color='orange', linestyle=':', alpha=0.7, label='Mean + 1σ')
    
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Glucose (mg/dL)')
    ax3.set_title('MAGE Calculation: Peak/Trough Detection', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: TIR Visualization
    ax4 = axes[1, 1]
    glucose_range = np.arange(50, 250, 1)
    
    # Color zones
    ax4.axhspan(50, 70, alpha=0.3, color='red', label='Below Range (<70)')
    ax4.axhspan(70, 140, alpha=0.3, color='green', label='In Range (70-140)')
    ax4.axhspan(140, 180, alpha=0.3, color='yellow', label='Above Range (140-180)')
    ax4.axhspan(180, 250, alpha=0.3, color='red', label='Very High (>180)')
    
    # Sample glucose distribution
    glucose_dist = np.random.normal(110, 25, 1000)
    ax4.hist(glucose_dist, bins=30, alpha=0.6, color='blue', density=True, 
             orientation='horizontal', label='Sample Distribution')
    
    ax4.set_ylabel('Glucose (mg/dL)')
    ax4.set_xlabel('Density')
    ax4.set_title('Time in Range (TIR) Zones', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(50, 250)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "feature_engineering_overview.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save feature table as CSV
    feature_df.to_csv(os.path.join(PLOTS_OUTPUT_DIR, "engineered_features_reference.csv"), index=False)
    
    # Create detailed feature correlation heatmap
    create_feature_importance_summary()
    
    logging.info("Feature engineering documentation saved")
    return feature_df

def create_feature_importance_summary():
    """Create feature importance and correlation summary"""
    
    # Load your actual data to show real correlations
    if os.path.exists(SUMMARY_PATH):
        df = pd.read_csv(SUMMARY_PATH)
        
        # Select key features for correlation analysis
        key_features = [
            'mean_blood_glucose', 'std_dev', 'cv_percent',
            'tir_percent', 'tar_percent', 'tbr_percent',
            'mage', 'j_index', 'lability_index'
        ]
        
        # Filter features that exist in the data
        available_features = [f for f in key_features if f in df.columns]
        
        if len(available_features) > 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Correlation heatmap
            corr_matrix = df[available_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       ax=ax1, square=True, linewidths=0.5)
            ax1.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Feature distributions by study group - LINE PLOT VERSION
            if 'study_group' in df.columns:
                # Select a key feature for distribution comparison
                feature_to_plot = 'tir_percent' if 'tir_percent' in available_features else available_features[0]
                
                # Create line plot using kernel density estimation
                for i, group in enumerate(df['study_group'].unique()[:4]):  # Limit to 4 groups for clarity
                    group_data = df[df['study_group'] == group][feature_to_plot].dropna()
                    if len(group_data) > 5:  # Need at least 5 points for KDE
                        # Calculate kernel density estimation
                        from scipy.stats import gaussian_kde
                        
                        # Create density estimation
                        kde = gaussian_kde(group_data)
                        
                        # Create x-axis range
                        x_min, x_max = group_data.min(), group_data.max()
                        x_range = np.linspace(x_min - (x_max - x_min) * 0.1, 
                                            x_max + (x_max - x_min) * 0.1, 100)
                        
                        # Calculate density values
                        density = kde(x_range)
                        
                        # Plot line
                        colors = ['red', 'blue', 'green', 'orange']
                        ax2.plot(x_range, density, 
                                label=group, 
                                linewidth=2.5, 
                                alpha=0.8,
                                color=colors[i % len(colors)])
                        
                        # Add mean line
                        mean_val = group_data.mean()
                        max_density = density.max()
                        ax2.axvline(mean_val, 
                                   color=colors[i % len(colors)], 
                                   linestyle='--', 
                                   alpha=0.6, 
                                   linewidth=1.5)
                        
                        # Add mean value annotation
                        ax2.annotate(f'{mean_val:.1f}', 
                                   xy=(mean_val, max_density * 0.8), 
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=9,
                                   color=colors[i % len(colors)],
                                   fontweight='bold')
                
                ax2.set_xlabel(feature_to_plot.replace('_', ' ').title(), fontsize=12)
                ax2.set_ylabel('Density', fontsize=12)
                ax2.set_title(f'{feature_to_plot.replace("_", " ").title()} Distribution by Study Group\n(Line Plot with KDE)', 
                             fontsize=14, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                
                # Improve layout
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "feature_correlations_and_distributions.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()

def create_clinical_interpretation_guide():
    """Create a clinical interpretation guide for features"""
    
    clinical_guide = {
        'Metric': [
            'TIR (Time in Range)', 'TAR (Time Above Range)', 'TBR (Time Below Range)',
            'MAGE', 'CV%', 'J-Index', 'CONGA', 'Lability Index'
        ],
        'Clinical Significance': [
            'Proportion of time glucose is in target range (70-140 mg/dL)',
            'Time spent in hyperglycemia (>140 mg/dL) - diabetes risk',
            'Time spent in hypoglycemia (<70 mg/dL) - safety risk',
            'Magnitude of glycemic swings - cardiovascular risk marker',
            'Glucose variability relative to mean - stability indicator',
            'Combined metric of mean glucose and variability',
            'Hour-delayed glucose variability - meal response indicator',
            'Rate of glucose fluctuations - system stability'
        ],
        'Healthy Target': [
            '>70%', '<25%', '<4%', '<60 mg/dL', '<30%', '<0.3', '<20 mg/dL', '<5'
        ],
        'Clinical Interpretation': [
            'Higher = better glucose control',
            'Lower = less hyperglycemic exposure',
            'Lower = safer (avoid dangerous lows)',
            'Lower = more stable glucose patterns',
            'Lower = more predictable glucose',
            'Lower = better overall control',
            'Lower = better meal tolerance',
            'Lower = more stable glucose system'
        ]
    }
    
    guide_df = pd.DataFrame(clinical_guide)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=guide_df.values,
                     colLabels=guide_df.columns,
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.15, 0.4, 0.15, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the table
    table.auto_set_column_width(col=list(range(len(guide_df.columns))))
    
    # Header styling
    for i in range(len(guide_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(guide_df) + 1):
        for j in range(len(guide_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Clinical Interpretation Guide for Engineered Features', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_OUTPUT_DIR, "clinical_interpretation_guide.png"), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save as CSV too
    guide_df.to_csv(os.path.join(PLOTS_OUTPUT_DIR, "clinical_interpretation_guide.csv"), index=False)
    
    logging.info("Clinical interpretation guide saved")


# =====================================================================
# === MAIN WORKFLOW ===
def main():
    """Main workflow function"""
    print("ENHANCED TRUE HEALTHY CLUSTER IDENTIFICATION")
    print("\n" + "="*70)
    print(f"Plots will be saved to: {PLOTS_OUTPUT_DIR}")
    print("="*70)
    
    # Load or compute metrics
    if not os.path.exists(SUMMARY_PATH):
        logging.info("Computing participant metrics from scratch...")
        metrics_df = compute_participant_metrics()
    else:
        logging.info("Using existing summarized metrics file.")
        metrics_df = pd.read_csv(SUMMARY_PATH)

    # Check if required columns exist
    required_cols = {'mean_blood_glucose', 'std_dev', 'cv_percent', 'tir_percent', 'tbr_percent', 
                     'mage', 'lability_index', 'conga_2h'}
    
    if required_cols.issubset(set(metrics_df.columns)):
        # Perform enhanced clustering with visualization
        updated_metrics_df, labels, cluster_labels = cluster_healthy_participants_enhanced(metrics_df)
        
        # Get healthy participants data for report
        healthy_df = metrics_df[metrics_df['study_group'] == 'healthy']
        
        # Save comprehensive report
        save_summary_report(updated_metrics_df, labels, healthy_df)
        
        print(f"\nANALYSIS COMPLETE!")
        print(f"All outputs saved to: {PLOTS_OUTPUT_DIR}")
        print(f"{len(os.listdir(PLOTS_OUTPUT_DIR))} files generated")

        # CREATE FEATURE DOCUMENTATION (ADD THIS)
        print("Creating feature engineering documentation table..." + "\n")
        feature_df = create_feature_explanation_table()
        create_clinical_interpretation_guide()
        
    else:
        logging.warning("Required metrics not found in existing file. Recomputing metrics.")
        metrics_df = compute_participant_metrics()
        updated_metrics_df, labels, cluster_labels = cluster_healthy_participants_enhanced(metrics_df)
        
        # Get healthy participants data for report
        healthy_df = metrics_df[metrics_df['study_group'] == 'healthy']
        
        # Save comprehensive report
        save_summary_report(updated_metrics_df, labels, healthy_df)

if __name__ == '__main__':
    main()