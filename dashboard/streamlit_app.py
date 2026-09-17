"""
Streamlit Dashboard for CipherBench (MySQL-backed variant).

This dashboard requires a configured, reachable MySQL database (see
config.yaml / CIPHERBENCH_DB_PASSWORD). For a dependency-free demo that
reads directly from the CSV/JSON files in experiments/results/, use
`app/streamlit_app.py` instead - that is the recommended entry point and
does not require a database.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_operations import ExperimentDB


# Page configuration
st.set_page_config(
    page_title="CipherBench Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database (fails gracefully - no database is required to explore
# the project; see app/streamlit_app.py for the file-based alternative).
@st.cache_resource
def get_db():
    return ExperimentDB()

st.title("🔐 CipherBench: Cryptographic Algorithm Identification")
st.markdown("""
This dashboard visualizes the performance of machine learning models for identifying
block cipher algorithms from ciphertext using NIST-derived statistical features.
""")

try:
    db = get_db()
    all_experiments = db.get_experiments()
except Exception as e:
    st.error(
        f"⚠️ Could not connect to the MySQL database ({e}).\n\n"
        "This dashboard variant requires a configured database. For a dependency-free "
        "demo that reads directly from `experiments/results/`, run:\n\n"
        "`streamlit run app/streamlit_app.py`"
    )
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

if not all_experiments:
    st.error("⚠️ No experiments found in database. Please run training first: `python src/train.py`")
    st.stop()

# Extract unique values
all_models = sorted(list(set([exp['model_name'] for exp in all_experiments])))
all_tasks = sorted(list(set([exp['task_type'] for exp in all_experiments])))
all_sizes = sorted(list(set([exp['ciphertext_size'] for exp in all_experiments])))

# Sidebar selections
task_type = st.sidebar.selectbox("Task Type", all_tasks)
ciphertext_size = st.sidebar.selectbox("Ciphertext Size", all_sizes)
selected_models = st.sidebar.multiselect("Models", all_models, default=all_models)

# Fetch filtered data
filtered_experiments = [
    exp for exp in all_experiments
    if exp['task_type'] == task_type
    and exp['ciphertext_size'] == ciphertext_size
    and exp['model_name'] in selected_models
]

if not filtered_experiments:
    st.warning("No experiments match the selected filters.")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(filtered_experiments)

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "🎯 Best Models", "📈 Confusion Matrices", "📋 Detailed Results"])

# Tab 1: Model Comparison
with tab1:
    st.header("Model Performance Comparison")

    # Aggregate by model (average if multiple runs)
    model_stats = df.groupby('model_name').agg({
        'accuracy': 'mean',
        'precision_score': 'mean',
        'recall_score': 'mean',
        'f1_score': 'mean'
    }).reset_index()

    # Bar chart comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Comparison")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(model_stats['model_name'], model_stats['accuracy'], color='steelblue')
        ax1.set_xlabel('Accuracy')
        ax1.set_title(f'Model Accuracy - {task_type} ({ciphertext_size})')
        ax1.set_xlim(0, 1)
        for i, v in enumerate(model_stats['accuracy']):
            ax1.text(v + 0.01, i, f'{v:.3f}', va='center')
        st.pyplot(fig1)

    with col2:
        st.subheader("F1-Score Comparison")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.barh(model_stats['model_name'], model_stats['f1_score'], color='coral')
        ax2.set_xlabel('F1-Score')
        ax2.set_title(f'Model F1-Score - {task_type} ({ciphertext_size})')
        ax2.set_xlim(0, 1)
        for i, v in enumerate(model_stats['f1_score']):
            ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        st.pyplot(fig2)

    # Metrics table
    st.subheader("Metrics Summary")
    st.dataframe(
        model_stats.style.format({
            'accuracy': '{:.4f}',
            'precision_score': '{:.4f}',
            'recall_score': '{:.4f}',
            'f1_score': '{:.4f}'
        }).background_gradient(cmap='YlGn', subset=['accuracy', 'f1_score']),
        use_container_width=True
    )

# Tab 2: Best Models
with tab2:
    st.header("Top Performing Models")

    best_models = db.get_best_models(task_type, ciphertext_size, top_n=10)

    if best_models:
        best_df = pd.DataFrame(best_models)

        # Display ranking
        st.subheader(f"Top {len(best_models)} Models")

        # Add rank column
        best_df.insert(0, 'Rank', range(1, len(best_df) + 1))

        st.dataframe(
            best_df[['Rank', 'model_name', 'accuracy', 'precision_score', 'recall_score', 'f1_score']].style.format({
                'accuracy': '{:.4f}',
                'precision_score': '{:.4f}',
                'recall_score': '{:.4f}',
                'f1_score': '{:.4f}'
            }).background_gradient(cmap='RdYlGn', subset=['accuracy']),
            use_container_width=True
        )

        # Winner highlight
        winner = best_models[0]
        st.success(f"🏆 **Best Model:** {winner['model_name']} with {winner['accuracy']:.4f} accuracy")

    else:
        st.info("No models found for this configuration.")

# Tab 3: Confusion Matrices
with tab3:
    st.header("Confusion Matrices")

    # Select model to display
    selected_model = st.selectbox("Select Model", selected_models)

    # Get experiment for selected model
    model_exp = [exp for exp in filtered_experiments if exp['model_name'] == selected_model]

    if model_exp:
        exp = model_exp[0]  # Take first if multiple
        cm = np.array(exp['confusion_matrix'])

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)

        # Labels
        if task_type == 'binary':
            classes = ['Class 0', 'Class 1']
        else:
            classes = ['AES', '3DES', 'Blowfish', 'CAST', 'RC2']

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes[:cm.shape[1]],
               yticklabels=classes[:cm.shape[0]],
               title=f'Confusion Matrix - {selected_model}',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        st.pyplot(fig)

        # Display metrics for this model
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{exp['accuracy']:.4f}")
        col2.metric("Precision", f"{exp['precision_score']:.4f}")
        col3.metric("Recall", f"{exp['recall_score']:.4f}")
        col4.metric("F1-Score", f"{exp['f1_score']:.4f}")

    else:
        st.warning(f"No confusion matrix found for {selected_model}")

# Tab 4: Detailed Results
with tab4:
    st.header("Detailed Experiment Results")

    # Display full dataframe
    display_df = df[[
        'model_name', 'algorithm_pair', 'accuracy', 'precision_score',
        'recall_score', 'f1_score', 'training_time', 'timestamp'
    ]].copy()

    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(
        display_df.style.format({
            'accuracy': '{:.4f}',
            'precision_score': '{:.4f}',
            'recall_score': '{:.4f}',
            'f1_score': '{:.4f}',
            'training_time': '{:.2f}s'
        }),
        use_container_width=True,
        height=400
    )

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"cipherbench_results_{task_type}_{ciphertext_size}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**CipherBench** - Machine Learning Framework for Block Cipher Algorithm Identification
Built with Streamlit, Flask, MySQL, and TensorFlow
""")
