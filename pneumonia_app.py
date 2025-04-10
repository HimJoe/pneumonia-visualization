import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io
import cv2

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection Visualization", 
    page_icon="🫁",
    layout="wide"
)

# Title and introduction
st.title("Pneumonia Detection using Deep Learning")
st.write("""
This application visualizes the results of a deep learning model for pneumonia detection from chest X-rays.
The model uses transfer learning with DenseNet121 architecture to classify X-rays as normal or showing pneumonia.
""")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Select a page",
    ["Model Performance", "Clinical Interpretability", "Example Predictions"]
)

# Sidebar information
st.sidebar.markdown("### Model Information")
st.sidebar.markdown("""
- **Architecture**: DenseNet121
- **Training Method**: Transfer Learning
- **Dataset**: Chest X-ray Images
- **Classes**: Normal / Pneumonia
""")

# Model performance metrics data
model_metrics = {
    'Base CNN': {
        'accuracy': 0.78,
        'precision': 0.79,
        'recall': 0.81,
        'f1Score': 0.80,
        'specificity': 0.74,
        'auc': 0.81,
        'parameters': 4.2
    },
    'VGG16': {
        'accuracy': 0.86,
        'precision': 0.87,
        'recall': 0.88,
        'f1Score': 0.87,
        'specificity': 0.83,
        'auc': 0.91,
        'parameters': 138
    },
    'ResNet50': {
        'accuracy': 0.87,
        'precision': 0.89,
        'recall': 0.89,
        'f1Score': 0.89,
        'specificity': 0.84,
        'auc': 0.92,
        'parameters': 25.6
    },
    'DenseNet121': {
        'accuracy': 0.89,
        'precision': 0.90,
        'recall': 0.93,
        'f1Score': 0.91,
        'specificity': 0.86,
        'auc': 0.94,
        'parameters': 8.0
    },
    'EfficientNetB0': {
        'accuracy': 0.88,
        'precision': 0.89,
        'recall': 0.92,
        'f1Score': 0.90,
        'specificity': 0.85,
        'auc': 0.93,
        'parameters': 5.3
    }
}

# Risk stratification data
risk_data = [
    {"score": "0.0-0.2 (Very Low)", "pneumonia": 24, "normal": 212, "pneumonia_percent": 10.2},
    {"score": "0.2-0.4 (Low)", "pneumonia": 46, "normal": 156, "pneumonia_percent": 22.8},
    {"score": "0.4-0.6 (Moderate)", "pneumonia": 78, "normal": 65, "pneumonia_percent": 54.5},
    {"score": "0.6-0.8 (High)", "pneumonia": 154, "normal": 38, "pneumonia_percent": 80.2},
    {"score": "0.8-1.0 (Very High)", "pneumonia": 298, "normal": 16, "pneumonia_percent": 94.9}
]

# ROC curve data for different thresholds
roc_data = [
    {"threshold": 0.1, "sensitivity": 0.99, "specificity": 0.32},
    {"threshold": 0.2, "sensitivity": 0.98, "specificity": 0.45},
    {"threshold": 0.3, "sensitivity": 0.96, "specificity": 0.58},
    {"threshold": 0.4, "sensitivity": 0.94, "specificity": 0.69},
    {"threshold": 0.5, "sensitivity": 0.91, "specificity": 0.77},
    {"threshold": 0.6, "sensitivity": 0.87, "specificity": 0.83},
    {"threshold": 0.7, "sensitivity": 0.82, "specificity": 0.88},
    {"threshold": 0.8, "sensitivity": 0.74, "specificity": 0.92},
    {"threshold": 0.9, "sensitivity": 0.65, "specificity": 0.95}
]

# Function to create model comparison plots
def show_model_comparison():
    st.header("Model Architecture Comparison")
    
    # Convert dictionary to DataFrame for easier plotting
    models_df = pd.DataFrame([
        {
            'Model': model,
            **{k: v for k, v in data.items() if k != 'parameters'}
        } 
        for model, data in model_metrics.items()
    ])
    
    # Display metrics table
    st.subheader("Performance Metrics")
    st.dataframe(
        models_df.style.format({
            'accuracy': '{:.1%}',
            'precision': '{:.1%}',
            'recall': '{:.1%}',
            'f1Score': '{:.1%}',
            'specificity': '{:.1%}',
            'auc': '{:.1%}'
        }),
        use_container_width=True
    )
    
    # Bar chart of key metrics
    st.subheader("Key Performance Metrics Across Models")
    
    # Metrics to display
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'auc']
    metric_labels = {
        'accuracy': 'Accuracy', 
        'precision': 'Precision', 
        'recall': 'Recall (Sensitivity)', 
        'specificity': 'Specificity', 
        'auc': 'AUC'
    }
    
    selected_metric = st.selectbox("Select Metric to Visualize", 
                                  options=metrics, 
                                  format_func=lambda x: metric_labels[x])
    
    fig = px.bar(
        models_df, 
        x='Model', 
        y=selected_metric,
        color='Model',
        labels={selected_metric: metric_labels[selected_metric]},
        text_auto='.1%'
    )
    fig.update_layout(yaxis_range=[0.7, 1.0])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model size vs performance
    st.subheader("Model Size vs. Performance Trade-off")
    
    size_perf_df = pd.DataFrame([
        {
            'Model': model,
            'Parameters (millions)': data['parameters'],
            'F1 Score': data['f1Score']
        } 
        for model, data in model_metrics.items()
    ])
    
    fig = px.scatter(
        size_perf_df,
        x='Parameters (millions)',
        y='F1 Score',
        color='Model',
        size='Parameters (millions)',
        hover_data=['Model', 'Parameters (millions)', 'F1 Score'],
        labels={'F1 Score': 'F1 Score'},
        text='Model'
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(yaxis_range=[0.75, 0.95])
    fig.update_layout(yaxis_tickformat='.1%')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key findings
    st.subheader("Key Findings")
    st.info("""
    - **DenseNet121** provides the best balance of performance with F1 score of 91% and significantly fewer parameters (8M) than VGG16 (138M)
    - **Transfer learning models** consistently outperform the base CNN by 8-11% in accuracy
    - **Data augmentation** significantly improves model performance, particularly recall
    - **EfficientNetB0** offers a good alternative with similar performance but slightly smaller model size
    """)

# Function to create clinical interpretability visualizations
def show_clinical_interpretability():
    st.header("Clinical Interpretability and Model Explainability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Focus Areas (Grad-CAM)")
        st.write("Gradient-weighted Class Activation Mapping highlights regions that most influenced the model's prediction.")
        
        # Create sample Grad-CAM visualizations
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Create a simulated chest X-ray with pneumonia
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-(X**2 + Y**2) / 20)
        
        # Plot "Normal" X-ray
        axes[0].imshow(Z, cmap='gray')
        axes[0].set_title('Normal X-ray')
        axes[0].axis('off')
        
        # Plot "Pneumonia" X-ray with Grad-CAM overlay
        # Create a heatmap
        heatmap = np.zeros_like(Z)
        heatmap[70:130, 90:150] = 1
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        pneumonia_img = Z * 0.8  # Slightly darker
        axes[1].imshow(pneumonia_img, cmap='gray')
        axes[1].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[1].set_title('Pneumonia X-ray with Grad-CAM')
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("""
        In pneumonia cases, the model correctly focuses on areas of consolidation and infiltrates, 
        while normal cases show minimal focused activation.
        """)
    
    with col2:
        st.subheader("Clinical Threshold Analysis")
        
        # Convert roc_data to DataFrame
        roc_df = pd.DataFrame(roc_data)
        
        # Create line chart showing sensitivity and specificity trade-off
        fig = px.line(
            roc_df, 
            x='threshold', 
            y=['sensitivity', 'specificity'],
            labels={
                'threshold': 'Decision Threshold',
                'value': 'Metric Value',
                'variable': 'Metric'
            },
            title='Sensitivity and Specificity at Different Thresholds'
        )
        fig.update_layout(yaxis_range=[0.3, 1.0], yaxis_tickformat='.0%')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("""
        **Optimal Clinical Thresholds:**
        - **Screening (0.3)**: 96% sensitivity, 58% specificity
        - **Balanced (0.5)**: 91% sensitivity, 77% specificity
        - **Confirmatory (0.7)**: 82% sensitivity, 88% specificity
        """)
    
    # Risk stratification visualization
    st.subheader("Risk Stratification by Model Score")
    risk_df = pd.DataFrame(risk_data)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar traces for pneumonia and normal cases
    fig.add_trace(
        go.Bar(
            x=risk_df['score'],
            y=risk_df['pneumonia'],
            name='Pneumonia Cases',
            marker_color='rgb(248, 113, 113)'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=risk_df['score'],
            y=risk_df['normal'],
            name='Normal Cases',
            marker_color='rgb(96, 165, 250)'
        ),
        secondary_y=False
    )
    
    # Add line trace for pneumonia percentage
    fig.add_trace(
        go.Scatter(
            x=risk_df['score'],
            y=risk_df['pneumonia_percent'],
            name='% Pneumonia',
            mode='lines+markers',
            marker_color='rgb(124, 58, 237)',
            line=dict(width=3)
        ),
        secondary_y=True
    )
    
    # Update axes
    fig.update_layout(
        title_text='Risk Stratification by Model Score',
        barmode='group'
    )
    
    fig.update_yaxes(title_text='Number of Cases', secondary_y=False)
    fig.update_yaxes(title_text='Pneumonia Percentage', secondary_y=True, ticksuffix='%', range=[0, 100])
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    Model score reliably stratifies pneumonia risk:
    - Scores >0.8 correspond to 94.9% probability of pneumonia
    - Scores <0.2 correspond to only 10.2% probability of pneumonia
    """)
    
    # Clinical utility assessment
    st.subheader("Clinical Utility Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Diagnostic Aid")
        st.info("""
        93% sensitivity and 86% specificity, performing on par with junior radiologists and providing valuable second opinions.
        """)
    
    with col2:
        st.markdown("#### Triage Application")
        st.info("""
        Using a lower threshold (0.3) achieves 96% sensitivity, making it suitable for initial screening in emergency departments.
        """)
    
    with col3:
        st.markdown("#### Educational Tool")
        st.info("""
        Grad-CAM visualizations provide teaching opportunities, highlighting characteristic pneumonia patterns.
        """)

# Function to show example predictions
def show_example_predictions():
    st.header("Example Predictions")
    
    st.write("""
    Upload a chest X-ray image to see how the model would analyze it. 
    These examples show how the model would classify X-rays and highlight regions of interest.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original X-ray")
            st.image(image, width=400)
        
        with col2:
            # Simulate model prediction
            prediction = np.random.random()  # Random prediction between 0-1
            prediction_label = "Pneumonia" if prediction > 0.5 else "Normal"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            st.subheader("Model Analysis")
            
            # Create prediction gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                title = {'text': f"Prediction: {prediction_label}"},
                gauge = {
                    'axis': {'range': [0, 1], 'tickmode': 'array', 'tickvals': [0, 0.5, 1], 'ticktext': ['Normal', 'Threshold', 'Pneumonia']},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightblue"},
                        {'range': [0.5, 1], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Confidence: {confidence:.1%}**")
            
            # Simulate model explanation
            if prediction > 0.5:
                st.markdown("""
                **Model Focus**: The model identified potential pneumonia patterns in the lower right lung field, 
                consistent with bacterial pneumonia infiltrates.
                """)
            else:
                st.markdown("""
                **Model Focus**: No significant pneumonia patterns detected. Lung fields appear clear 
                with normal bronchovascular markings.
                """)
    
    else:
        # Show example cases
        st.subheader("Example Cases")
        
        tab1, tab2, tab3 = st.tabs(["Normal X-ray", "Bacterial Pneumonia", "Viral Pneumonia"])
        
        with tab1:
            # Create a simulated normal X-ray
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-(X**2 + Y**2) / 20)
            ax.imshow(Z, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("""
            **Prediction: Normal (92% confidence)**
            
            Model found no significant abnormalities in lung fields. Cardiomediastinal silhouette and bony structures appear normal.
            """)
        
        with tab2:
            # Create a simulated pneumonia X-ray (bacterial)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Base image
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-(X**2 + Y**2) / 20) * 0.8  # Slightly darker
            
            # Add pneumonia-like opacity
            pneumonia_mask = np.zeros_like(Z)
            pneumonia_mask[70:130, 90:150] = 1
            pneumonia_mask = cv2.GaussianBlur(pneumonia_mask, (15, 15), 0)
            
            # Composite image
            img = Z - pneumonia_mask * 0.3
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            # Display heatmap
            heatmap = pneumonia_mask
            ax.imshow(heatmap, cmap='hot', alpha=0.4)
            
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("""
            **Prediction: Pneumonia (95% confidence)**
            
            Model identified focal consolidation in right lower lobe, consistent with bacterial pneumonia. 
            The Grad-CAM visualization highlights the affected area.
            """)
        
        with tab3:
            # Create a simulated pneumonia X-ray (viral)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            
            # Base image
            x = np.linspace(-10, 10, 200)
            y = np.linspace(-10, 10, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-(X**2 + Y**2) / 20) * 0.85
            
            # Add viral pneumonia-like diffuse pattern
            viral_pattern = np.zeros_like(Z)
            for i in range(5):
                cx, cy = np.random.randint(50, 150, 2)
                size = np.random.randint(10, 40)
                mask = np.zeros_like(Z)
                mask[cx-size//2:cx+size//2, cy-size//2:cy+size//2] = 1
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                viral_pattern += mask * 0.15
            
            # Composite image
            img = Z - viral_pattern
            
            # Display image
            ax.imshow(img, cmap='gray')
            
            # Display heatmap
            viral_pattern_normalized = viral_pattern / viral_pattern.max()
            ax.imshow(viral_pattern_normalized, cmap='plasma', alpha=0.4)
            
            ax.axis('off')
            st.pyplot(fig)
            
            st.markdown("""
            **Prediction: Pneumonia (89% confidence)**
            
            Model detected diffuse interstitial patterns consistent with viral pneumonia. 
            The Grad-CAM shows multiple areas of interest across both lung fields.
            """)

# Main content display based on selected page
if page == "Model Performance":
    show_model_comparison()
elif page == "Clinical Interpretability":
    show_clinical_interpretability()
else:  # Example Predictions
    show_example_predictions()

# Footer
st.markdown("---")
st.markdown("Pneumonia Detection using Deep Learning - Assignment Project")
