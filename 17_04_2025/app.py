from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import io
import base64
import os
from matplotlib.figure import Figure
from scipy.sparse import hstack, issparse
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__, static_url_path='/static')

# Define the HybridRecommender class before loading the model
class HybridRecommender:
    def __init__(self, kmeans_model, preprocessor, product_encoder, cluster_recs):
        self.kmeans = kmeans_model
        self.preprocessor = preprocessor
        self.product_encoder = product_encoder
        self.cluster_recommendations = cluster_recs
        self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')

        # Prepare data for collaborative filtering
        self.df = None  # Will be populated when loading
        self.user_features = None  # Will be populated when loading
        self.product_features = None  # Will be populated when loading

    def get_cluster_recommendations(self, user_data):
        """Get recommendations based on the user's cluster"""
        # Transform user data
        user_features = self.preprocessor.transform(user_data)

        # Predict cluster
        cluster = self.kmeans.predict(user_features)[0]

        # Return top products from that cluster
        return self.cluster_recommendations.get(cluster, [])

    def get_collaborative_recommendations(self, user_data, current_product=None):
        """Get collaborative filtering recommendations"""
        # Transform user data
        user_features = self.preprocessor.transform(user_data)

        # If a current product is provided, find similar user-product pairs
        if current_product is not None and hasattr(self, 'nn_model') and self.nn_model is not None:
            # Transform product data
            product_array = np.array([[current_product]])
            product_features = self.product_encoder.transform(product_array)

            # Combine user and product features
            hybrid_query = hstack([user_features, product_features])

            # Find similar user-product pairs
            distances, indices = self.nn_model.kneighbors(hybrid_query)

            # Extract products from the similar user-product pairs
            similar_products = []
            for idx in indices[0]:
                similar_products.append(self.df.iloc[idx]['Product'])

            # Remove the current product and return unique products
            if current_product in similar_products:
                similar_products.remove(current_product)

            return list(dict.fromkeys(similar_products))[:5]  # Return top 5 unique products

        return []

    def recommend(self, user_data, current_product=None, weight_cluster=0.6):
        """Generate hybrid recommendations"""
        # Get recommendations from both methods
        cluster_recs = self.get_cluster_recommendations(user_data)
        collab_recs = self.get_collaborative_recommendations(user_data, current_product)

        # Combine recommendations with weighting
        # Start with cluster recommendations
        hybrid_recs = cluster_recs.copy()

        # Add collaborative recommendations with custom weighting
        for product in collab_recs:
            if product not in hybrid_recs:
                hybrid_recs.append(product)

        # Reorder based on weights if we have both types of recommendations
        if current_product is not None and cluster_recs and collab_recs:
            # Create a scoring system (simple implementation)
            scores = {}

            # Score cluster recommendations
            for i, prod in enumerate(cluster_recs):
                scores[prod] = weight_cluster * (len(cluster_recs) - i) / len(cluster_recs)

            # Score collaborative recommendations
            for i, prod in enumerate(collab_recs):
                collab_score = (1 - weight_cluster) * (len(collab_recs) - i) / len(collab_recs)
                scores[prod] = scores.get(prod, 0) + collab_score

            # Sort by score
            hybrid_recs = sorted(hybrid_recs, key=lambda x: scores.get(x, 0), reverse=True)

        return hybrid_recs[:5]  # Return top 5 recommendations

    def visualize_recommendation_paths(self, user_data, current_product=None):
        """Visualize how recommendations are made"""
        cluster_recs = self.get_cluster_recommendations(user_data)
        collab_recs = self.get_collaborative_recommendations(user_data, current_product)
        hybrid_recs = self.recommend(user_data, current_product)

        # Return the paths for reference
        return {
            'cluster_recommendations': cluster_recs,
            'collaborative_recommendations': collab_recs,
            'hybrid_recommendations': hybrid_recs
        }

# Load the trained models
try:
    print("Loading trained models...")
    preprocessor = joblib.load('preprocessor.pkl')
    kmeans_model = joblib.load('kmeans_model.pkl')
    hybrid_recommender = joblib.load('hybrid_recommender.pkl')
    print("✅ Models loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Please make sure you've trained the models first by running the hybrid_recommender.py script.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Trying to create a new recommender instance...")
    try:
        # If loading fails, try to create a basic recommender with just the preprocessor and kmeans
        # This is a fallback and won't have all functionality
        cluster_recommendations = {}  # Empty dict as fallback
        hybrid_recommender = HybridRecommender(
            kmeans_model=kmeans_model,
            preprocessor=preprocessor,
            product_encoder=None,
            cluster_recs=cluster_recommendations
        )
        print("✅ Created basic recommender instance.")
    except Exception as e:
        print(f"❌ Error creating basic recommender: {e}")

# Define cluster characteristics (customize based on your actual clusters)
cluster_profiles = {
    0: "Young Mobile Users - Tech Enthusiasts",
    1: "Middle-aged Desktop Users - Home & Office Products",
    2: "Senior Tablet Users - Health & Wellness Products",
    3: "Young Adult Female Mobile Users - Fashion & Beauty",
    4: "Middle-aged Male Desktop Users - Electronics & Gadgets",
    # Add more cluster profiles based on your analysis
}

# Define training visualization images
training_visualizations = [
    {
        "id": "data_exploration",
        "title": "Data Exploration",
        "description": "Overview of customer demographics and product preferences",
        "filename": "data_exploration.png"
    },
    {
        "id": "cluster_evaluation",
        "title": "Cluster Evaluation",
        "description": "Metrics used to determine the optimal number of clusters",
        "filename": "cluster_evaluation.png"
    },
    {
        "id": "cluster_analysis",
        "title": "Cluster Analysis",
        "description": "Detailed analysis of customer segments",
        "filename": "cluster_analysis.png"
    },
    {
        "id": "cluster_products",
        "title": "Cluster Products",
        "description": "Top products for each customer segment",
        "filename": "cluster_products.png"
    }
]

def predict_cluster(user_data):
    """Predict which cluster the user belongs to"""
    # Transform user data
    user_features = preprocessor.transform(user_data)
    # Predict cluster
    cluster = kmeans_model.predict(user_features)[0]
    return cluster

def get_recommendations(user_data, current_product=None):
    """Get product recommendations for the user"""
    return hybrid_recommender.recommend(user_data, current_product)

def create_cluster_visualization(user_data, cluster):
    """Create visualization of user's position in cluster space"""
    # Transform user data
    user_features = preprocessor.transform(user_data)

    # Get all data points for visualization
    X_transformed = hybrid_recommender.user_features if hasattr(hybrid_recommender, 'user_features') else None
    
    if X_transformed is None:
        # Create a simple visualization if we don't have the full data
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()
        ax.text(0.5, 0.5, f"You are in Cluster {cluster}", 
                horizontalalignment='center', verticalalignment='center', 
                fontsize=20, transform=ax.transAxes)
        ax.set_title("Your Customer Segment", fontsize=14)
        ax.axis('off')
        fig.tight_layout()
    else:
        cluster_labels = kmeans_model.labels_

        # Convert to array if sparse
        X_array = X_transformed.toarray() if issparse(X_transformed) else X_transformed

        # Use PCA to reduce dimensionality
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(X_array)

        # Add user point to the reduced features
        user_reduced = pca.transform(user_features.toarray() if issparse(user_features) else user_features)

        # Create figure
        fig = Figure(figsize=(10, 8))
        ax = fig.subplots()

        # Plot all data points
        for i in range(len(np.unique(cluster_labels))):
            ax.scatter(
                reduced_features[cluster_labels == i, 0],
                reduced_features[cluster_labels == i, 1],
                s=50, alpha=0.6, label=f"Cluster {i}"
            )

        # Plot user point
        ax.scatter(
            user_reduced[0, 0],
            user_reduced[0, 1],
            s=200, color='red', marker='*', edgecolor='black', linewidth=2,
            label='You'
        )

        ax.set_title("Your Position in Customer Segments", fontsize=14)
        ax.set_xlabel("Principal Component 1", fontsize=12)
        ax.set_ylabel("Principal Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

    # Convert plot to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode PNG image to base64 string
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

def create_recommendations_visualization(recommendations, user_cluster):
    """Create visualization of recommended products"""
    if not recommendations:
        return None

    # Create figure
    fig = Figure(figsize=(12, 6))
    ax = fig.subplots()

    # Get colors
    colors = sns.color_palette("viridis", 10)

    # Create a bar chart of recommendations
    y_pos = np.arange(len(recommendations))
    ax.barh(y_pos, [1] * len(recommendations), color=colors[user_cluster % len(colors)])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(recommendations)
    ax.set_title(f"Top Recommended Products for You (Cluster: {user_cluster})", fontsize=14)
    ax.set_xlabel("Recommendation Strength", fontsize=12)
    ax.set_ylabel("Product", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    fig.tight_layout()

    # Convert plot to PNG image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode PNG image to base64 string
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/png;base64,{data}"

@app.route('/')
def index():
    return render_template('index.html', visualizations=training_visualizations)

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from form
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    device = request.form.get('device')
    
    # Create user data DataFrame
    user_data = pd.DataFrame({
        'Aging': [age],
        'Gender': [gender],
        'Device_Type': [device]
    })
    
    # Predict cluster
    user_cluster = predict_cluster(user_data)
    
    # Get recommendations
    recommendations = get_recommendations(user_data)
    
    # Create visualizations
    cluster_viz = create_cluster_visualization(user_data, user_cluster)
    recommendations_viz = create_recommendations_visualization(recommendations, user_cluster)
    
    # Get cluster profile
    cluster_profile = cluster_profiles.get(user_cluster, f"Cluster {user_cluster}")
    
    # Return results
    return jsonify({
        'profile': {
            'age': age,
            'gender': gender,
            'device': device
        },
        'cluster': {
            'id': int(user_cluster),
            'profile': cluster_profile
        },
        'recommendations': recommendations,
        'visualizations': {
            'cluster': cluster_viz,
            'recommendations': recommendations_viz
        }
    })

@app.route('/visualizations')
def get_visualizations():
    return jsonify(training_visualizations)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
