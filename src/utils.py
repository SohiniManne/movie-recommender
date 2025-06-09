"""
Utility functions for movie recommender
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

def plot_recommendations_analysis(movies_df: pd.DataFrame):
    """Create visualizations for recommendation analysis"""
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Language distribution
    movies_df['language'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
    axes[0,0].set_title('Movies by Language')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Genre distribution
    movies_df['genre'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightgreen')
    axes[0,1].set_title('Movies by Genre')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Rating distribution
    movies_df['rating'].hist(bins=20, ax=axes[1,0], color='orange', alpha=0.7)
    axes[1,0].set_title('Rating Distribution')
    axes[1,0].set_xlabel('Rating')
    
    # Year vs Rating
    movies_df.plot.scatter(x='year', y='rating', ax=axes[1,1], color='red', alpha=0.6)
    axes[1,1].set_title('Year vs Rating')
    
    plt.tight_layout()
    plt.show()

def format_movie_info(movie_info: Dict) -> str:
    """Format movie information for display"""
    return f"""
    🎬 {movie_info['title']}
    📅 {movie_info['year']} | 🌍 {movie_info['language']}
    🎭 {movie_info['genre']} | ⭐ {movie_info['rating']}/10
    """

def export_recommendations(recommendations: List[Dict], filename: str):
    """Export recommendations to CSV"""
    df = pd.DataFrame(recommendations)
    df.to_csv(filename, index=False)
    print(f"Recommendations exported to {filename}")