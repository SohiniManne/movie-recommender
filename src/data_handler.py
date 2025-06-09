"""
Data handling utilities for movie recommender
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class MovieDataHandler:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
    
    def load_from_csv(self, movies_path: str, ratings_path: str):
        """Load data from CSV files"""
        try:
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)
            print(f"Data loaded successfully!")
            print(f"Movies: {len(self.movies_df)}, Ratings: {len(self.ratings_df)}")
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            return False
        return True
    
    def save_to_csv(self, movies_path: str, ratings_path: str):
        """Save data to CSV files"""
        if self.movies_df is not None:
            self.movies_df.to_csv(movies_path, index=False)
        if self.ratings_df is not None:
            self.ratings_df.to_csv(ratings_path, index=False)
        print("Data saved successfully!")
    
    def get_movie_stats(self) -> Dict:
        """Get basic statistics about the dataset"""
        if self.movies_df is None:
            return {}
        
        return {
            'total_movies': len(self.movies_df),
            'languages': self.movies_df['language'].value_counts().to_dict(),
            'genres': self.movies_df['genre'].value_counts().to_dict(),
            'avg_rating': self.movies_df['rating'].mean(),
            'year_range': f"{self.movies_df['year'].min()}-{self.movies_df['year'].max()}"
        }