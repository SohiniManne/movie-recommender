"""
Main entry point for Indian Cinema Movie Recommender
"""
from src.movie_recommender import IndianCinemaRecommender

def main():
    print("🎬 Starting Indian Cinema Movie Recommender...")
    
    # Initialize recommender
    recommender = IndianCinemaRecommender()
    
    # Create sample data
    recommender.create_sample_data()
    
    # Build models
    recommender.build_content_based_recommender()
    recommender.build_collaborative_filtering()
    
    # Demo recommendations
    print("\n=== DEMO RECOMMENDATIONS ===")
    
    # Content-based
    content_recs = recommender.get_content_recommendations(
        'Baahubali: The Beginning', 
        language_filter=['Telugu', 'Tamil'],
        n_recommendations=5
    )
    recommender.print_recommendations(content_recs, "Similar Movies")
    
    # Hybrid recommendations
    hybrid_recs = recommender.get_hybrid_recommendations(
        user_id=1,
        movie_title='RRR',
        language_filter=['Telugu', 'Hindi'],
        n_recommendations=5
    )
    recommender.print_recommendations(hybrid_recs, "Hybrid Recommendations")

if __name__ == "__main__":
    main()