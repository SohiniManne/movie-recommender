# Indian Cinema Movie Recommendation System
# Supports Telugu, Tamil, Hindi, Malayalam movies
# Multiple recommendation algorithms: Content-based, Collaborative Filtering, Hybrid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

class IndianCinemaRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.content_similarity = None
        self.tfidf_matrix = None
        self.user_movie_matrix = None
        self.svd_model = None
        self.language_weights = {
            'Telugu': 1.0,
            'Tamil': 1.0, 
            'Hindi': 1.0,
            'Malayalam': 1.0
        }
        
    def create_sample_data(self):
        """Create sample movie dataset for Indian cinema"""
        
        # Create exactly 100 movie titles (25 per language)
        telugu_movies = [
            'Baahubali: The Beginning', 'Baahubali 2: The Conclusion', 'RRR', 'Pushpa: The Rise',
            'Arjun Reddy', 'Jersey', 'Fidaa', 'Rangasthalam', 'Ala Vaikunthapurramuloo', 'Sarileru Neekevvaru',
            'Eega', 'Magadheera', 'Chatrapathi', 'Simhadri', 'Vikramarkudu', 'Temper', 'Race Gurram', 
            'Attarintiki Daredi', 'S/O Satyamurthy', 'Srimanthudu', 'Janatha Garage', 'Khaidi No. 150', 
            'Mersal', 'Sarkar', 'Bharat Ane Nenu'
        ]
        
        tamil_movies = [
            'KGF Chapter 1', 'KGF Chapter 2', 'Master', 'Vikram', 'Beast', 'Valimai', 'Thunivu',
            'Enthiran', '2.0', 'Kabali', 'Kaala', 'Petta', 'Darbar', 'Annaatthe', 'Maanaadu',
            'Karnan', 'Asuran', 'Super Deluxe', 'Vada Chennai', 'Vikram Vedha', 'Thani Oruvan',
            'Kaththi', 'Thuppakki', 'Ghilli', 'Pokkiri'
        ]
        
        hindi_movies = [
            'Dangal', 'Baahubali 2 (Hindi)', 'RRR (Hindi)', 'KGF Chapter 2 (Hindi)', 'Pathaan', 'Jawan',
            'Tiger Zinda Hai', 'War', 'Uri: The Surgical Strike', 'Article 15', 'Andhadhun', 'Badhaai Ho',
            'Stree', 'Tumhari Sulu', 'Hindi Medium', 'Toilet: Ek Prem Katha', 'Shubh Mangal Saavdhan',
            'Bareilly Ki Barfi', 'Judwaa 2', 'Golmaal Again', 'Fukrey Returns', 'Jolly LLB 2',
            'Raees', 'Kaabil', 'Tubelight'
        ]
        
        malayalam_movies = [
            'Drishyam', 'Premam', 'Bangalore Days', 'Charlie', 'Ennu Ninte Moideen', 'Action Hero Biju',
            'Maheshinte Prathikaaram', 'Thondimuthalum Driksakshiyum', 'Angamaly Diaries', 'Take Off',
            'The Great Indian Kitchen', 'Kumbakonam Gopals', 'Minnal Murali', 'Malik', 'Kurup', 'Lucifer',
            'Mohanlal', 'Pulimurugan', 'Oppam', 'Kasaba', 'Jacobinte Swargarajyam', 'Kali', 'Ezra', 'Godha',
            'Virus'
        ]
        
        # Combine all movies
        all_movies = telugu_movies + tamil_movies + hindi_movies + malayalam_movies
        
        # Sample movie data with exactly 100 entries
        movies_data = {
            'movie_id': list(range(1, 101)),
            'title': all_movies,
            'language': (['Telugu'] * 25 + ['Tamil'] * 25 + ['Hindi'] * 25 + ['Malayalam'] * 25),
            'genre': np.random.choice([
                'Action', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Horror', 
                'Family', 'Historical', 'Fantasy', 'Crime'
            ], 100).tolist(),
            'director': np.random.choice([
                'S.S. Rajamouli', 'Sukumar', 'Trivikram', 'Koratala Siva', 'Vamshi Paidipally',
                'Lokesh Kanagaraj', 'Nelson Dilipkumar', 'Shankar', 'Pa. Ranjith', 'Karthik Subbaraj',
                'Rohit Shetty', 'Kabir Khan', 'Rajkumar Hirani', 'Aanand L. Rai', 'Imtiaz Ali',
                'Dileesh Pothan', 'Lijo Jose Pellissery', 'Aashiq Abu', 'Mahesh Narayanan', 'Midhun Manuel Thomas'
            ], 100).tolist(),
            'year': np.random.randint(2015, 2024, 100).tolist(),
            'rating': np.round(np.random.uniform(6.0, 9.5, 100), 1).tolist(),
            'votes': np.random.randint(10000, 500000, 100).tolist()
        }
        
        # Create descriptions combining genre and some keywords
        descriptions = []
        for i in range(100):
            genre = movies_data['genre'][i]
            lang = movies_data['language'][i]
            desc_templates = {
                'Action': f"{lang} action-packed thriller with intense fight sequences and drama",
                'Drama': f"Emotional {lang} family drama with strong performances and storytelling",
                'Romance': f"Heartwarming {lang} romantic story with beautiful music and chemistry",
                'Comedy': f"Hilarious {lang} comedy entertainer with witty dialogues and humor",
                'Thriller': f"Gripping {lang} psychological thriller with suspense and mystery",
                'Horror': f"Spine-chilling {lang} horror film with supernatural elements",
                'Family': f"Feel-good {lang} family entertainer suitable for all ages",
                'Historical': f"Epic {lang} historical drama with grand visuals and war sequences",
                'Fantasy': f"Magical {lang} fantasy adventure with visual effects and mythology",
                'Crime': f"Dark {lang} crime drama with investigation and underworld elements"
            }
            descriptions.append(desc_templates.get(genre, f"{lang} movie with engaging storyline"))
        
        movies_data['description'] = descriptions
        
        # Verify all arrays have the same length
        lengths = {key: len(value) for key, value in movies_data.items()}
        print(f"Array lengths: {lengths}")
        
        # Ensure all arrays are exactly 100 elements
        for key, value in movies_data.items():
            if len(value) != 100:
                print(f"Warning: {key} has {len(value)} elements, expected 100")
                if len(value) < 100:
                    # Pad with last element if too short
                    movies_data[key] = value + [value[-1]] * (100 - len(value))
                else:
                    # Truncate if too long
                    movies_data[key] = value[:100]
        
        self.movies_df = pd.DataFrame(movies_data)
        
        # Generate ratings data
        n_users = 1000
        n_ratings = 15000
        
        ratings_data = {
            'user_id': np.random.randint(1, n_users + 1, n_ratings),
            'movie_id': np.random.randint(1, 101, n_ratings),
            'rating': np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.1, 0.2, 0.35, 0.3])
        }
        
        self.ratings_df = pd.DataFrame(ratings_data)
        self.ratings_df = self.ratings_df.drop_duplicates(['user_id', 'movie_id'])
        
        print("Sample dataset created successfully!")
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        print(f"Languages: {self.movies_df['language'].value_counts().to_dict()}")
        
    def explore_data(self):
        """Perform exploratory data analysis"""
        
        if self.movies_df is None:
            print("Please create or load data first!")
            return
            
        plt.figure(figsize=(15, 12))
        
        # Language distribution
        plt.subplot(2, 3, 1)
        self.movies_df['language'].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribution of Movies by Language')
        plt.xticks(rotation=45)
        
        # Genre distribution
        plt.subplot(2, 3, 2)
        self.movies_df['genre'].value_counts().plot(kind='bar', color='lightgreen')
        plt.title('Distribution of Movies by Genre')
        plt.xticks(rotation=45)
        
        # Year distribution
        plt.subplot(2, 3, 3)
        self.movies_df['year'].hist(bins=10, color='orange', alpha=0.7)
        plt.title('Movies by Year')
        plt.xlabel('Year')
        
        # Rating distribution
        plt.subplot(2, 3, 4)
        self.movies_df['rating'].hist(bins=15, color='pink', alpha=0.7)
        plt.title('Movie Ratings Distribution')
        plt.xlabel('Rating')
        
        # User ratings distribution
        plt.subplot(2, 3, 5)
        self.ratings_df['rating'].value_counts().sort_index().plot(kind='bar', color='purple')
        plt.title('User Ratings Distribution')
        plt.xlabel('Rating')
        
        # Average rating by language
        plt.subplot(2, 3, 6)
        avg_rating_by_lang = self.movies_df.groupby('language')['rating'].mean()
        avg_rating_by_lang.plot(kind='bar', color='red', alpha=0.7)
        plt.title('Average Rating by Language')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Display statistics
        print("\n=== DATA STATISTICS ===")
        print(f"Total Movies: {len(self.movies_df)}")
        print(f"Total Users: {self.ratings_df['user_id'].nunique()}")
        print(f"Total Ratings: {len(self.ratings_df)}")
        print(f"Average Movie Rating: {self.movies_df['rating'].mean():.2f}")
        print(f"Rating Range: {self.movies_df['rating'].min():.1f} - {self.movies_df['rating'].max():.1f}")
        
        print("\n=== TOP RATED MOVIES BY LANGUAGE ===")
        for lang in self.movies_df['language'].unique():
            top_movie = self.movies_df[self.movies_df['language'] == lang].nlargest(1, 'rating')
            print(f"{lang}: {top_movie['title'].iloc[0]} ({top_movie['rating'].iloc[0]})")
    
    def build_content_based_recommender(self):
        """Build content-based recommendation system"""
        
        print("Building content-based recommender...")
        
        # Combine features for TF-IDF
        self.movies_df['combined_features'] = (
            self.movies_df['genre'] + ' ' + 
            self.movies_df['director'] + ' ' +
            self.movies_df['language'] + ' ' +
            self.movies_df['description']
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', lowercase=True, max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined_features'])
        
        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(self.tfidf_matrix)
        
        print("Content-based recommender built successfully!")
        
    def build_collaborative_filtering(self):
        """Build collaborative filtering recommendation system"""
        
        print("Building collaborative filtering recommender...")
        
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        # Apply SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=50, random_state=42)
        user_movie_svd = self.svd_model.fit_transform(self.user_movie_matrix)
        
        print("Collaborative filtering recommender built successfully!")
        
    def get_content_recommendations(self, movie_title, language_filter=None, n_recommendations=10):
        """Get content-based recommendations"""
        
        if self.content_similarity is None:
            print("Please build content-based recommender first!")
            return []
            
        # Find movie index
        try:
            movie_idx = self.movies_df[self.movies_df['title'] == movie_title].index[0]
        except IndexError:
            print(f"Movie '{movie_title}' not found!")
            return []
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.content_similarity[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get recommendations
        recommendations = []
        for idx, score in similarity_scores[1:]:  # Skip the movie itself
            movie_info = self.movies_df.iloc[idx]
            
            # Apply language filter if specified
            if language_filter and movie_info['language'] not in language_filter:
                continue
                
            recommendations.append({
                'title': movie_info['title'],
                'language': movie_info['language'],
                'genre': movie_info['genre'],
                'rating': movie_info['rating'],
                'year': movie_info['year'],
                'similarity_score': score
            })
            
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations
    
    def get_collaborative_recommendations(self, user_id, language_filter=None, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        
        if self.user_movie_matrix is None:
            print("Please build collaborative filtering recommender first!")
            return []
            
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found!")
            return []
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Find similar users using cosine similarity
        user_similarity = cosine_similarity([user_ratings], self.user_movie_matrix)[0]
        similar_users_idx = np.argsort(user_similarity)[::-1][1:11]  # Top 10 similar users
        
        # Get recommendations based on similar users
        movie_scores = {}
        for similar_user_idx in similar_users_idx:
            similar_user_id = self.user_movie_matrix.index[similar_user_idx]
            similar_user_ratings = self.user_movie_matrix.loc[similar_user_id]
            
            for movie_id, rating in similar_user_ratings.items():
                if rating > 0 and user_ratings[movie_id] == 0:  # User hasn't rated this movie
                    if movie_id not in movie_scores:
                        movie_scores[movie_id] = []
                    movie_scores[movie_id].append(rating * user_similarity[similar_user_idx])
        
        # Calculate average scores
        for movie_id in movie_scores:
            movie_scores[movie_id] = np.mean(movie_scores[movie_id])
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for movie_id, score in sorted_recommendations:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id].iloc[0]
            
            # Apply language filter if specified
            if language_filter and movie_info['language'] not in language_filter:
                continue
                
            recommendations.append({
                'title': movie_info['title'],
                'language': movie_info['language'],
                'genre': movie_info['genre'],
                'rating': movie_info['rating'],
                'year': movie_info['year'],
                'predicted_score': score
            })
            
            if len(recommendations) >= n_recommendations:
                break
                
        return recommendations
    
    def get_hybrid_recommendations(self, user_id=None, movie_title=None, language_filter=None, n_recommendations=10):
        """Get hybrid recommendations combining content and collaborative filtering"""
        
        content_recs = []
        collab_recs = []
        
        # Get content-based recommendations
        if movie_title:
            content_recs = self.get_content_recommendations(
                movie_title, language_filter, n_recommendations * 2
            )
        
        # Get collaborative filtering recommendations
        if user_id:
            collab_recs = self.get_collaborative_recommendations(
                user_id, language_filter, n_recommendations * 2
            )
        
        # Combine and rank recommendations
        hybrid_scores = {}
        
        # Add content-based scores
        for i, rec in enumerate(content_recs):
            title = rec['title']
            content_score = rec['similarity_score'] * (1 - i * 0.05)  # Decay factor
            hybrid_scores[title] = hybrid_scores.get(title, {})
            hybrid_scores[title]['content_score'] = content_score
            hybrid_scores[title]['info'] = rec
        
        # Add collaborative scores
        for i, rec in enumerate(collab_recs):
            title = rec['title']
            collab_score = rec['predicted_score'] * (1 - i * 0.05)  # Decay factor
            hybrid_scores[title] = hybrid_scores.get(title, {})
            hybrid_scores[title]['collab_score'] = collab_score
            hybrid_scores[title]['info'] = rec
        
        # Calculate hybrid scores
        for title in hybrid_scores:
            content_score = hybrid_scores[title].get('content_score', 0) * 0.4
            collab_score = hybrid_scores[title].get('collab_score', 0) * 0.6
            hybrid_scores[title]['hybrid_score'] = content_score + collab_score
        
        # Sort by hybrid score
        sorted_hybrid = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1]['hybrid_score'], 
            reverse=True
        )
        
        # Format recommendations
        recommendations = []
        for title, scores in sorted_hybrid[:n_recommendations]:
            rec_info = scores['info']
            rec_info['hybrid_score'] = scores['hybrid_score']
            recommendations.append(rec_info)
            
        return recommendations
    
    def get_popular_movies_by_language(self, language, top_n=10):
        """Get most popular movies by language"""
        
        lang_movies = self.movies_df[self.movies_df['language'] == language]
        popular_movies = lang_movies.nlargest(top_n, ['rating', 'votes'])
        
        return popular_movies[['title', 'genre', 'rating', 'year', 'director']].to_dict('records')
    
    def search_movies(self, query, language_filter=None):
        """Search movies by title or description"""
        
        df = self.movies_df.copy()
        
        if language_filter:
            df = df[df['language'].isin(language_filter)]
        
        # Search in title and description
        mask = (
            df['title'].str.contains(query, case=False, na=False) |
            df['description'].str.contains(query, case=False, na=False) |
            df['genre'].str.contains(query, case=False, na=False)
        )
        
        results = df[mask].sort_values('rating', ascending=False)
        return results[['title', 'language', 'genre', 'rating', 'year']].to_dict('records')
    
    def print_recommendations(self, recommendations, title="Recommendations"):
        """Pretty print recommendations"""
        
        print(f"\n=== {title.upper()} ===")
        if not recommendations:
            print("No recommendations found!")
            return
            
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']}")
            print(f"   Language: {rec['language']} | Genre: {rec['genre']}")
            print(f"   Rating: {rec['rating']} | Year: {rec['year']}")
            
            if 'similarity_score' in rec:
                print(f"   Similarity Score: {rec['similarity_score']:.3f}")
            if 'predicted_score' in rec:
                print(f"   Predicted Score: {rec['predicted_score']:.3f}")
            if 'hybrid_score' in rec:
                print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")

# Demo usage
def main():
    print("🎬 Indian Cinema Movie Recommendation System")
    print("Supporting Telugu, Tamil, Hindi, Malayalam movies")
    
    # Initialize recommender
    recommender = IndianCinemaRecommender()
    
    # Create sample data
    recommender.create_sample_data()
    
    # Explore data
    print("\n" + "="*50)
    print("EXPLORING DATA")
    print("="*50)
    recommender.explore_data()
    
    # Build recommendation models
    print("\n" + "="*50)
    print("BUILDING RECOMMENDATION MODELS")
    print("="*50)
    recommender.build_content_based_recommender()
    recommender.build_collaborative_filtering()
    
    # Test content-based recommendations
    print("\n" + "="*50)
    print("CONTENT-BASED RECOMMENDATIONS")
    print("="*50)
    
    # Get recommendations for a Telugu movie
    content_recs = recommender.get_content_recommendations(
        'Baahubali: The Beginning', 
        language_filter=['Telugu', 'Tamil'],
        n_recommendations=5
    )
    recommender.print_recommendations(content_recs, "Similar to Baahubali: The Beginning")
    
    # Test collaborative filtering
    print("\n" + "="*50)
    print("COLLABORATIVE FILTERING RECOMMENDATIONS")
    print("="*50)
    
    collab_recs = recommender.get_collaborative_recommendations(
        user_id=1,
        language_filter=['Hindi', 'Telugu'],
        n_recommendations=5
    )
    recommender.print_recommendations(collab_recs, "Recommendations for User 1")
    
    # Test hybrid recommendations
    print("\n" + "="*50)
    print("HYBRID RECOMMENDATIONS")
    print("="*50)
    
    hybrid_recs = recommender.get_hybrid_recommendations(
        user_id=1,
        movie_title='RRR',
        language_filter=['Telugu', 'Tamil', 'Hindi'],
        n_recommendations=8
    )
    recommender.print_recommendations(hybrid_recs, "Hybrid Recommendations")
    
    # Popular movies by language
    print("\n" + "="*50)
    print("POPULAR MOVIES BY LANGUAGE")
    print("="*50)
    
    for language in ['Telugu', 'Tamil', 'Hindi', 'Malayalam']:
        popular = recommender.get_popular_movies_by_language(language, top_n=3)
        print(f"\n🏆 Top {language} Movies:")
        for i, movie in enumerate(popular, 1):
            print(f"   {i}. {movie['title']} ({movie['rating']}) - {movie['genre']}")
    
    # Search functionality
    print("\n" + "="*50)
    print("MOVIE SEARCH")
    print("="*50)
    
    search_results = recommender.search_movies('action', language_filter=['Telugu', 'Tamil'])
    print("\n🔍 Search Results for 'action':")
    for movie in search_results[:5]:
        print(f"   • {movie['title']} ({movie['language']}) - {movie['rating']}")
    
    print("\n✅ Demo completed successfully!")
    print("You can now use this system to get personalized movie recommendations!")

if __name__ == "__main__":
    main()
    