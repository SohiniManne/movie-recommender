import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.movie_recommender import IndianCinemaRecommender

class TestMovieRecommender(unittest.TestCase):
    
    def setUp(self):
        self.recommender = IndianCinemaRecommender()
        self.recommender.create_sample_data()
        self.recommender.build_content_based_recommender()
    
    def test_data_creation(self):
        self.assertIsNotNone(self.recommender.movies_df)
        self.assertTrue(len(self.recommender.movies_df) > 0)
    
    def test_content_recommendations(self):
        recs = self.recommender.get_content_recommendations('Baahubali: The Beginning', n_recommendations=5)
        self.assertTrue(len(recs) > 0)
        self.assertTrue(all('title' in rec for rec in recs))

if __name__ == '__main__':
    unittest.main()