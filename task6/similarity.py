from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import combinations
from math import sqrt

class MRSimilarity(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.user_movie,
                   reducer=self.user_feature),
            MRStep(mapper=self.movie_pair,
                  reducer=self.calculate_similarity)
        ]
    
    def user_movie(self, _, line):
        '''
        key = user_id
        values = movie_id, rating
        '''
        user_id, item_id, rating = line.split('::')
        yield int(user_id), (int(item_id), float(rating))

    def user_feature(self, user_id, ratings):
        '''
        key = user_id
        values = movei_count, rating_sum, movieID_rating
        movieID_rating == [(movieID, rating)]
        '''
        movie_count = 0
        rating_sum = 0
        movieID_rating = []
        for movieID, rating in ratings:
            movie_count += 1
            rating_sum += rating
            movieID_rating.append((movieID, rating))

        yield user_id, (movie_count, rating_sum, movieID_rating)
    
    def movie_pair(self, _, values):
        '''
        key = pair of movie_1, movie_2
        value = pair of movie_1 and movie_2 ratings and mean rating for user
        '''
        movie_count, rating_sum, movieID_rating = values
        for movie1, movie2 in combinations(movieID_rating, 2):
            yield (movie1[0], movie2[0]), (movie1[1], movie2[1], rating_sum / movie_count)
    
    def calculate_similarity(self, movie_pairs, rating_pairs):
        '''
        key = (i, j)
        value = sim(i, j)
        '''
        movie_i, movie_j = movie_pairs
        sum_numerator, sum_detominator_i, sum_detominator_j = 0.0, 0.0, 0.0
        for r_i, r_j, r_hat in rating_pairs:
            sum_numerator += (r_i - r_hat) * (r_j - r_hat)
            sum_detominator_i += (r_i - r_hat) ** 2
            sum_detominator_j += (r_j - r_hat) ** 2
        
        if ((sum_detominator_i == 0.0) or (sum_detominator_j == 0)):
            sim_i_j = 0
        else:
            sim_i_j = sum_numerator / ((sqrt(sum_detominator_i) * sqrt(sum_detominator_j)))
        
        if sim_i_j > 0.0:
            print '%s %s %f' % (movie_i, movie_j, sim_i_j)
            
if __name__ == '__main__':
    MRSimilarity.run()