from mrjob.job import MRJob
from mrjob.step import MRStep

N = 2000

class MRTop_N_Similarity(MRJob):

    def steps(self):
        return [
            MRStep(mapper=self.sim_movies,
                   reducer=self.top_sim_K),
            ]
    
    def sim_movies(self, _, line):
        movie_i, movie_j, sim_i_j = line.split()
        yield movie_i, (movie_j, sim_i_j)

    def top_sim_K(self, key, values):
        sim_mov = []
        for sim_movie, sim_val in values:
            sim_mov.append((sim_movie, sim_val))

        sim_mov.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(N, len(sim_mov))):
            yield (key, sim_mov[i][0]), sim_mov[i][1]

if __name__ == '__main__':
    MRTop_N_Similarity.run()