import numpy as np
import random
import itertools
import os
import sys
import NK_model

class gc():
    def __init__(self, binding_landscape=None, 
                 growth_landscape=None, possible_genotypes=None, mu=0, capacity=1000,
                 selection_factor=0.99, random_seed=233423, clone_number=50):
        self.population_collapse = False
        np.random.seed(seed=random_seed)
        #whole genotype space
        self.possible_genotypes = possible_genotypes
        #landscape is dict landscape[genotype] = [fitness, is_local_peak, is_global_peak]
        self.binding_landscape = binding_landscape
        self.growth_landscape = growth_landscape
        #select a number of founder genotypes
        self.nInd = clone_number
        picked_indices = np.random.choice(np.arange(len(possible_genotypes)), clone_number, replace=True)
        self.population = np.take(possible_genotypes, picked_indices, axis=0)
        #here fitness is independent of a phenotype, maybe we want to change it in the future
        self.fitness = np.array(list(map(lambda x: self.assign_value(x, 'growth'), self.population)))
        #affinity is defined by the landscape
        self.affinity = np.array(list(map(lambda x: self.assign_value(x, 'binding'), self.population)))
        #mutation probability
        self.mu = mu
        #environmental capacity (max size of a population)
        self.K = capacity
        #how harsh the selection is for binding
        #(the percentage of a population that survives after selection phase)
        self.selection_factor = selection_factor
        #history of my population
        self.history = [self.population]
        self.fitness_history = [self.fitness]
        self.affinity_history = [self.affinity]
        #for debug
        self.temp = []
        
    def proliferation(self):
        #Wright-Fisher model
        #each individual can be chosen to the next generation according to its fitness i.e. growth rate
        #we first calculate the population size in the next generation and then sample from the current poplation
        #the population size is growing logistically
        #growth rate
        r = 1
        delta_N = r*self.nInd*(1-self.nInd/self.K)
        new_N = np.rint(self.nInd + delta_N).astype(int)
        #not sure if it's a fair model, maybe should implement it as honest birth-death process
        new_indices = np.random.choice(np.arange(self.nInd), new_N, 
                                       replace=True, p=self.fitness/self.fitness.sum())
        self.fitness = np.take(self.fitness, new_indices)
        self.affinity = np.take(self.affinity, new_indices)
        self.population = np.take(self.population, new_indices, axis=0)
        self.nInd = new_N
        #potentially, we can allow mutation only after certain number of generations passed
        self.mutation()
    
    def selection(self):
        #decreasing population size
        #factor can be changed in the future, should be one of tunable parameters
        new_N = np.rint(self.nInd * self.selection_factor).astype(int)
        new_indices = np.random.choice(np.arange(self.nInd), new_N, 
                                       replace=False, p=self.affinity/self.affinity.sum())
        self.nInd = new_N
        self.fitness = np.take(self.fitness, new_indices)
        self.affinity = np.take(self.affinity, new_indices)
        self.population = np.take(self.population, new_indices, axis=0)
    
    def mutation(self):
        #picking up genotypes that will be mutated, maybe should implement mutation process differently
        mask = np.random.choice([True, False], size=self.nInd, p=[mu, 1-mu])
        for index in np.where(mask)[0]:
            picked_genotype = self.population[index]
            mutated_site = np.random.choice(np.arange(len(picked_genotype)), 1)
            picked_genotype[mutated_site] = 1 - picked_genotype[mutated_site]
            new_phenotype = self.assign_value(picked_genotype, 'binding')
            new_fitness = self.assign_value(picked_genotype, 'growth')
            self.affinity[index] = new_phenotype
            self.fitness[index] = new_fitness
    
    def assign_value(self, genotype, which_landscape, check_local_peak=False, check_global_peak=False):
        #according to fitness landscape generated with NK model
        if which_landscape == 'growth':
            #for now, there is only one peak in the fitness landscape
            if type(self.growth_landscape[tuple(genotype)]) is tuple:
                fitness = self.growth_landscape[tuple(genotype)][0]
            else:
                fitness = self.growth_landscape[tuple(genotype)]
            return fitness
        elif which_landscape == 'binding':
            if check_local_peak:
                return self.binding_landscape[tuple(genotype)][1]
            elif check_global_peak:
                return self.binding_landscape[tuple(genotype)][2]
            else:
                fitness = self.binding_landscape[tuple(genotype)][0]
                return fitness
    
    def evolve(self, nprol, nselection=1, ncyc=1):
        for j in range(ncyc):
            for i in range(nprol):
                self.proliferation()
                self.history.append(self.population)
                self.fitness_history.append(self.fitness)
                self.affinity_history.append(self.affinity)
            for i in range(nselection):
                self.selection()
                if self.population_collapse:
                    print(f'Population collapse after {j} cycle')
                    break
                self.history.append(self.population)
                self.fitness_history.append(self.fitness)
                self.affinity_history.append(self.affinity)

    def track_diversity(self, verbose=False):
        if not verbose:
            return self.history
        else:
            unique_elements = []
            dominant_frequencies = []
            for p in self.history:
                unique_elements.append(len(np.unique(p, axis=0)))
                dominant_freq = np.max(np.unique(p, return_counts=True, axis=0)[1])/len(p)
                dominant_frequencies.append(dominant_freq)
            self.unique_elements = np.array(unique_elements)
            self.dominant_frequencies = np.array(dominant_frequencies)
        #how many sequences are resting on local peaks
        local_peaks = np.array(list(map(lambda x: self.assign_value(x,'binding',check_local_peak=True), 
                                        self.population))).sum()
        #number of sequences at local peaks at the beginning
        local_peaks_start = np.array(list(map(lambda x: self.assign_value(x,'binding',check_local_peak=True), 
                                              self.history[0]))).sum()
        #print(f'{local_peaks} sequences at the local peaks')
        #how many local peaks were reached??
        
        #if global peak is reached
        global_peak = np.array(list(map(lambda x: self.assign_value(x,'binding',check_global_peak=True), 
                                        self.population))).sum()
        global_peak_start = np.array(list(map(lambda x: self.assign_value(x,'binding',check_global_peak=True), 
                                              self.history[0]))).sum()
        #print(f'{global_peak} sequences at the global peak')
        #to measure the diversity of my genotypes taking into account a distance between them
        
def run_gc(nprol, nselection, ncyc, binding_landscape_dict, growth_landscape_dict, 
           possible_genotypes, K, N, mu=0.01, capacity=1000, selection_factor=0.99, fitness_affinity_corr=0, 
           clone_number=50, landscape_index=0):
    #creating GC and running its evolution
    pop = gc(binding_landscape=binding_landscape_dict, growth_landscape=growth_landscape_dict, 
             possible_genotypes=possible_genotypes, mu=mu, capacity=capacity,
             selection_factor=selection_factor, random_seed=landscape_index, clone_number=clone_number)
    pop.evolve(nprol, nselection=nselection, ncyc=ncyc)
    pop.track_diversity(verbose=True)
    #writing down the results
    unique_elements = pop.unique_elements
    dominant_frequencies = pop.dominant_frequencies
    fitness_history = np.array([i.mean() for i in pop.fitness_history])
    affinity_history = np.array([i.mean() for i in pop.affinity_history])
    genotype_history = np.array(list(itertools.zip_longest(*pop.history, fillvalue=np.array([-1]*N))))
    np.save(f'germinal_centers_WF/gc_nprol={nprol},nselection={nselection},ncyc={ncyc},N={N},K={K},landscape_id={landscape_index},selection_factor={selection_factor},fitness_affinity_corr={fitness_affinity_corr},mu={mu},unique_labels.npy', unique_elements)
    np.save(f'germinal_centers_WF/gc_nprol={nprol},nselection={nselection},ncyc={ncyc},N={N},K={K},landscape_id={landscape_index},selection_factor={selection_factor},fitness_affinity_corr={fitness_affinity_corr},mu={mu},dominant_frequencies.npy', dominant_frequencies)
    np.save(f'germinal_centers_WF/gc_nprol={nprol},nselection={nselection},ncyc={ncyc},N={N},K={K},landscape_id={landscape_index},selection_factor={selection_factor},fitness_affinity_corr={fitness_affinity_corr},mu={mu},fitness_history.npy', fitness_history)
    np.save(f'germinal_centers_WF/gc_nprol={nprol},nselection={nselection},ncyc={ncyc},N={N},K={K},landscape_id={landscape_index},selection_factor={selection_factor},fitness_affinity_corr={fitness_affinity_corr},mu={mu},phenotype_history.npy', affinity_history)
    np.save(f'germinal_centers_WF/gc_nprol={nprol},nselection={nselection},ncyc={ncyc},N={N},K={K},landscape_id={landscape_index},selection_factor={selection_factor},fitness_affinity_corr={fitness_affinity_corr},mu={mu},genotype_history.npy', genotype_history)
    
def make_dict_from_nparray(nk_landscape, N, landscape_index=0):
    genotypes = nk_landscape[landscape_index][:, :N]
    values = nk_landscape[landscape_index][:, -3:]
    landscape_dict = dict(zip(map(tuple, genotypes), map(tuple, values)))
    return landscape_dict

def generate_with_corrcoef(arr1, p):
    n = len(arr1)
    # generate noise
    noise = np.random.uniform(0, 1, n)
    # least squares linear regression for noise = m*arr1 + c
    m, c = np.linalg.lstsq(np.vstack([arr1, np.ones(n)]).T, noise)[0]
    # residuals have 0 correlation with arr1
    residuals = noise - (m*arr1 + c)
    # the right linear combination a*arr1 + b*residuals
    a = p * np.std(residuals)
    b = (1 - p**2)**0.5 * np.std(arr1)
    arr2 = a*arr1 + b*residuals
    # return a scaled/shifted result to have the same mean/sd as arr1
    # this doesn't change the correlation coefficient
    arr2_normalized = np.mean(arr1) + (arr2 - np.mean(arr2)) * np.std(arr1) / np.std(arr2)
    return (arr2_normalized - arr2_normalized.min())/(arr2_normalized.max() - arr2_normalized.min())
    
def generate_correlated_landscape(landscape_to_correlate, N, landscape_index=0, corrcoef=0.5):
    genotypes = landscape_to_correlate[landscape_index][:, :N]
    values = landscape_to_correlate[landscape_index][:, -3]
    correlated_values = generate_with_corrcoef(values, corrcoef)
    landscape_dict = dict(zip(map(tuple, genotypes), correlated_values))
    return landscape_dict

if __name__ == '__main__':
    #type of an interaction matrix in NK model used, 1 is random
    which_imatrix = 1
    #K parameter for fitness landscape, for now it's constant
    K_growth = 0
    #number of landscapes
    i = 100
    #N parameter in NK model (number of sites)
    N = 10
    #environmental capacity of GC
    capacity = 1000
    #changeble parameters 
    nprol = sys.argv[1]
    nselection = sys.argv[2]
    ncyc = sys.argv[3]
    #mutation rate
    mu = float(sys.argv[4])
    selection_factor = float(sys.argv[5])
    #binding-fitness correlation
    fitness_affinity_corr = float(sys.argv[6])
    #loading fitness landscape
    if not fitness_affinity_corr:
        growth_landscape = np.load(f'NK_workshop/NK_land_type_{which_imatrix}_K_{K_growth}_i_{i}.npy')
        growth_landscape_dict = make_dict_from_nparray(growth_landscape, N)
    #all possible genotypes space
    possible_genotypes = list(itertools.product(range(2), repeat=N))
    os.makedirs('./germinal_centers_WF/', exist_ok=True)
    for K_bind in range(1, 10):
        #print(f'K={K_bind}')
        #loading binding landscape
        binding_landscape = np.load(f'NK_workshop/NK_land_type_{which_imatrix}_K_{K_bind}_i_{i}.npy')
        for landscape_index in range(0, i):
            binding_landscape_dict = make_dict_from_nparray(binding_landscape, N, landscape_index=landscape_index)
            if fitness_affinity_corr:
                growth_landscape_dict = generate_correlated_landscape(binding_landscape, N, 
                                                                      landscape_index=landscape_index, 
                                                                      corrcoef=fitness_affinity_corr)
            run_gc(int(nprol), int(nselection), int(ncyc), binding_landscape_dict, growth_landscape_dict, 
                   possible_genotypes, K_bind, N, mu=mu, selection_factor=selection_factor, 
                   fitness_affinity_corr=fitness_affinity_corr, 
                   landscape_index=landscape_index)