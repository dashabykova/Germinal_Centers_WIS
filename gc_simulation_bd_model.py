import numpy as np
import random
import math
import sys
import os
from joblib import Parallel, delayed

class Bcell():
    def __init__(self, lambda_val, mu, diff, dt, n_max, lambda_base, affinity, corr=0):
        self.lambda_val = lambda_val
        self.mu = mu
        self.diff = diff
        self.dt = dt
        self.N = None
        self.parent_pos = None
        self.n_max = n_max
        self.lambda_base = lambda_base
        self.affinity = affinity
        #correlation between growth rate and affinity
        self.corr = corr
    
    def birth(self):
        kid = self.birth_kid_mut()
        return kid
    
    def birth_tc(self, lambda_mean_all, selection_mode='birth'):
        kid = self.birth_kid_mut_norm(lambda_mean_all, selection_mode=selection_mode)
        return kid
    
    def birth_tg(self):
        p = self.dt * self.lambda_val
        if random.random() <= p:
            kid = Bcell(self.lambda_val, self.mu, self.diff, self.dt, self.n_max, self.lambda_base, self.affinity, corr=self.corr)
        else:
            kid = None
        return kid
    
    def birth_kid_mut(self):
        p = self.dt * self.lambda_val
        if random.random() <= p:
            #mut_std = math.sqrt(2 * self.diff)
            #lambda_son = abs(self.lambda_val + random.normalvariate(0, mut_std))
            #affinity_son = abs(self.affinity + random.normalvariate(0, mut_std))
            #to impement the correlation between affinity and growth rate
            mvn = np.random.multivariate_normal(mean=[0, 0], cov=[[2 * self.diff, self.corr*2 * self.diff], 
                                                [self.corr*2 * self.diff, 2 * self.diff]], size=1)
            lambda_std, affinity_std = mvn[0,0], mvn[0,1]
            lambda_son = abs(self.lambda_val + lambda_std)
            affinity_son = abs(self.affinity + affinity_std)
            kid = Bcell(lambda_son, self.mu, self.diff, self.dt, self.n_max, self.lambda_base, affinity_son, corr=self.corr)
        else:
            kid = None
        return kid
    
    def birth_kid_mut_norm(self, lambda_mean_all, selection_mode='birth'):
        if (selection_mode == 'birth') | (selection_mode == 'mixed'):
            lambda_val = self.lambda_base * self.lambda_val / lambda_mean_all
        elif selection_mode == 'death':
            #lambda_val = self.lambda_base
            #or no proliferation at this stage
            return None
        p = self.dt * lambda_val
        if random.random() <= p:
            #mut_std = math.sqrt(2 * self.diff)
            #lambda_son = abs(self.lambda_val + random.normalvariate(0, mut_std))
            #affinity_son = abs(self.affinity + random.normalvariate(0, mut_std))
            mvn = np.random.multivariate_normal(mean=[0, 0], cov=[[2 * self.diff, self.corr*2 * self.diff], 
                                                [self.corr*2 * self.diff, 2 * self.diff]], size=1)
            lambda_std, affinity_std = mvn[0,0], mvn[0,1]
            lambda_son = abs(self.lambda_val + lambda_std)
            affinity_son = abs(self.affinity + affinity_std)
            kid = Bcell(lambda_son, self.mu, self.diff, self.dt, self.n_max, self.lambda_base, affinity_son, corr=self.corr)
        else:
            kid = None
        return kid
    
    def death_t(self):
        p = self.dt * self.mu
        if random.random() <= p:
            d_flag = 1
        else:
            d_flag = 0
        return d_flag
    
    def death(self, lambda_mean_all, ntot, selection_mode='birth'):
        r = self.lambda_base - self.mu
        #self.mu can be dependent on affinity
        if selection_mode == 'birth':
            m_ntot = self.mu + r * ntot / self.n_max
        elif (selection_mode == 'death') | (selection_mode == 'mixed'):
            m_ntot = np.exp(-self.affinity + 1) + r * ntot / self.n_max
        p = self.dt * m_ntot
        if random.random() <= p:
            d_flag = 1
        else:
            d_flag = 0
        return d_flag

def LambdaMean(CellsArrSp, NspArr):
    lambda_all = []
    for n in NspArr:
        lambda_all.extend([i.lambda_val for i in CellsArrSp[n-1]])
    
    if len(lambda_all) == 0:
        return 0
    
    lambda_mean_all = sum(lambda_all) / len(lambda_all)
    return lambda_mean_all

def BD_step(CellsArr, lambdaMeanAll, MuFlag, Ntot, selection_mode='birth'):
    Deadidx = []
    for i in range(len(CellsArr)):
        if MuFlag == 0:
            Kid = CellsArr[i].birth_tg()
        else:
            # Kid = CellsArr[i].Birth()
            Kid = CellsArr[i].birth_tc(lambdaMeanAll, selection_mode=selection_mode)

        if Kid is not None:
            CellsArr.append(Kid)

        if MuFlag == 0:
            Dflag = CellsArr[i].death_t()
        else:
            Dflag = CellsArr[i].death(lambdaMeanAll, Ntot, selection_mode=selection_mode)
            # Dflag = CellsArr[i].DeathTInterCloneCom(lambdaMeanAll, Ntot, len(CellsArr))

        if Dflag:
            Deadidx.append(i)
    idx = list(range(len(CellsArr)))
    idxSur = list(set(idx) - set(Deadidx))
    CellsArr = [CellsArr[i] for i in idxSur]
    return CellsArr

def evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep, selection_mode='birth'):    
    Ntot = sum(len(cells) for cells in CellsArrSp)
    exitFlag = 0
    lambdaMeanAll = LambdaMean(CellsArrSp, NspArr)
    to_delete = []
    for n in NspArr:
        CellsArrSp[n - 1] = BD_step(CellsArrSp[n - 1], lambdaMeanAll, MuFlag, Ntot, 
                                    selection_mode=selection_mode)
        if len(CellsArrSp[n - 1]) == 0:
            to_delete.append(n)
    for i in to_delete:
        NspArr.remove(i)
    if MuFlag:
        temp = np.zeros(50)
        for n in NspArr:
            temp[n-1] = len(CellsArrSp[n - 1])
            #cellNumDyn[timestep, n - 1] = len(CellsArrSp[n - 1])
        cellNumDyn.append(temp)
    if len(NspArr) == 0:
        exitFlag = 1
    if MuFlag:
        lambdaPop.append({n: [cell.lambda_val for cell in CellsArrSp[n - 1]] for n in NspArr})
        affinityPop.append({n: [cell.affinity for cell in CellsArrSp[n - 1]] for n in NspArr})
    return exitFlag

def run_gc_model(dt, lambda_val, mu, affinity, Diff, Nsp, Tg, Tc, Nmax, nprol, nsel, ncyc, k, mixed=False, corr=0):
    FlagDeathStop = 0
    random.seed(k)
    
    #cellnumGrowth = np.zeros((runNum, Nsp))
    #cellnumSel = np.zeros((runNum, Nsp))
    #PopulationTrack = np.zeros((runNum, 2, Nsp))
    lambdaSim = []

    lambdaPop, mean_dom_lambda = [], []
    affinityPop, mean_dom_affinity, average_affinity = [], [], []
    NspArr = list(np.arange(1, Nsp + 1))
    exitFlag = 0
    #bcells belonging to a clone
    CellsArrSp = []
    for n in range(1, Nsp + 1):
        #everyone has the same affinity (birth rate) at the beginning
        bcArr = []
        bc = Bcell(lambda_val, mu, Diff, dt, Nmax, lambda_val, affinity, corr=corr)
        bcArr.append(bc)
        CellsArrSp.append(bcArr)
    
    #growth phase, death is not dependent on population size, birth is without mutation
    MuFlag = 0
    #cellNumDyn = np.zeros((len(np.arange(0, Tg + Tc + 17*dt, dt)), Nsp))
    #cellNumDyn = np.zeros((len(np.arange(0, Tg + dt, dt)), Nsp))
    cellNumDyn = []
    C = 0
    for t in np.arange(0, Tg + dt, dt):
        exitflag = evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep=C)
        C += 1
        if exitflag:
            break

    #competition phase
    MuFlag = 1
    #cellNumDyn = np.zeros((len(np.arange(0, Tc + 16*dt, dt)), Nsp))
    C = 0
    for cyc in range(ncyc):
        #proliferation phase
        for t in np.arange(0, nprol + dt, dt):
            if not mixed:
                exitflag = evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep=C, 
                                     selection_mode='birth')
            else:
                exitflag = evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep=C, 
                                     selection_mode='mixed')
            C += 1
            if exitflag:
                break
        #selection phase
        for t in np.arange(0, nsel + dt, dt):
            if not mixed:
                exitflag = evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep=C,
                                     selection_mode='death')
            else:
                exitflag = evolution(CellsArrSp, NspArr, MuFlag, lambdaPop, affinityPop, cellNumDyn, timestep=C,
                                     selection_mode='mixed')
            C += 1
            if exitflag:
                break
    cellNumDyn = np.vstack(cellNumDyn)
    expected_shape = (len(np.arange(0, nprol + dt, dt)) + len(np.arange(0, nsel + dt, dt)))*ncyc
    real_shape = cellNumDyn.shape[0]
    #print(f'real: {real_shape}, exp: {expected_shape}')
    if real_shape != expected_shape:
        cellNumDyn = np.pad(cellNumDyn, [(0, expected_shape - real_shape), (0, 0)], mode='constant', constant_values=0)
    #saving data
    #average affinity of all clones
    for gen in range(expected_shape):
        if len(affinityPop) > gen:
            if len(affinityPop[gen]):
                average_affinity.append(np.mean([a for n in affinityPop[gen] for a in affinityPop[gen][n]]))
            else:
                average_affinity.append(0)
        else:
            average_affinity.append(0)
    #dominant clone affinity and growth rate
    for gen, ind in enumerate(np.argmax(cellNumDyn, axis=1)):
        if cellNumDyn[gen, ind] == 0:
            mean_dom_lambda.append(0)
            mean_dom_affinity.append(0)
        else:
            mean_dom_lambda.append(np.mean(lambdaPop[gen][ind + 1]))
            mean_dom_affinity.append(np.mean(affinityPop[gen][ind + 1]))
    mean_dom_lambda = np.array(mean_dom_lambda)
    mean_dom_affinity = np.array(mean_dom_affinity)
    average_affinity = np.array(average_affinity)
    if not mixed:
        np.save(f'BD_model_output/bd_model_mean_dom_lambda_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', mean_dom_lambda)
        np.save(f'BD_model_output/bd_model_cellnum_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', cellNumDyn)
        np.save(f'BD_model_output/bd_model_mean_dom_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', mean_dom_affinity)
        np.save(f'BD_model_output/bd_model_mean_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', average_affinity)
    else:
        np.save(f'BD_model_output/bd_model_mixed_selection_mean_dom_lambda_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', mean_dom_lambda)
        np.save(f'BD_model_output/bd_model_mixed_selection_mean_dom_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', mean_dom_affinity)
        np.save(f'BD_model_output/bd_model_mixed_selection_cellnum_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', cellNumDyn)
        np.save(f'BD_model_output/bd_model_mixed_selection_mean_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', average_affinity)
    
if __name__ == '__main__':
    dt = 0.01
    #basal birth rate, days**-1
    lambda_val = 1.5
    #basal death rate, days**-1
    mu = 1
    #basal affinity (here just equals to basal birth rate)
    affinity = 1.5
    #diffusion coefficient (mutation effect), make this tunable
    Diff = 0.01
    #change to something larger in the future (200?)
    runNum = 100
    #number of initial clones
    Nsp = 50
    #time in the growth phase, days
    Tg = 6
    #time in the competition phase, days
    Tc = 16
    #2000 in the original paper
    Nmax = 1000
    #mixed model or not
    mixed = True
    #correlation between growth rate and affinity
    corr = -0.8
    #number of days spent in proliferation and selection phase
    nprol = int(sys.argv[1])
    nsel = int(sys.argv[2])
    ncyc = int(sys.argv[3])
    os.makedirs('./BD_model_output/summary/', exist_ok=True)
    print('ready to go')
    for k in range(runNum):
        run_gc_model(dt, lambda_val, mu, affinity, Diff, Nsp, Tg, Tc, Nmax, nprol, nsel, ncyc, k, mixed=mixed, corr=corr)
    print('calculated everything, ready to make output files')
    
    lambdas, cellnums, affinities, mean_affinities = [], [], [], []
    for k in range(runNum):
        if not mixed:
            mean_dom_lambda = np.load(f'BD_model_output/bd_model_mean_dom_lambda_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            mean_dom_affinity = np.load(f'BD_model_output/bd_model_mean_dom_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            mean_affinity = np.load(f'BD_model_output/bd_model_mean_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            cellNumDyn = np.load(f'BD_model_output/bd_model_cellnum_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
        else:
            mean_dom_lambda = np.load(f'BD_model_output/bd_model_mixed_selection_mean_dom_lambda_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            mean_dom_affinity = np.load(f'BD_model_output/bd_model_mixed_selection_mean_dom_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            mean_affinity = np.load(f'BD_model_output/bd_model_mixed_selection_mean_affinity_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
            cellNumDyn = np.load(f'BD_model_output/bd_model_mixed_selection_cellnum_run_{k}_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy')
        lambdas.append(mean_dom_lambda)
        affinities.append(mean_dom_affinity)
        mean_affinities.append(mean_affinity)
        cellnums.append(cellNumDyn)
    all_lambdas = np.vstack(lambdas)
    all_affinities = np.vstack(affinities)
    all_mean_affinities = np.vstack(mean_affinities)
    all_cell_nums = np.stack(cellnums, axis=2)
    if not mixed:
        np.save(f'BD_model_output/summary/bd_model_mean_dom_lambda_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_lambdas)
        np.save(f'BD_model_output/summary/bd_model_mean_dom_affinity_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_affinities)
        np.save(f'BD_model_output/summary/bd_model_cellnum_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_cell_nums)
        np.save(f'BD_model_output/summary/bd_model_mean_affinity_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_mean_affinities)
    else:
        np.save(f'BD_model_output/summary/bd_model_mixed_selection_mean_dom_lambda_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_lambdas)
        np.save(f'BD_model_output/summary/bd_model_mixed_selection_mean_dom_affinity_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_affinities)
        np.save(f'BD_model_output/summary/bd_model_mixed_selection_cellnum_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_cell_nums)
        np.save(f'BD_model_output/summary/bd_model_mixed_selection_mean_affinity_nprol={nprol},nselection={nsel},ncyc={ncyc},D={Diff},corr={corr}.npy', all_mean_affinities)