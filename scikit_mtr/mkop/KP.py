# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 2019

@author: Jimena Ferreira FING-UdelaR
"""

__all__ = ['KP_run']

import multiprocessing.dummy as mp
import operator
import random
import warnings

from deap import base, creator, tools, gp

from .LinRegUtil import CleanLD, PLSR
from .Utils import tree2symb
from .basic_functions import *

warnings.filterwarnings("ignore", category=RuntimeWarning)


def Plan(population, toolbox, cxpb, mutpb):
    # create offspings
    from deap.algorithms import varAnd
    set = toolbox.select(population, len(population))
    offspring = varAnd(set, toolbox, cxpb, mutpb)
    return offspring


def Do(population, offspring):
    # create expanded population
    pop = population.copy()
    pop.extend(offspring)
    return pop


def Check(population, toolbox, y, ps, alfa, theta, y_max):
    # Build a model on ExandedPop, and calculate the importance of each individual
    # Return:
    #   B: Coefficients of each individual
    #   AdjR2: Adjusted R2 for each output variable
    #   population: List of individuals
    #   RMSE: Root Mean Squared Error
    #   RMSE_Norm: RMSE/y_max
    #   F: Evaluation of each point in each individual
    #   I: Index of the sorted terms using p-values

    n = y.shape[0]  # len(y) number of points
    m = y.shape[1]  # number of outlets
    # r= len(population) # population

    # Evaluate the individuals with an invalid fitness - O(r*nm)
    # if evaluation is inf, individual is discarded
    values = np.empty((0, n))
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.calculate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    i = 0
    while i < len(population):
        if np.all(np.isfinite(population[i].fitness.values)):
            values = np.append(values, np.array(population[i].fitness.values).reshape(1, -1), axis=0)
            i = i + 1
        else:
            population.remove(population[i])

    F = values.T
    r = len(population)  # Update population size

    # LD individuals are eliminated - max(O(nm),O(r))
    F, ind_dejar = CleanLD(F)
    aux_pop = toolbox.population(0)
    for pos in ind_dejar:
        aux_pop.append(population[pos])
    population = aux_pop
    r = len(ind_dejar)  # Update population size

    # Fit
    B, p_value, R2_1, AdjR2, RMSE, RMSE_Norm = PLSR(F, y, n, r, y_max, I=None)

    if len(B) > 0:
        # Select the most important individuals into ReducedPop O(r2)
        aux_pop = toolbox.population(0)
        F_aux = np.empty((n, 0))
        B_aux = np.empty((0, m))
        p_value_aux = np.empty((0, m))
        I = np.empty((0, m), dtype=int)
        for pos in range(r):
            pV = p_value[pos, :] <= alfa
            B_ = abs(B[pos, :]) > theta
            if np.all(np.isfinite(p_value[pos, :])) and np.any(pV * B_):
                aux_pop.append(population[pos])
                F_aux = np.append(F_aux, F[:, pos].reshape(-1, 1), axis=1)
                B_aux = np.append(B_aux, B[pos, :].reshape(1, -1), axis=0)
                p_value_aux = np.append(p_value_aux, p_value[pos, :].reshape(1, -1), axis=0)
                I = np.append(I, (pV * B_).reshape(1, -1), axis=0)
        population = aux_pop
        F = F_aux
        B = B_aux
        p_value = p_value_aux
        r = len(population)  # Update population size

        # If population size is grater than ps, individuals with less importance are discarded to fit the population size to ps.
        # Sort the p-values of each individual at each outlet model - O(rm)
        if r > ps:
            Indices = np.empty((r, 0), dtype=int)
            for col in range(m):
                aux = np.arange(r).reshape(-1, 1)
                pop_sorted = sorted(zip(p_value[:, col], aux))
                p_value[:, col], ind_ord = zip(*pop_sorted)

                Indices = np.append(Indices, ind_ord, axis=1)

            Ordenado = np.empty(0, dtype=int)
            for filas in range(p_value.shape[0]):
                for val in Indices[filas, :]:
                    if val not in Ordenado:
                        Ordenado = np.append(Ordenado, val)

            pop_aux = population.copy()
            for pos, val in enumerate(Ordenado):
                population[pos] = pop_aux[val]

            B_aux = B
            for pos, val in enumerate(Ordenado):
                B[pos, :] = B_aux[val, :]
            B = B[0:ps, :]
            F_aux = np.zeros((F.shape[0], ps))
            I_aux = np.zeros((ps, I.shape[1]))
            for k in range(ps):
                F_aux[:, k] = F[:, Ordenado[k]]
                I_aux[k, :] = I[Ordenado[k], :]
            F = F_aux
            I = I_aux

            for k in range(ps, len(population)):
                population.remove(population[ps])

        r = len(population)  # Update population size
        if r > ps:
            print('final population:   ', r)
            ValueError

    else:  # There is no individuals
        r = 0
        I = None

    return B, AdjR2, population, RMSE, RMSE_Norm, F, I


def Act(CurrentPop, CurrentQual, CurrentBeta, CurrentF, CurrentI, max_rest, n_rest, population, AdjR2, beta_est, F, I,
        restart, Fit):
    # Update Current Best
    # Update restart state
    if Fit == 'min':
        AdjR2_aux = np.min(AdjR2)
        CurrentQual_aux = np.min(CurrentQual)
    elif Fit == 'mean':
        AdjR2_aux = np.mean(AdjR2)
        CurrentQual_aux = np.mean(CurrentQual)

    if AdjR2_aux > CurrentQual_aux:
        CurrentPop = population.copy()
        CurrentQual = AdjR2
        CurrentBeta = beta_est
        CurrentF = F
        CurrentI = I
    else:
        n_rest = n_rest + 1
        if n_rest > max_rest or AdjR2 < 0 or np.isnan(AdjR2):
            restart = True
            n_rest = 0

    return CurrentPop, CurrentQual, CurrentBeta, CurrentF, CurrentI, n_rest, restart


def evalInd(individual, points, toolbox):
    # Evaluation of an individual for each data point
    func = toolbox.compile(expr=individual)
    fx = [func(*points[pos]) for pos in range(points.shape[0])]
    return fx


def KP_run(x, y, ngen=1000, size=8, ops={'add', 'mul'}, Deph=(2, 6), Deph_max=8, y_max=1, tol=1e-10, \
           verbose=False, Fit='min', RSeed=None, paralell=False, warmStart=False, popStart=None \
           , logbook=None, inicio=None, restart=True):
    # Return:
    # BestPop: The individuals in the best iteration (greatest adjusted R2)
    # BestQual: Greatest Adjusted R2
    # BestBeta: Coefficients of individuals in the best iteration (greatest adjusted R2), for each output variable
    # EcEst: Model for each output variable (greatest adjusted R2)
    # RMSE_Best: Root mean squared error (greatest adjusted R2)
    # RMSE_Best_Norm: RMSE_Best[i]/y_max[i]
    # logbook: save gen, AdjR2 and RMSE of each iteration
    # mod_sim: Best pop as symbolic equation
    if RSeed == None:
        import datetime
        t = datetime.datetime.now().second
        random.seed(t)
    else:
        random.seed(int(RSeed))

    pset = gp.PrimitiveSet("MAIN", int(x.shape[1]))  # number of input variables
    if 'add' in ops:
        pset.addPrimitive(np.add, 2, name="add")
    if 'sub' in ops:
        pset.addPrimitive(np.subtract, 2, name="sub")
    if 'mul' in ops:
        pset.addPrimitive(np.multiply, 2, name="mul")
    if 'div' in ops:
        pset.addPrimitive(protectedDiv, 2)
    if 'neg' in ops:
        pset.addPrimitive(np.negative, 1, name='neg')
    if 'cos' in ops:
        pset.addPrimitive(np.cos, 1)
    if 'sin' in ops:
        pset.addPrimitive(np.sin, 1)
    if 'log' in ops:
        pset.addPrimitive(protectedLog, 1)
    if 'exp' in ops:
        pset.addPrimitive(protectedExp, 1)
    if 'abs' in ops:
        pset.addPrimitive(np.abs, 1, name='abs')
    if 'sqrt' in ops:
        pset.addPrimitive(protectedSqrt, 1)
    if 'square' in ops:
        pset.addPrimitive(np.square, 1)
    if 'powReal' in ops:
        pset.addPrimitive(powerReal, 2)

    pset.addEphemeralConstant("ephem", lambda: random.uniform(0, 1))  # Has to update at each test!!

    for n_var in range(x.shape[1]):
        arg_name = 'ARG{}'.format(n_var)
        new_name = 'X{}'.format(n_var)
        pset.renameArguments(**{arg_name: new_name})

    def evalSymbReg(individual):
        return 1,

    creator.create("FitnessMin", base.Fitness, weights=tuple(len(x) * [1.0]))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    Deph_Min = Deph[0]
    Deph_Max = Deph[1]
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=Deph_Min, max_=Deph_Max)  # create poblation
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def twoMut(individual, probMut, expr, pset):
        if random.random() < probMut:
            return gp.mutEphemeral(individual, mode="all")
        else:
            return gp.mutUniform(individual, expr=expr, pset=pset)

    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selRandom)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=Deph_Min, max_=Deph_Max)
    toolbox.register("mutate", twoMut, expr=toolbox.expr_mut, pset=pset, probMut=.1)
    toolbox.register("calculate", evalInd, points=x, toolbox=toolbox)

    if paralell:
        pool = mp.Pool(4)
        # pool=mp.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    # This is done to avoid  bloat
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=Deph_max))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=Deph_max))

    if warmStart:
        pop = popStart
    else:
        pop = toolbox.population(n=size)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'AdjR2', 'RMSE', 'tiempos']
        inicio = 0

    cxpb = 1
    mutpb = 1
    alfa = .05
    theta = 1e-4
    generations = int(ngen)
    if restart:
        max_rest = int(.25 * generations)
    else:
        max_rest = np.inf

    n_rest = 0
    restart = False
    CurrentPop = pop.copy()
    BestPop = CurrentPop.copy()
    CurrentBeta = []
    CurrentQual = -np.inf
    BestQual = CurrentQual
    it_best = np.inf
    RMSE = np.empty((0, y.shape[1]))
    RMSE_N = np.empty((0, y.shape[1]))

    for i in range(generations):
        if restart == True:
            del CurrentPop
            CurrentPop = toolbox.population(n=size)
            restart = False
        elif len(CurrentPop) < size:
            ext = size - len(CurrentPop)
            if i > max_rest:
                if len(BestPop) > ext:
                    extension = BestPop[0:ext]
                else:
                    extension1 = BestPop[0:len(BestPop)]
                    extension2 = toolbox.population(n=ext - len(BestPop))
                    extension = Do(extension1, extension2)
            else:
                extension = toolbox.population(n=ext)
            CurrentPop = Do(CurrentPop, extension)
        offspring = Plan(CurrentPop, toolbox, cxpb, mutpb)
        del pop
        pop = Do(CurrentPop, offspring)
        beta_est, AdjR2, pop, RMSE_it, RMSE_Norm, F_it, I_it = Check(pop, toolbox, y, size, alfa, theta, y_max)
        try:
            RMSE = np.append(RMSE, RMSE_it.reshape(1, -1), axis=0)
            RMSE_N = np.append(RMSE_N, np.asarray(RMSE_Norm).reshape(1, -1), axis=0)
        except:
            pass
        if i == 0:
            CurrentF = F_it
            CurrentI = I_it
        if Fit == 'min':
            CurrentPop, CurrentQual, CurrentBeta, CurrentF, CurrentI, n_rest, restart = Act(CurrentPop, CurrentQual,
                                                                                            CurrentBeta, CurrentF,
                                                                                            CurrentI, max_rest, n_rest,
                                                                                            pop, np.min(AdjR2),
                                                                                            beta_est, F_it, I_it,
                                                                                            restart, Fit)
        elif Fit == 'mean':
            CurrentPop, CurrentQual, CurrentBeta, CurrentF, CurrentI, n_rest, restart = Act(CurrentPop, CurrentQual,
                                                                                            CurrentBeta, CurrentF,
                                                                                            CurrentI, max_rest, n_rest,
                                                                                            pop, np.mean(AdjR2),
                                                                                            beta_est, F_it, I_it,
                                                                                            restart, Fit)

        logbook.record(gen=i + 1 + inicio, AdjR2=np.asarray(AdjR2), RMSE=RMSE_it)
        if Fit == 'min':
            BestQual_aux = np.min(BestQual)
            CurrentQual_aux = np.min(CurrentQual)
        elif Fit == 'mean':
            BestQual_aux = np.mean(BestQual)
            CurrentQual_aux = np.mean(CurrentQual)
        if CurrentQual_aux > BestQual_aux:
            BestQual = CurrentQual
            BestPop = CurrentPop.copy()
            # BestBeta= CurrentBeta
            # it_best=i
            F = CurrentF
            I = CurrentI
            if tol is not None:
                if (1 - BestQual) < tol:
                    break

        if verbose:
            print(logbook.stream)
    if paralell:
        pool.close()

    BestBeta, p_vals, _, AdjR2, RMSE_Best, RMSE_Best_Norm = PLSR(F, y, len(y), len(BestPop), y_max, I=I)
    mod_sim = np.arange(0)
    for i in range(len(BestPop)):
        final = tree2symb(BestPop[i].__str__(), x.shape[1])
        mod_sim = np.append(mod_sim, final)

    EcEst = []
    try:
        for pos in range(BestBeta.shape[1]):
            Ec_aux = np.dot(BestBeta[:, pos].flat, mod_sim)
            EcEst.append(Ec_aux)
    except:
        try:
            EcEst = np.dot(BestBeta.flat, mod_sim)
        except:
            pass

    return BestPop, BestQual, BestBeta, EcEst, RMSE_Best, RMSE_Best_Norm, logbook, mod_sim, I, toolbox
