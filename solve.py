# -*- coding: iso-8859-1 -*-
import matplotlib.pyplot as plt
from matplotlib import rc
from dwave.embedding import unembed_sampleset, chain_break_frequency
import dimod

def get_solver_structure(n, m):
    edgelist = []

    for i in range(n - 1):  # für jede Zeile (letzte Spalte muss gesondert behandelt werden)
        for j in range(m - 1):  # für jede Spalte (letzte Spalte muss gesondert behandelt werden)
            for k in range(4):  # jedes Qubit auf der linken Seite
                q1 = (i * m + j) * 8 + k
                for l in range(4):  # je mit einem Qubit rechts verbinden
                    q2 = (i * m + j) * 8 + 4 + l
                    edgelist.append(tuple([q1, q2]))
                q2 = q1 + 8 * m
                edgelist.append(tuple([q1, q2]))  # zur darunterliegenden Zelle verbinden

            for k in range(4):  # für jedes Qubit auf der rechten seite
                q1 = (i * m + j) * 8 + 4 + k
                q2 = q1 + 8
                edgelist.append(tuple([q1, q2]))  # mit einem qubit in der rechten zelle verbinden

        # nur für die letzte Spalte m - 1
        for k in range(4):  # jedes Qubit auf der linken Seite
            q1 = (i * m + m - 1) * 8 + k
            for l in range(4):
                q2 = (i * m + m - 1) * 8 + 4 + l
                edgelist.append(tuple([q1, q2]))  # zu jedem Qubit in der Zelle verbinden
            q2 = q1 + 8 * m
            edgelist.append(tuple([q1, q2]))  # zur darunterliegenden Zelle verbinden

    # nur für die letzte Zeile n - 1
    for j in range(m - 1):  # für jede Spalte (letzte Spalte muss gesondert behandelt werden)
        for k in range(4):  # jedes Qubit auf der linken Seite
            q1 = ((n - 1) * m + j) * 8 + k
            for l in range(4):  # je mit einem Qubit rechts verbinden
                q2 = ((n - 1) * m + j) * 8 + 4 + l
                edgelist.append(tuple([q1, q2]))

        for k in range(4):  # für jedes Qubit auf der rechten seite
            q1 = ((n - 1) * m + j) * 8 + 4 + k
            q2 = q1 + 8
            edgelist.append(tuple([q1, q2]))  # mit einem qubit in der rechten zelle verbinden

    for k in range(4):  # jedes Qubit auf der linken Seite
        q1 = ((n - 1) * m + m - 1) * 8 + k
        for l in range(4):
            q2 = ((n - 1) * m + m - 1) * 8 + 4 + l
            edgelist.append(tuple([q1, q2]))  # zu jedem Qubit in der Zelle verbinden

    num_vars = m * n * 8
    nodelist = list(range(num_vars))

    return nodelist, edgelist


def solution_distribution(Q, nJ, nP, diff, c1, c2):

    size = nP * nJ + (nP - 1) * len(diff)
    sample = [0] * size
    energies = []
    for i in range(2**size):
        current = i
        for j in reversed(range(size)):
            if(int(bin(current),2) & int(bin(1),2) > 0):
                sample[j] = 1
            else:
                sample[j] = 0
            current = current >> 1

        energy = 0
        for r in range(size):
            if sample[r] == 1:
                energy += Q[(r, r)]
                for c in range(r+1, size):
                    if sample[c] == 1:
                        energy += Q[(r, c)]

        energies.append(energy)

    # plt.plot(energies)
    # plt.title('A = '+ str(c1) + ', B = ' + str(c2))
    # plt.ylabel('Value of QUBO formula')
    # plt.xlabel('Solutions')

    # fig.savefig('knapsack')
    # filename = "landscape" + str(c1) + "-" + str(c2) + ".tikz"
    # tikz_save(filename,
    #           figureheight='\\figureheight',
    #           figurewidth='\\figurewidth')
    # plt.show()

    energies.sort()
    for i in range(len(energies)):
        if energies[0] != energies[i]:
            next = energies[i]
            break

    return energies[0], next, energies[len(energies)-1]


def solveQUBO(Q, jobs, nP, sampler):


    schedules = []
    for i in range(nP):
        schedules.append([])

    answer = sampler.sample_qubo(Q, postprocess='optimization')
    solution = {}

    #print "#############################"
    #print "Answer : ", answer.first[0]
    #print "Energy: ", answer.first[1]

    solution['sample'] = answer.first[0]
    solution['energy'] = answer.first[1]
    solution['chain_break_fraction'] = answer.first[3]
    solution['valid'] = True                        # Default
    processing_times = [0] * nP

    for i in range(len(jobs)):
        assigned = False
        for j in range(nP):
            curr = i * nP + j
            if answer.first[0][curr] == 1:
                # "Job ", i, " assigned to processor ", j
                schedules[j].append(i)
                processing_times[j] += jobs[i]
                if not assigned:
                    assigned = True
                else:
                    solution['valid'] = False
        if not assigned:
            solution['valid'] = False

    # print "Processors and their assigned jobs: ", schedules
    solution['max_time'] = max(processing_times)
    return solution


def evaluate(solutions):

    sol_energy = 0
    sol_time = 0
    sol_range = 0
    sol_corr = 0
    sol_erange = 0
    sol_break = 0

    # opt = nJ * -(const1) + bearbeitungszeit von prozessor1
    opt = 4 * (-4) + 5
    for i in range(len(solutions['energy'])):
        if solutions['valid'][i]:
            sol_corr += 1
            if solutions['energy'][i] == opt:
                sol_energy += 1
            if solutions['max_time'][i] == 7:
                sol_time += 1
            if solutions['max_time'][i] <= 9:
                sol_range += 1
        if solutions['energy'][i] <= 0.5 * opt:
            sol_erange += 1
        if solutions['chain_break_fraction'][i] == 0.0:
            sol_break += 1

    print ("Valid Solutions: ", sol_corr)
    print ("Correct Energies: ", sol_energy)
    print ("Assignment as least as good as BKS: ", sol_time)
    print ("Timespan in range of 25% of BKS: ", sol_range)
    print ("Energy at least half of optimum: ", sol_erange)
    print ("No broken chains: ", sol_break)