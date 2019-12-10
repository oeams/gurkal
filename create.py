import numpy as np


def createQUBO(processing_times, nP,  difference):

    Q = {}
    num_entries = 0
    max_value = 0
    nJ = len(processing_times)
    q_size = nP * nJ + (nP - 1) * len(difference)
    for i in range(q_size):
        for j in range(i, q_size):
            Q[(i, j)] = 0

    const1 = max(processing_times) + 1
    const2 = 1

    # Constraint1\1: Every Job at exactly one Processor
    for i in range(nJ):
        for j in range(nP):
            r = i * nP + j
            Q[(r, r)] += -1 * const1
            num_entries+=1
            for k in range(j+1, nP):
                c = i * nP + k
                Q[(r, c)] += 2 * const1
                num_entries+=1

    # Constraint1: No processor needs longer than processor 1
    # Constraint1/2: One processor may not have several difference times y_n,a
    # difference time x difference time
    for i in range(len(difference)):
        for j in range(nP-1):               # variable y_n,1 is not defined
            r = nJ * nP + i * (nP-1) + j
            Q[(r, r)] += difference[i] * difference[i] * const1
            num_entries+=1
            for k in range(i+1, len(difference)):
                c = nJ * nP + k * (nP-1) + j
                Q[(r, c)] += 2 * difference[i] * difference[k] * const1
                num_entries+=1

    # Constraint1/3: The time difference is exactly n
    # difference time x job assignment
    for i in range(nJ):
        for j in range(1, nP):              # variable y_n,1 is not defined
            r = nP * i + j
            for k in range(len(difference)):
                c = nP * nJ + (j-1) + k * (nP-1)
                Q[(r, c)] += 2 * difference[k] * processing_times[i] * const1
                num_entries+=1
        r = nP * i
        for j in range(len(difference)):
            for k in range(nP - 1):           # variable y_n,1 is not defined
                c = nP * nJ + j * (nP-1) + k
                Q[(r, c)] += -2 * difference[j] * processing_times[i] * const1
                num_entries += 1

    # Constraint1/4: Not too many jobs on the same machine
    # job assignment x job assignment
    for i in range(nJ):
        for j in range(nP):
            r = i * nP + j
            if j == 0:                           # processor 1 must be treated different
                for k in range(i, nJ):
                    for l in range(1, nP):
                        c = k * nP + l
                        # -2*a*b
                        if Q[(r, c)] == 0:
                            num_entries += 1
                        Q[(r, c)] += -2 * processing_times[i] * processing_times[k] * const1
                for k in range(i + 1, nJ):
                    c = k * nP
                    # b*b
                    if Q[(r, c)] == 0:
                        num_entries += 1
                    Q[(r, c)] = 2 * processing_times[i] * processing_times[k] * (nP - 1) * const1
                # b*b
                Q[(r, r)] += processing_times[i] * processing_times[i] * (nP - 1)* const1
            else:
                for k in range(i + 1, nJ):
                    c = k * nP
                    # -2 * a * b
                    if Q[(r, c)] == 0:
                        num_entries += 1
                    Q[(r, c)] += -2 * processing_times[i] * processing_times[k] * const1
                for k in range(i + 1, nJ):
                    c = k * nP + j
                    # a * a
                    if Q[(r, c)] == 0:
                        num_entries += 1
                    Q[(r, c)] = 2 * processing_times[i] * processing_times[k] * const1
                # a * a
                Q[(r, r)] += processing_times[i] * processing_times[i] * const1

    # Constraint 2: Processor 1 as fast as possible
    for i in range(nJ):
        r = i * nP
        Q[r, r] += processing_times[i] * const2

    QM = np.zeros((q_size, q_size), dtype=object)
    for key in Q:
        r = key[0]
        c = key[1]
        v = Q[key]
        if abs(v) > max_value:
            max_value = abs(v)
        QM[r, c] = v
    np.savetxt("QUBO.csv", QM, delimiter=";", fmt="%s")

    return Q, num_entries, max_value