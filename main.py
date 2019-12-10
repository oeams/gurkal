# -*- coding: iso-8859-1 -*-
import create
import solve
import neal
import dimod
from dwave.system.composites import FixedEmbeddingComposite, VirtualGraphComposite
import dwave_networkx as dnx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy
import operator
import minorminer
import csv


from dwave.system.samplers import DWaveSampler
from matplotlib import rc
from tikzplotlib import save as tikz_save



def check_if_proper():

    nP = 2
    instances = [[[1, 2, 3, 4], [0]], [[2, 3, 4, 5], [0]], [[2, 4, 6, 8], [0]], [[1, 2, 3, 6], [0]], [[2, 3, 4, 7], [2]],
                 [[10, 11, 12, 15], [2]],
                 [[10, 20, 30, 40], [0]]]

    for i in instances:
        jobs = i[0]
        diff = i[1]
        table = []
        print
        print ("#########################")
        print ("Instance ", i)
        print ("c1       | c2        | Optimum   | Range     | Gap       |Relation")
        for c1_mult in range(1,11):
            c1 = max(jobs)*c1_mult + 1
            Q = create.createQUBO(jobs, nP, diff, c1_mult, 1)
            best, second, worst = solve.solution_distribution(Q, len(jobs),nP, diff, c1, 1)
            print (c1, "     | 1         | ", best, "    | ", worst-best, "  | ", second - best, "       | ", float(second-best)/float(worst-best))

            embedding = {0: [1760, 1632, 1764], 1: [1628, 1636, 1634], 2: [1638, 1630, 1627],
                         3: [1752, 1624, 1759, 1767],
                         4: [1763, 1635, 1765], 5: [1761, 1633], 6: [1754, 1626, 1758, 1766], 7: [1631, 1639, 1625],
                         8: [1637, 1629]}

            base_sampler = neal.SimulatedAnnealingSampler()
            G = dnx.chimera_graph(16,16,4)
            nodelist = G.nodes()
            edgelist = G.edges()
            source_edgelist = list(Q)
            sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
            new_sampler = FixedEmbeddingComposite(sampler, embedding)
            solution = {'energy': [], "valid": [], 'max_time': []}
            for k in range(100):
                s = solve.solveQUBO(Q, jobs, nP, new_sampler)
                solution['energy'].append(s['energy'])
                solution['valid'].append(s['valid'])
                solution['max_time'].append(s['max_time'])
            table.append(solution)


        print ("Solutions ")
        print ("c1       | c2    | Energies                                          | Valid                                                | Max_Time       ")
        for c1_mult in range(1, 11):
            print (max(jobs)*c1_mult + 1, "     | 1         | ", table[c1_mult-1]['energy'], "    | ", table[c1_mult-1]['valid'], "  | ", table[c1_mult-1]['max_time'])


def compare_coefficients():
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True

    jobs = [1,2,3,4]
    nP = 2
    diff = [1]

    for c2 in range(4, 5):
        ran = []
        gap = []
        rel = []
        for c1 in range(1, 6):
            Q = create.createQUBO(jobs, nP, diff, c1, c2)
            best, second, worst = solve.solution_distribution(Q, len(jobs), nP, diff, max(jobs)*c1 + 1, 1)
            ran.append(worst - best)
            gap.append(second - best)
            rel.append(float(second - best)/float(worst - best))

        x_ran = list(np.arange(0.8, 5.8, 1))
        x_gap = list(np.arange(1.2, 6.2, 1))
        x_rel = list(range(1, 6, 1))
        f, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.bar(x_ran, ran, width=0.4)
        ax1.bar(x_gap, gap, color='g', width=0.4)
        ax1.set_yscale('log')
        ax2.plot(x_rel, rel, 'r')
        ax2.tick_params(axis='y', colors='r')
        plt.title(str(c1) + ", " + str(c2))
        ax1.legend(['Spannweite', u'Lucke'])
        ax1.set_ylabel("Energiespektrum")
        ax1.set_xlabel("Koeffizent A")
        ax2.set_ylabel(u"Verhaltnis")

        tikz_save(str(c1) + ", " + str(c2),
                  figureheight='\\figureheight',
                  figurewidth='\\figurewidth')


def get_threshold_qubo(percentage, Q, max_entry):

    Q_threshold = {}
    deleted_entrys = 0
    max = 0
    # use only values which are big enough
    for key in Q:
        if abs(Q[key]) >= percentage * max_entry:
            Q_threshold[key] = Q[key]
        else:
            Q_threshold[key] = 0.0
            if Q[key] != 0:
                if abs(Q[key]) > max:
                    max = abs(Q[key])
                deleted_entrys += 1
    return Q_threshold, deleted_entrys, max


def get_given_percentage_qubo(percentage, Q, sorted_q, num_dist):

    Q_deleted = copy.deepcopy(Q)
    d = int(num_dist * percentage)
    del sorted_q[d: len(sorted_q)]              # contains now all values that have to be deleted
    print(len(sorted_q))
    for key in sorted_q:
        Q_deleted[key] = 0.0

    if len(sorted_q) == 0:
        return Q_deleted, 0, 0
    max = Q[sorted_q[len(sorted_q)-1]]
    return Q_deleted, max, len(sorted_q)


def get_random_qubo(Q, percentage, num_dist):

    to_delete = int(num_dist * percentage)
    deleted = to_delete
    Q_random = copy.deepcopy(Q)
    max = 0
    for key in Q.keys():  # delete random entries
        if to_delete <= 0:
            break
        if abs(Q[key]) > max:
            max = abs(Q[key])
        Q_random[key] = 0.0
        to_delete -= 1

    return Q_random, max, deleted


def example_instance():

    nP = 2
    jobs = [1,2,3,4]
    diff = []

    embedding = {0: [1760, 1632, 1764], 1: [1628, 1636, 1634], 2: [1638, 1630, 1627], 3: [1752, 1624, 1759, 1767],
    4: [1763, 1635, 1765], 5: [1761, 1633], 6: [1754, 1626, 1758, 1766], 7: [1631, 1639, 1625],
    8: [1637, 1629]}

    # base_sampler = neal.SimulatedAnnealingSampler()
    base_sampler = DWaveSampler()
    #base_sampler.properties['extended_j_range'] = [-10.0, 10.0]
    G = dnx.chimera_graph(16,16,4)
    nodelist = G.nodes()
    edgelist = G.edges()
    sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    new_sampler = FixedEmbeddingComposite(sampler, embedding)
    # print new_sampler.properties
    # print new_sampler.parameters
    #new_sampler = VirtualGraphComposite(sampler, embedding, chain_strength=-6.0)

    Q = create.createQUBO(jobs, nP, diff)

    solutions = {'energy': [], "valid": [], 'max_time': [], 'chain_break_fraction':[]}
    for k in range(10):
        s = solve.solveQUBO(Q, jobs, nP, new_sampler, 0)
        solutions['energy'].append(s['energy'])
        solutions['valid'].append(s['valid'])
        solutions['max_time'].append(s['max_time'])
        solutions['chain_break_fraction'].append(s['chain_break_fraction'])
    solve.evaluate(solutions)
    print(solutions)
    print("Durchschnittliche Anteil gebrochender Ketten: ", sum(solutions['chain_break_fraction'])/100)


def example_instance_big():

    nP = 3
    jobs = [5,7,3,5,9,6,3,1,3,1]
    diff = [1,2,2,5]

    base_sampler = DWaveSampler()
    nodelist = base_sampler.nodelist
    edgelist = base_sampler.edgelist

    Q, num_entries, max_value = create.createQUBO(jobs, nP, diff)

    embedding = {0: [622, 614, 606, 598, 726, 594, 722, 978, 630, 638, 850, 593, 465, 337, 602, 1106, 590],
                 1: [1090, 834, 1110, 578, 1102, 1094, 962, 1118, 706, 1134, 1126, 1142, 1150, 710, 836, 718, 1112, 702,
                     583], 2: [894, 830, 838, 846, 854, 886, 878, 870, 862, 851, 723, 595, 979],
                 3: [362, 106, 490, 234, 618, 749, 1002, 746, 874, 110, 118, 126, 1130],
                 4: [342, 334, 326, 318, 382, 374, 569, 953, 366, 358, 350, 313, 441, 697, 825, 1081],
                 5: [315, 325, 341, 333, 317, 365, 357, 349, 443, 571, 699, 381, 827, 955, 373],
                 6: [123, 635, 379, 1019, 763, 251, 891, 1147, 507, 1275, 1151],
                 7: [764, 756, 716, 748, 740, 732, 724, 708, 700],
                 8: [591, 599, 607, 345, 985, 601, 729, 857, 615, 473, 639, 1113, 623, 631],
                 9: [511, 455, 704, 448, 471, 463, 960, 576, 479, 487, 496, 495, 503, 320, 624, 832, 1088],
                 10: [709, 1141, 971, 1099, 1109, 843, 717, 844, 715, 701, 1101, 1117, 1125, 1133, 725, 733, 1149, 741],
                 11: [761, 1017, 633, 889, 505, 377, 249, 1145, 121, 1273, 127],
                 12: [893, 829, 885, 877, 837, 845, 853, 861, 869, 880, 752, 1008],
                 13: [1007, 999, 991, 959, 967, 975, 983, 1015, 1023, 977, 849, 721],
                 14: [1127, 961, 1119, 833, 1095, 1103, 1111, 1089, 705, 839, 1135, 831, 847, 1120, 1248, 1254, 1262,
                      1270, 1278, 855, 863],
                 15: [608, 589, 605, 1098, 597, 613, 480, 352, 586, 842, 714, 970, 224, 96, 101, 109, 117, 125],
                 16: [611, 355, 483, 227, 253, 245, 237, 229, 739, 867, 871, 995, 1123],
                 17: [491, 1003, 107, 363, 619, 747, 246, 238, 235, 875, 254, 1131],
                 18: [482, 758, 766, 750, 742, 994, 738, 610, 866, 1122, 354],
                 19: [988, 835, 1004, 996, 980, 707, 964, 972, 992, 864, 868, 876, 884, 892, 956, 963],
                 20: [632, 1144, 888, 504, 1016, 248, 376, 760, 1272, 120],
                 21: [982, 958, 966, 974, 950, 946, 818, 690, 990, 998, 1006, 1014, 1022, 692],
                 22: [442, 452, 468, 460, 1082, 444, 570, 698, 826, 954, 476, 484, 458, 492, 500, 508, 314],
                 23: [728, 216, 344, 856, 472, 236, 228, 220, 600, 244, 984, 734, 252],
                 24: [510, 502, 494, 865, 486, 481, 478, 353, 609, 993, 737, 1121, 470, 462],
                 25: [84, 72, 456, 200, 328, 92, 76, 100, 98, 226, 968, 108, 116, 584, 461, 469, 712, 124, 840, 1096],
                 26: [703, 751, 719, 727, 767, 759, 743, 735, 711, 695, 736],
                 27: [475, 493, 485, 477, 987, 603, 731, 859, 501, 509, 347, 1115],
                 28: [1011, 629, 371, 621, 627, 637, 755, 243, 499, 883, 1139, 115, 1267],
                 29: [1005, 997, 989, 1013, 965, 976, 973, 981, 957, 1021, 848, 720, 592, 596, 588],
                 30: [329, 343, 335, 585, 383, 359, 351, 457, 367, 375, 713, 841, 969, 338],
                 31: [340, 332, 324, 364, 348, 356, 316, 312, 440, 568, 696, 824, 952, 372, 380, 1080],
                 32: [1116, 1114, 1124, 1084, 1092, 1100, 1108, 986, 730, 1132, 858, 1140, 1148, 1097, 1083],
                 33: [1018, 762, 250, 634, 122, 506, 378, 890, 1020, 1146, 1274],
                 34: [757, 626, 1010, 754, 882, 498, 1138, 370, 242, 765, 114, 1266],
                 35: [1001, 617, 636, 628, 620, 873, 489, 745, 361, 233, 105, 1129, 612],
                 36: [1009, 625, 881, 497, 369, 241, 895, 753, 887, 1137, 113, 1265],
                 37: [255, 247, 872, 239, 488, 616, 232, 360, 744, 1000, 1128, 240, 112, 231]}
    sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    new_sampler = FixedEmbeddingComposite(sampler, embedding)

    solutions = {'energy': [], "valid": [], 'max_time': [], 'chain_break_fraction':[]}
    for k in range(10):
        s = solve.solveQUBO(Q, jobs, nP, new_sampler)
        solutions['energy'].append(s['energy'])
        solutions['valid'].append(s['valid'])
        solutions['max_time'].append(s['max_time'])
        solutions['chain_break_fraction'].append(s['chain_break_fraction'])
    solve.evaluate(solutions)
    print(solutions)
    print("Durchschnittliche Anteil gebrochender Ketten: ", sum(solutions['chain_break_fraction'])/100)


def directly_embedded():
    nP = 2
    jobs = [1, 2, 3, 4]
    diff = [1]

    cs = -20.0
    # für constraint 1 = 5
    quadratic = {(1752, 1758): -40.0, (1630, 1626): 80.0, (1638, 1632): 20.0, (1631, 1626): -150.0, (1760, 1632): cs,
         (1632, 1637): -10.0, (1633, 1639): 120.0, (1635, 1639): -120.0, (1634, 1637): 10.0, (1752, 1759): cs,
         (1765, 1763): cs, (1630, 1627): cs, (1638, 1635): 60.0, (1633, 1637): 30.0, (1763, 1766): 120.0,
         (1638, 1633): -60.0, (1624, 1628): 20.0, (1760, 1766): 40.0, (1763, 1764): 15.0, (1633, 1636): 30.0,
         (1631, 1625): cs, (1635, 1637): -30.0, (1635, 1636): -30.0, (1630, 1624): -30.0, (1758, 1766): cs,
         (1625, 1629): 40.0, (1627, 1629): -20.0, (1631, 1639): cs, (1752, 1624): cs, (1754, 1626): cs,
         (1761, 1767): 60.0, (1632, 1636): 0.0, (1760, 1767): -20.0, (1626, 1628): -40.0, (1638, 1634): -10.0,
         (1631, 1624): 80.0, (1761, 1764): -30.0, (1632, 1639): -40.0, (1765, 1761): -80.0, (1630, 1625): -40.0,
         (1629, 1637): cs, (1625, 1628): 20.0, (1626, 1629): -40.0, (1628, 1636): cs, (1624, 1629): 20.0,
         (1635, 1763): cs, (1627, 1628): -10.0, (1634, 1639): 20.0, (1759, 1767): cs, (1638, 1630): cs,
         (1754, 1759): -40.0, (1631, 1627): -40.0, (1754, 1758): cs, (1634, 1636): cs, (1765, 1760): 15.0,
         (1760, 1764): cs, (1763, 1767): -60.0, (1761, 1766): -120.0, (1761, 1633): cs}

    # {(1752, 1758): -40.0, (1630, 1626): 80.0, (1638, 1632): 20.0, (1631, 1626): -150.0, (1760, 1632): -160.0,
    #  (1632, 1637): -10.0, (1633, 1639): 120.0, (1635, 1639): -120.0, (1634, 1637): 10.0, (1752, 1759): -160.0,
    #  (1765, 1763): -160.0, (1630, 1627): -160.0, (1638, 1635): 60.0, (1633, 1637): 30.0, (1763, 1766): 120.0,
    #  (1638, 1633): -60.0, (1624, 1628): 20.0, (1760, 1766): 40.0, (1763, 1764): 15.0, (1633, 1636): 30.0,
    #  (1631, 1625): -160.0, (1635, 1637): -30.0, (1635, 1636): -30.0, (1630, 1624): -30.0, (1758, 1766): -160.0,
    #  (1625, 1629): 40.0, (1627, 1629): -20.0, (1631, 1639): -160.0, (1752, 1624): -160.0, (1754, 1626): -160.0,
    #  (1761, 1767): 60.0, (1632, 1636): 0.0, (1760, 1767): -20.0, (1626, 1628): -40.0, (1638, 1634): -10.0,
    #  (1631, 1624): 80.0, (1761, 1764): -30.0, (1632, 1639): -40.0, (1765, 1761): -80.0, (1630, 1625): -40.0,
    #  (1629, 1637): -160.0, (1625, 1628): 20.0, (1626, 1629): -40.0, (1628, 1636): -160.0, (1624, 1629): 20.0,
    #  (1635, 1763): -160.0, (1627, 1628): -10.0, (1634, 1639): 20.0, (1759, 1767): -160.0, (1638, 1630): -160.0,
    #  (1754, 1759): -40.0, (1631, 1627): -40.0, (1754, 1758): -160.0, (1634, 1636): -160.0, (1765, 1760): 15.0,
    #  (1760, 1764): -160.0, (1763, 1767): -60.0, (1761, 1766): -120.0, (1761, 1633): -160.0}

    linear = {1637: 4.5, 1752: 26.5,  1626: 73.0, 1766: -22.0, 1630: -3.666666666666668, 1631: 65.66666666666667,
              1632: 15.666666666666668, 1633: -50.5, 1763: -37.5, 1624: -40.5, 1625: -1.3333333333333321,
              1754: 20.0, 1627: 34.33333333333333, 1628: 8.666666666666666, 1629: 4.5, 1758: 20.0, 1759: 26.5,
              1760: -14.833333333333332, 1761: 94.5, 1634: -6.333333333333333, 1635: 58.0, 1636: 5.666666666666666,
              1765: 30.5, 1638: -5.666666666666664, 1767: 14.5, 1639: 18.66666666666667, 1764: 8.166666666666666}
    # cs = 40
    # {1637: 82.5, 1752: 182.5, 1626: 151.0, 1766: 56.0, 1630: 152.33333333333331, 1631: 221.66666666666669,
    #  1632: 93.66666666666667, 1633: 27.5, 1763: 118.5, 1624: 37.5, 1625: 76.66666666666667, 1754: 176.0,
    #  1627: 112.33333333333333, 1628: 86.66666666666667, 1629: 82.5, 1758: 176.0, 1759: 182.5, 1760: 141.16666666666669,
    #  1761: 172.5, 1634: 71.66666666666667, 1635: 136.0, 1636: 161.66666666666669, 1765: 108.5, 1638: 72.33333333333333,
    #  1767: 92.5, 1639: 96.66666666666667, 1764: 86.16666666666667}

    # cs = 4
    # {1637: 10.5, 1752: 38.5, 1626: 79.0, 1766: -16.0, 1630: 8.333333333333332, 1631: 77.66666666666667,
    #  1632: 21.666666666666668, 1633: -44.5, 1763: -25.5, 1624: -34.5, 1625: 4.666666666666668, 1754: 32.0,
    #  1627: 40.333333333333336, 1628: 14.666666666666666, 1629: 10.5, 1758: 32.0, 1759: 38.5, 1760: -2.833333333333332,
    #  1761: 100.5, 1634: -0.33333333333333215, 1635: 64.0, 1636: 17.666666666666664, 1765: 36.5,
    #  1638: 0.3333333333333357, 1767: 20.5, 1639: 24.66666666666667, 1764: 14.166666666666666}

    # cs = 3
    # {1637: 8.5, 1752: 34.5, 1626: 77.0, 1766: -18.0, 1630: 4.333333333333332, 1631: 73.66666666666667,
    #  1632: 19.666666666666668, 1633: -46.5, 1763: -29.5, 1624: -36.5, 1625: 2.666666666666668, 1754: 28.0,
    #  1627: 38.33333333333333, 1628: 12.666666666666666, 1629: 8.5, 1758: 28.0, 1759: 34.5, 1760: -6.833333333333332,
    #  1761: 98.5, 1634: -2.333333333333333, 1635: 62.0, 1636: 13.666666666666666, 1765: 34.5, 1638: -1.6666666666666643,
    #  1767: 18.5, 1639: 22.66666666666667, 1764: 12.166666666666666}

    schedules = []
    for i in range(nP):
        schedules.append([])

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)  # QUBO
    answer = neal.SimulatedAnnealingSampler().sample(bqm)

    solution = {}
    # solution['energy'] = answer.first[1]
    # solution['chain_break_fraction'] = answer.first[3]
    solution['valid'] = True  # Default
    processing_times = [0] * nP
    broken = 0

    sample = [0] * 9
    sample[0] = 1 if (sum([answer.first[0][1760], answer.first[0][1632], answer.first[0][1764]]) > 1) else 0
    sample[1] = 1 if (sum([answer.first[0][1628], answer.first[0][1636], answer.first[0][1634]]) > 1) else 0
    sample[2] = 1 if (sum([answer.first[0][1638], answer.first[0][1630], answer.first[0][1627]]) > 1) else 0
    sample[3] = 1 if (sum([answer.first[0][1752], answer.first[0][1624], answer.first[0][1759], answer.first[0][1767]]) > 2) else 0
    sample[4] = 1 if (sum([answer.first[0][1763], answer.first[0][1635], answer.first[0][1765]]) > 1) else 0
    sample[5] = 1 if (sum([answer.first[0][1761], answer.first[0][1633]]) > 1) else 0
    sample[6] = 1 if (sum([answer.first[0][1754], answer.first[0][1626], answer.first[0][1758], answer.first[0][1766]]) > 2) else 0
    sample[7] = 1 if (sum([answer.first[0][1631], answer.first[0][1639], answer.first[0][1625]]) > 1) else 0
    sample[8] = 1 if (sum([answer.first[0][1637], answer.first[0][1629]]) > 1) else 0
    solution['sample'] = sample


    embedding = {0: [1760, 1632, 1764], 1: [1628, 1636, 1634], 2: [1638, 1630, 1627], 3: [1752, 1624, 1759, 1767],
                 4: [1763, 1635, 1765], 5: [1761, 1633], 6: [1754, 1626, 1758, 1766], 7: [1631, 1639, 1625],
                 8: [1637, 1629]}

    for value in embedding.itervalues():
        for i in range(1, len(value)):
            if answer.first[0][value[0]] != answer.first[0][value[i]]:
                broken += 1
                break
    solution['chain_break_frequancy'] = broken/9.0

    for i in range(len(jobs)):
        assigned = False
        for j in range(nP):
            curr = i * nP + j
            if sample[curr] == 1:
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
    print (solution)
    return solution


def example_instance_delete_random(percentage):
    nP = 3
    jobs = [5, 7, 3, 5, 9, 6, 3, 1, 3, 1]
    diff = [1, 2, 2, 5]


    base_sampler = DWaveSampler()
    nodelist = base_sampler.nodelist
    edgelist = base_sampler.edgelist

    Q, num_entries, max_value= create.createQUBO(jobs, nP, diff)

    embedding = {0: [622, 614, 606, 598, 726, 594, 722, 978, 630, 638, 850, 593, 465, 337, 602, 1106, 590],
                 1: [1090, 834, 1110, 578, 1102, 1094, 962, 1118, 706, 1134, 1126, 1142, 1150, 710, 836, 718, 1112, 702,
                     583], 2: [894, 830, 838, 846, 854, 886, 878, 870, 862, 851, 723, 595, 979],
                 3: [362, 106, 490, 234, 618, 749, 1002, 746, 874, 110, 118, 126, 1130],
                 4: [342, 334, 326, 318, 382, 374, 569, 953, 366, 358, 350, 313, 441, 697, 825, 1081],
                 5: [315, 325, 341, 333, 317, 365, 357, 349, 443, 571, 699, 381, 827, 955, 373],
                 6: [123, 635, 379, 1019, 763, 251, 891, 1147, 507, 1275, 1151],
                 7: [764, 756, 716, 748, 740, 732, 724, 708, 700],
                 8: [591, 599, 607, 345, 985, 601, 729, 857, 615, 473, 639, 1113, 623, 631],
                 9: [511, 455, 704, 448, 471, 463, 960, 576, 479, 487, 496, 495, 503, 320, 624, 832, 1088],
                 10: [709, 1141, 971, 1099, 1109, 843, 717, 844, 715, 701, 1101, 1117, 1125, 1133, 725, 733, 1149, 741],
                 11: [761, 1017, 633, 889, 505, 377, 249, 1145, 121, 1273, 127],
                 12: [893, 829, 885, 877, 837, 845, 853, 861, 869, 880, 752, 1008],
                 13: [1007, 999, 991, 959, 967, 975, 983, 1015, 1023, 977, 849, 721],
                 14: [1127, 961, 1119, 833, 1095, 1103, 1111, 1089, 705, 839, 1135, 831, 847, 1120, 1248, 1254, 1262,
                      1270, 1278, 855, 863],
                 15: [608, 589, 605, 1098, 597, 613, 480, 352, 586, 842, 714, 970, 224, 96, 101, 109, 117, 125],
                 16: [611, 355, 483, 227, 253, 245, 237, 229, 739, 867, 871, 995, 1123],
                 17: [491, 1003, 107, 363, 619, 747, 246, 238, 235, 875, 254, 1131],
                 18: [482, 758, 766, 750, 742, 994, 738, 610, 866, 1122, 354],
                 19: [988, 835, 1004, 996, 980, 707, 964, 972, 992, 864, 868, 876, 884, 892, 956, 963],
                 20: [632, 1144, 888, 504, 1016, 248, 376, 760, 1272, 120],
                 21: [982, 958, 966, 974, 950, 946, 818, 690, 990, 998, 1006, 1014, 1022, 692],
                 22: [442, 452, 468, 460, 1082, 444, 570, 698, 826, 954, 476, 484, 458, 492, 500, 508, 314],
                 23: [728, 216, 344, 856, 472, 236, 228, 220, 600, 244, 984, 734, 252],
                 24: [510, 502, 494, 865, 486, 481, 478, 353, 609, 993, 737, 1121, 470, 462],
                 25: [84, 72, 456, 200, 328, 92, 76, 100, 98, 226, 968, 108, 116, 584, 461, 469, 712, 124, 840, 1096],
                 26: [703, 751, 719, 727, 767, 759, 743, 735, 711, 695, 736],
                 27: [475, 493, 485, 477, 987, 603, 731, 859, 501, 509, 347, 1115],
                 28: [1011, 629, 371, 621, 627, 637, 755, 243, 499, 883, 1139, 115, 1267],
                 29: [1005, 997, 989, 1013, 965, 976, 973, 981, 957, 1021, 848, 720, 592, 596, 588],
                 30: [329, 343, 335, 585, 383, 359, 351, 457, 367, 375, 713, 841, 969, 338],
                 31: [340, 332, 324, 364, 348, 356, 316, 312, 440, 568, 696, 824, 952, 372, 380, 1080],
                 32: [1116, 1114, 1124, 1084, 1092, 1100, 1108, 986, 730, 1132, 858, 1140, 1148, 1097, 1083],
                 33: [1018, 762, 250, 634, 122, 506, 378, 890, 1020, 1146, 1274],
                 34: [757, 626, 1010, 754, 882, 498, 1138, 370, 242, 765, 114, 1266],
                 35: [1001, 617, 636, 628, 620, 873, 489, 745, 361, 233, 105, 1129, 612],
                 36: [1009, 625, 881, 497, 369, 241, 895, 753, 887, 1137, 113, 1265],
                 37: [255, 247, 872, 239, 488, 616, 232, 360, 744, 1000, 1128, 240, 112, 231]}
    sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    new_sampler = FixedEmbeddingComposite(sampler, embedding)

    alldata = []
    n_deleted = []
    v_deleted = []

    for p in percentage:
        print ("Percentage: ", p)
        Q_threshold, deleted_entrys, max = get_random_qubo(Q, p, num_entries)
        print ("deleted elements: ", deleted_entrys)
        n_deleted.append(deleted_entrys)
        print (u'Größter gelöschter Wert: ', max)
        v_deleted.append(max)
        solution = []
        kaputt = 0
        for i in range(100):
            result = solve.solveQUBO(Q_threshold, jobs, nP, new_sampler)
            #print(result)
            solution.append(result)
            if result['valid'] == False:
                kaputt += 1
        print(kaputt, " kaputte Lösungen für Threshold und ", p)
        alldata.append(solution)

    dist = []
    for i in range(len(alldata)):
        dist.append([alldata[i][j]['max_time'] for j in range(len(alldata[i]))])
    print(dist)

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.boxplot(dist)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
               [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                  1.0])
    plt.xlabel("Prozentanzahl entfernter Elemente")
    plt.ylabel("Umsteigedistanzen")
    plt.title("Jeweils 200 Aufrufe, beste Ergebnisse, Datensatz tai12, " + u'12 Flüge, 12 Gates, keine Kosten')
    tikz_save('mms_random.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')
    #plt.show()

    return alldata, n_deleted, v_deleted


def example_instance_delete_threshold(percentage):
    nP = 3
    jobs = [5, 7, 3, 5, 9, 6, 3, 1, 3, 1]
    diff = [1, 2, 2, 5]

    base_sampler = DWaveSampler()
    nodelist = base_sampler.nodelist
    edgelist = base_sampler.edgelist


    Q, num_entries, max_value = create.createQUBO(jobs, nP, diff)

    embedding = {0: [622, 614, 606, 598, 726, 594, 722, 978, 630, 638, 850, 593, 465, 337, 602, 1106, 590],
                 1: [1090, 834, 1110, 578, 1102, 1094, 962, 1118, 706, 1134, 1126, 1142, 1150, 710, 836, 718, 1112, 702,
                     583], 2: [894, 830, 838, 846, 854, 886, 878, 870, 862, 851, 723, 595, 979],
                 3: [362, 106, 490, 234, 618, 749, 1002, 746, 874, 110, 118, 126, 1130],
                 4: [342, 334, 326, 318, 382, 374, 569, 953, 366, 358, 350, 313, 441, 697, 825, 1081],
                 5: [315, 325, 341, 333, 317, 365, 357, 349, 443, 571, 699, 381, 827, 955, 373],
                 6: [123, 635, 379, 1019, 763, 251, 891, 1147, 507, 1275, 1151],
                 7: [764, 756, 716, 748, 740, 732, 724, 708, 700],
                 8: [591, 599, 607, 345, 985, 601, 729, 857, 615, 473, 639, 1113, 623, 631],
                 9: [511, 455, 704, 448, 471, 463, 960, 576, 479, 487, 496, 495, 503, 320, 624, 832, 1088],
                 10: [709, 1141, 971, 1099, 1109, 843, 717, 844, 715, 701, 1101, 1117, 1125, 1133, 725, 733, 1149, 741],
                 11: [761, 1017, 633, 889, 505, 377, 249, 1145, 121, 1273, 127],
                 12: [893, 829, 885, 877, 837, 845, 853, 861, 869, 880, 752, 1008],
                 13: [1007, 999, 991, 959, 967, 975, 983, 1015, 1023, 977, 849, 721],
                 14: [1127, 961, 1119, 833, 1095, 1103, 1111, 1089, 705, 839, 1135, 831, 847, 1120, 1248, 1254, 1262,
                      1270, 1278, 855, 863],
                 15: [608, 589, 605, 1098, 597, 613, 480, 352, 586, 842, 714, 970, 224, 96, 101, 109, 117, 125],
                 16: [611, 355, 483, 227, 253, 245, 237, 229, 739, 867, 871, 995, 1123],
                 17: [491, 1003, 107, 363, 619, 747, 246, 238, 235, 875, 254, 1131],
                 18: [482, 758, 766, 750, 742, 994, 738, 610, 866, 1122, 354],
                 19: [988, 835, 1004, 996, 980, 707, 964, 972, 992, 864, 868, 876, 884, 892, 956, 963],
                 20: [632, 1144, 888, 504, 1016, 248, 376, 760, 1272, 120],
                 21: [982, 958, 966, 974, 950, 946, 818, 690, 990, 998, 1006, 1014, 1022, 692],
                 22: [442, 452, 468, 460, 1082, 444, 570, 698, 826, 954, 476, 484, 458, 492, 500, 508, 314],
                 23: [728, 216, 344, 856, 472, 236, 228, 220, 600, 244, 984, 734, 252],
                 24: [510, 502, 494, 865, 486, 481, 478, 353, 609, 993, 737, 1121, 470, 462],
                 25: [84, 72, 456, 200, 328, 92, 76, 100, 98, 226, 968, 108, 116, 584, 461, 469, 712, 124, 840, 1096],
                 26: [703, 751, 719, 727, 767, 759, 743, 735, 711, 695, 736],
                 27: [475, 493, 485, 477, 987, 603, 731, 859, 501, 509, 347, 1115],
                 28: [1011, 629, 371, 621, 627, 637, 755, 243, 499, 883, 1139, 115, 1267],
                 29: [1005, 997, 989, 1013, 965, 976, 973, 981, 957, 1021, 848, 720, 592, 596, 588],
                 30: [329, 343, 335, 585, 383, 359, 351, 457, 367, 375, 713, 841, 969, 338],
                 31: [340, 332, 324, 364, 348, 356, 316, 312, 440, 568, 696, 824, 952, 372, 380, 1080],
                 32: [1116, 1114, 1124, 1084, 1092, 1100, 1108, 986, 730, 1132, 858, 1140, 1148, 1097, 1083],
                 33: [1018, 762, 250, 634, 122, 506, 378, 890, 1020, 1146, 1274],
                 34: [757, 626, 1010, 754, 882, 498, 1138, 370, 242, 765, 114, 1266],
                 35: [1001, 617, 636, 628, 620, 873, 489, 745, 361, 233, 105, 1129, 612],
                 36: [1009, 625, 881, 497, 369, 241, 895, 753, 887, 1137, 113, 1265],
                 37: [255, 247, 872, 239, 488, 616, 232, 360, 744, 1000, 1128, 240, 112, 231]}
    sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    new_sampler = FixedEmbeddingComposite(sampler, embedding)

    alldata = []
    n_deleted = []
    v_deleted = []

    for p in percentage:
        print ("Percentage: ", p)
        Q_threshold, deleted_entrys, max = get_threshold_qubo(p, copy.deepcopy(Q), max_value)
        print ("deleted elements: ", deleted_entrys)
        n_deleted.append(deleted_entrys)
        print (u'Größter gelöschter Wert: ', max)
        v_deleted.append(max)
        solution = []
        kaputt = 0
        for i in range(100):
            result = solve.solveQUBO(Q_threshold, jobs, nP, new_sampler)
            #print(result)
            solution.append(result)
            if result['valid'] == False:
                kaputt += 1
        print(kaputt, " kaputte Lösungen für Threshold und ", p)
        alldata.append(solution)

    dist = []
    for i in range(len(alldata)):
        dist.append([alldata[i][j]['max_time'] for j in range(len(alldata[i]))])
    print(dist)

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.boxplot(dist)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
               [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                  1.0])
    plt.xlabel("Prozentanzahl entfernter Elemente")
    plt.ylabel("Umsteigedistanzen")
    plt.title("Jeweils 200 Aufrufe, beste Ergebnisse, Datensatz tai12, " + u'12 Flüge, 12 Gates, keine Kosten')
    tikz_save('mms_tresh.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')
    #plt.show()

    return alldata, n_deleted, v_deleted


def example_instance_delete_given(percentage):
    nP = 3
    jobs = [5, 7, 3, 5, 9, 6, 3, 1, 3, 1]
    diff = [1, 2, 2, 5]

    base_sampler = DWaveSampler()
    nodelist = base_sampler.nodelist
    edgelist = base_sampler.edgelist

    # print new_sampler.properties
    # print new_sampler.parameters
    # new_sampler = VirtualGraphComposite(sampler, embedding, chain_strength=-6.0)

    Q, num_entries, max_entry = create.createQUBO(jobs, nP, diff)
    # nur noch nach absolutwerten sortiert, enthält nur die keys nicht die values
    sorted_q = sorted(Q, key=lambda dict_key: abs(Q[dict_key]))

    embedding = {0: [622, 614, 606, 598, 726, 594, 722, 978, 630, 638, 850, 593, 465, 337, 602, 1106, 590],
                 1: [1090, 834, 1110, 578, 1102, 1094, 962, 1118, 706, 1134, 1126, 1142, 1150, 710, 836, 718, 1112, 702,
                     583], 2: [894, 830, 838, 846, 854, 886, 878, 870, 862, 851, 723, 595, 979],
                 3: [362, 106, 490, 234, 618, 749, 1002, 746, 874, 110, 118, 126, 1130],
                 4: [342, 334, 326, 318, 382, 374, 569, 953, 366, 358, 350, 313, 441, 697, 825, 1081],
                 5: [315, 325, 341, 333, 317, 365, 357, 349, 443, 571, 699, 381, 827, 955, 373],
                 6: [123, 635, 379, 1019, 763, 251, 891, 1147, 507, 1275, 1151],
                 7: [764, 756, 716, 748, 740, 732, 724, 708, 700],
                 8: [591, 599, 607, 345, 985, 601, 729, 857, 615, 473, 639, 1113, 623, 631],
                 9: [511, 455, 704, 448, 471, 463, 960, 576, 479, 487, 496, 495, 503, 320, 624, 832, 1088],
                 10: [709, 1141, 971, 1099, 1109, 843, 717, 844, 715, 701, 1101, 1117, 1125, 1133, 725, 733, 1149, 741],
                 11: [761, 1017, 633, 889, 505, 377, 249, 1145, 121, 1273, 127],
                 12: [893, 829, 885, 877, 837, 845, 853, 861, 869, 880, 752, 1008],
                 13: [1007, 999, 991, 959, 967, 975, 983, 1015, 1023, 977, 849, 721],
                 14: [1127, 961, 1119, 833, 1095, 1103, 1111, 1089, 705, 839, 1135, 831, 847, 1120, 1248, 1254, 1262,
                      1270, 1278, 855, 863],
                 15: [608, 589, 605, 1098, 597, 613, 480, 352, 586, 842, 714, 970, 224, 96, 101, 109, 117, 125],
                 16: [611, 355, 483, 227, 253, 245, 237, 229, 739, 867, 871, 995, 1123],
                 17: [491, 1003, 107, 363, 619, 747, 246, 238, 235, 875, 254, 1131],
                 18: [482, 758, 766, 750, 742, 994, 738, 610, 866, 1122, 354],
                 19: [988, 835, 1004, 996, 980, 707, 964, 972, 992, 864, 868, 876, 884, 892, 956, 963],
                 20: [632, 1144, 888, 504, 1016, 248, 376, 760, 1272, 120],
                 21: [982, 958, 966, 974, 950, 946, 818, 690, 990, 998, 1006, 1014, 1022, 692],
                 22: [442, 452, 468, 460, 1082, 444, 570, 698, 826, 954, 476, 484, 458, 492, 500, 508, 314],
                 23: [728, 216, 344, 856, 472, 236, 228, 220, 600, 244, 984, 734, 252],
                 24: [510, 502, 494, 865, 486, 481, 478, 353, 609, 993, 737, 1121, 470, 462],
                 25: [84, 72, 456, 200, 328, 92, 76, 100, 98, 226, 968, 108, 116, 584, 461, 469, 712, 124, 840, 1096],
                 26: [703, 751, 719, 727, 767, 759, 743, 735, 711, 695, 736],
                 27: [475, 493, 485, 477, 987, 603, 731, 859, 501, 509, 347, 1115],
                 28: [1011, 629, 371, 621, 627, 637, 755, 243, 499, 883, 1139, 115, 1267],
                 29: [1005, 997, 989, 1013, 965, 976, 973, 981, 957, 1021, 848, 720, 592, 596, 588],
                 30: [329, 343, 335, 585, 383, 359, 351, 457, 367, 375, 713, 841, 969, 338],
                 31: [340, 332, 324, 364, 348, 356, 316, 312, 440, 568, 696, 824, 952, 372, 380, 1080],
                 32: [1116, 1114, 1124, 1084, 1092, 1100, 1108, 986, 730, 1132, 858, 1140, 1148, 1097, 1083],
                 33: [1018, 762, 250, 634, 122, 506, 378, 890, 1020, 1146, 1274],
                 34: [757, 626, 1010, 754, 882, 498, 1138, 370, 242, 765, 114, 1266],
                 35: [1001, 617, 636, 628, 620, 873, 489, 745, 361, 233, 105, 1129, 612],
                 36: [1009, 625, 881, 497, 369, 241, 895, 753, 887, 1137, 113, 1265],
                 37: [255, 247, 872, 239, 488, 616, 232, 360, 744, 1000, 1128, 240, 112, 231]}
    sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    new_sampler = FixedEmbeddingComposite(sampler, embedding)

    alldata = []
    n_deleted = []
    v_deleted = []

    for p in percentage:
        print("Percentage: ", p)
        Q_deleted, max, deleted = get_given_percentage_qubo(p, copy.deepcopy(Q), copy.deepcopy(sorted_q), num_entries)
        n_deleted.append(deleted)
        print ("Größter gelöschter Wert: ", max)
        v_deleted.append(max)
        solution = []
        kaputt = 0
        for i in range(100):
            result = solve.solveQUBO(Q_deleted, jobs, nP, new_sampler)
            #print(result)
            solution.append(result)
            if result['valid'] == False:
                kaputt += 1
        print(kaputt, " kaputte Lösungen für Threshold und ", p)
        alldata.append(solution)

    dist = []
    for i in range(len(alldata)):
        dist.append([alldata[i][j]['max_time'] for j in range(len(alldata[i]))])
    print (dist)

    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.boxplot(dist)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
               [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                  0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                  1.0])
    plt.xlabel("Prozentanzahl entfernter Elemente")
    plt.ylabel("Umsteigedistanzen")
    plt.title("Jeweils 200 Aufrufe, beste Ergebnisse, Datensatz tai12, " + u'12 Flüge, 12 Gates, keine Kosten')
    tikz_save('mms_given.tikz',
              figureheight='\\figureheight',
              figurewidth='\\figurewidth')
    #plt.show()

    return alldata, n_deleted, v_deleted


def evaluate_all():

    # alldata, n_deleted, v_deleted = example_instance_delete_given([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    #               0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    # with open("result-given-qc.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(alldata)):
    #         writer.writerow(alldata[i])
    #
    # with open("n_v_deleted-given-qc.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(n_deleted)
    #     writer.writerow(v_deleted)
    #
    # alldata, n_deleted, v_deleted = example_instance_delete_threshold([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    #                                                            0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    #                                                            1.0])
    #
    # with open("result-given-qc.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(alldata)):
    #         writer.writerow(alldata[i])
    #
    # with open("n_v_deleted-given-qc.csv", 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(n_deleted)
    #     writer.writerow(v_deleted)
    #
    alldata, n_deleted, v_deleted = example_instance_delete_random([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                                      0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                                                      1.0])

    with open("result-random-qc.csv", 'w') as file:
        writer = csv.writer(file)
        for i in range(len(alldata)):
            writer.writerow(alldata[i])

    with open("n_v_deleted-random-qc.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(n_deleted)
        writer.writerow(v_deleted)


if __name__ == "__main__":

    evaluate_all()

    #example_instance()
    # urllib3.contrib.pyopenssl.inject_into_urllib3()
    # jobs = [1,2,3,100]
    # nP = 2
    # diff = [94]
    #
    # solutions = []
    #
    # Q = create.createQUBO(jobs, nP, diff)
    # solve.solution_distribution(Q, len(jobs), nP, diff, 9, 1)
    #
    # embedding = {}
    # embedding = {0: [1760, 1632, 1764], 1: [1628, 1636, 1634], 2: [1638, 1630, 1627], 3: [1752, 1624, 1759, 1767],
                # 4: [1763, 1635, 1765], 5: [1761, 1633], 6: [1754, 1626, 1758, 1766], 7: [1631, 1639, 1625],
                # 8: [1637, 1629]}

    # base_sampler = DWaveSampler()
    # G = dnx.chimera_graph(16,16,4)
    # nodelist = G.nodes()
    # edgelist = G.edges()
    # source_edgelist = list(Q)
    #
    # if embedding == {}:
    #     embedding = minorminer.find_embedding(source_edgelist, edgelist)
    # sampler = dimod.StructureComposite(base_sampler, nodelist, edgelist)
    # new_sampler = FixedEmbeddingComposite(sampler, embedding)
    # print embedding
    # embed.draw_chimera_graph(embedding)
    #
    # for i in range(10):
    #     solutions.append(solve.solveQUBO(Q, jobs, nP, new_sampler))
    # solve.evaluate(solutions)