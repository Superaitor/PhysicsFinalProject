# This is a sample Python script.
import pickle
import time
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier
from math import exp
import sys
from multiprocessing import Pool, cpu_count

np.set_printoptions(threshold=sys.maxsize)
from scipy.optimize import dual_annealing
import os

glo = 0


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Saves data list from a simulation so it may be used later
def save_ML_data(train_data, train_labels, test_data, test_labels):
    with open("training_data.txt", "wb") as fp:
        pickle.dump(train_data, fp)
    with open("training_labels.txt", "wb") as fp:
        pickle.dump(train_labels, fp)
    with open("testing_data.txt", "wb") as fp:
        pickle.dump(test_data, fp)
    with open("testing_labels.txt", "wb") as fp:
        pickle.dump(test_labels, fp)


# Returns saved list of data from memory
def read_ML_data():
    with open("training_data.txt", "rb") as fp:  # Unpickling
        train_data = pickle.load(fp)
    with open("training_labels.txt", "rb") as fp:  # Unpickling
        train_labels = pickle.load(fp)
    with open("testing_data.txt", "rb") as fp:  # Unpickling
        test_data = pickle.load(fp)
    with open("testing_labels.txt", "rb") as fp:  # Unpickling
        test_labels = pickle.load(fp)

    return train_data, train_labels, test_data, test_labels


# Returns the saved lists created by find_T1_to_optimal_thresh_and_it_data
def read_T1_to_optimal_thresh_and_it_data():
    with open("thresholds.txt", "rb") as fp:  # Unpickling
        thresholds = pickle.load(fp)
    with open("int_times.txt", "rb") as fp:  # Unpickling
        int_times = pickle.load(fp)
    with open("fidelities.txt", "rb") as fp:  # Unpickling
        fidelities = pickle.load(fp)
    with open("t1s.txt", "rb") as fp:  # Unpickling
        t1s = pickle.load(fp)

    return thresholds, int_times, fidelities, t1s


# Prepares data for training and testing, returning training data, testing data, and labels
def prepare_data(upper_distribution, lower_distribution, low, high):
    training_data = upper_distribution + lower_distribution
    training_labels = ([1] * len(upper_distribution)) + ([0] * len(lower_distribution))
    testing_upper, testing_lower = create_multiple_distributions(low, high,
                                                                 5000)  # Create a smaller distribution for testing
    testing_data = testing_upper + testing_lower
    testing_labels = ([1] * len(testing_upper)) + ([0] * len(testing_lower))
    return training_data, training_labels, testing_data, testing_labels


# Simulates values of signals returned accounting for noise and possible qubit state change
def noise_simulator(M0, it, T1):
    t_between_pts = 0.005  # time between each measured point (in microseconds)
    num_points = it / t_between_pts
    y = []
    noise_mean = 0  # mean of noise
    noise_sd = 5  # standard deviation of noise
    # T1 = 10  # standard time for qubit to change states (in microseconds)
    for i in range(1, int(num_points) + 1):
        y.append(M0 + random.gauss(noise_mean, noise_sd))  # add a new random point with noise to graph
        decay_prob = 1 - exp((-t_between_pts) / T1)  # calculate the probability of qubit change to lower state
        if random.random() < decay_prob and M0 == 1:  # if random value ([0,1]) is larger, then qubit decays
            # print("here " , i, decay_prob)
            M0 = -1
    return y


# Creates a list of lists of length "num_iterations" simulations containing various iterations of the noise_simulator
# function. Essentially, it runs "num_iterations" number of cubit simulations, running noise_simulator and saving
# each simulation into a list. Useful for intensive operations (such as DA_threshold_it) where you may not want to
# create simulations over and over again, saving time. Creates size/2 simulations starting in lower state,
# size/2 in upper. Returns two lists of lists, one of simulations starting in upper state, the other in lower state
def noise_simulations(num_iterations, trial_time, T1):
    upper_simulations = []
    M0 = 1
    for i in range(0, int(num_iterations / 2)):
        upper_simulations.append(noise_simulator(M0, trial_time, T1))
    lower_simulations = []
    M0 = -1
    for i in range(0, int(num_iterations / 2)):
        lower_simulations.append(noise_simulator(M0, trial_time, T1))
    return upper_simulations, lower_simulations


# Makes a distribution of all the sums of the observed signals, for the given number of iterations, starting in state M0
def distributions(M0, num_iterations, it, T1):
    distribution = []
    for i in range(0, num_iterations):
        y = noise_simulator(M0, it, T1)
        t_between_pts = 0.005  # time between each measured point (in microseconds)
        num_points = it / t_between_pts
        integral = sum(y)
        # integral = integral / num_points  # done to normalize between different integration times, delete if not wanted
        distribution.append([integral, it])
    return distribution


# Returns two lists of distributions for the given low range to high range of integration times, as well as a list of
# labels denoting if the qubit started with spin up or spin down. Note: This returns two distributions (one starting
# low and one starting high for the purposes of plotting them and getting enough variety in training data, but those
# distributions must later be combined into one in order to train the random forest model. Size is distribution size,
# where half of the distribution started in upper state, other half started in lower state
def create_multiple_distributions(low, high, size, T1):
    upper_dist = []  # Starting at the upper state
    lower_dist = []  # Starting at the lower state
    M0 = 1  # original state of the qubit |1>, may later decay to |0>
    for integration_time in range(low, high):
        upper_dist = upper_dist + distributions(M0, int(size / 2),
                                                integration_time, T1)  # simulate 1000 iterations starting spin up
    M0 = -1
    for integration_time in range(low, high):
        lower_dist = lower_dist + distributions(M0, int(size / 2),
                                                integration_time, T1)  # sim 1000 iterations starting spin down
    return upper_dist, lower_dist


# Like create_distributions, but for only one integration time
def create_distribution(it, size, T1):
    M = 1  # original state of the qubit |1>, may later decay to |0>
    upper_dist = distributions(M, int(size / 2), it, T1)  # simulate size/2 iterations starting spin up
    M = -1
    lower_dist = distributions(M, int(size / 2), it, T1)  # sim size/2 iterations starting spin down
    # now simplify the distributions to a 1d list that does not include the integration time
    columns = list(zip(*upper_dist))
    upper_dist = columns[0]
    columns = list(zip(*lower_dist))
    lower_dist = columns[0]
    return upper_dist, lower_dist


# Makes a histogram of the given two distributions
def plot_distributions(distribution1, distribution2):
    hist1, edges1 = np.histogram(distribution1, bins=250)
    hist2, edges2 = np.histogram(distribution2, bins=250)
    plt.plot(edges1[:-1], hist1)
    plt.plot(edges2[:-1], hist2)
    plt.xlabel("Sum of the points")
    plt.ylabel("Number of instances")
    plt.show()


# An alternative approach for finding the threshold, puts data into histogram and then uses binary search on the
# different bins, calculating the threshold value until best one is found.
def binary_bins_threshold(dist1, dist2):
    # Makes the two histograms
    hist1, edges1 = np.histogram(dist1, bins=250)
    hist2, edges2 = np.histogram(dist2, bins=250)
    edges = edges1 + edges2  # combine all histogram x values
    edges.sort()  # sort these values
    low = 0
    high = len(edges)
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if (3 * mid) // 2 < len(edges):
            # if left half is better than right half, resume binary search on left half
            left_error = calculate_error(edges[mid // 2], dist1, dist2)
            right_error = calculate_error(edges[(3 * mid) // 2], dist1, dist2)
            print("left ", left_error)
            print("right ", right_error)
            if left_error < right_error:
                high = mid - 1
            elif left_error > right_error:
                low = mid + 1
            else:
                return edges[mid]
        else:
            return edges[mid]
    return edges[mid]


# An alternative approach for finding the threshold, puts data into histogram and then uses linear search on the
# different bins, calculating the threshold value until best one is found.
def linear_bins_threshold(dist1, dist2):
    # Makes the two histograms
    hist1, edges1 = np.histogram(dist1, bins=250)
    hist2, edges2 = np.histogram(dist2, bins=250)
    edges = edges1 + edges2  # combine all histogram x values
    edges.sort()  # sort these values
    min_thresh = 9999999  # set large number as min to start off
    min_error = 9999999
    for bin_val in edges:
        error = calculate_error(bin_val, dist1, dist2)
        if error < min_error:
            min_error = error
            min_thresh = bin_val
    return min_thresh


# Calculates error of the distributions relative to the threshold (Number of incorrect classifications vs total)
# d1 represents the distribution starting in the upper state, d2 starting in the lower state
def calculate_error_cpp(threshold, d1, d2):
    dist_size = len(d1) + len(d2)
    threshold = threshold[0]
    d1_array = (ctypes.c_float * len(d1))(*d1)
    d2_array = (ctypes.c_float * len(d2))(*d2)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    handle = ctypes.CDLL(dir_path + "/fasterTest.so")
    handle.Calculate_error.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
                                       ctypes.POINTER(ctypes.c_float)]
    handle.Calculate_error.restype = ctypes.c_float
    answer = handle.Calculate_error(threshold, dist_size, d1_array, d2_array)
    return answer


# Calculates error of the distributions relative to the threshold (Number of incorrect classifications vs total)
# d1 represents the distribution starting in the upper state, d2 starting in the lower state
def calculate_error(threshold, d1, d2):
    dist_size = len(d1) + len(d2)
    incorrect = 0
    for nums in d1:
        if nums < threshold:
            incorrect += 1
    for nums in d2:
        if nums > threshold:
            incorrect += 1
    fidelity = incorrect / dist_size
    # print(fidelity)
    return fidelity


def sum_simulation(sim, num_points):
    summation = 0
    for i in range(0, int(num_points)):  # For the number of points that this new integration time would give...
        summation += sim[i]
    return summation


# Same as advanced_calculate_error, but runs in C++ as well
def advanced_calculate_error_cpp(threshold_it, s1_array, s2_array, sim_size):
    threshold = threshold_it[0]
    # print("thresh ", threshold)
    it = threshold_it[1]
    # print("it ", it)
    t_between_pts = 0.005  # time between each measured point (in microseconds) Note: Remember to always keep this
    # number the same as the one in noise_simulator!
    num_points = int(it / t_between_pts)  # the number of points this new integration time would give per simulation

    # print(thresh)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    handle = ctypes.CDLL(dir_path + "/fasterTest.so")

    # print(len(s1_array[0]))
    handle.Advanced_calculate_error.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_int,
                                                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                                ctypes.POINTER(ctypes.POINTER(ctypes.c_float))]
    handle.Advanced_calculate_error.restype = ctypes.c_float
    answer = handle.Advanced_calculate_error(threshold, sim_size, num_points,
                                             ctypes.cast(s1_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
                                             ctypes.cast(s2_array, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))))

    return answer


# This is the function to be minimized by dual annealing in DA_threshold_it. It takes in two 2d arrays filled with
# simulations (one starting in upper state the other in lower state) and the dual annealing guess for threshold and
# integration time threshold_it (an array of only two values) and creates distributions for them for the given
# integration time. Then it calculates the error for the given threshold. Returns that error.
def advanced_calculate_error(threshold_it, upper_simulations, lower_simulations):
    threshold = threshold_it[0]
    # print("thresh ", threshold)
    it = threshold_it[1]
    # print("it ", it)
    t_between_pts = 0.005  # time between each measured point (in microseconds) Note: Remember to always keep this
    # number the same as the one in noise_simulator!
    num_points = it / t_between_pts  # the number of points this new integration time would give per simulation
    d1 = []  # The distribution of simulation sums starting in upper state
    for sim in upper_simulations:
        summation = 0
        for i in range(0, int(num_points)):  # For the number of points that this new integration time would give...
            summation += sim[i]
        d1.append(summation)
    d2 = []  # The distribution of simulation sums starting in lower state
    for sim in lower_simulations:
        summation = 0
        for i in range(0, int(num_points)):
            summation += sim[i]
        d2.append(summation)

    dist_size = len(d1) + len(d2)
    incorrect = 0
    for nums in d1:
        if nums < threshold:
            incorrect += 1
    for nums in d2:
        if nums > threshold:
            incorrect += 1
    fidelity = incorrect / dist_size
    global glo
    # print(glo)
    glo += 1
    return fidelity


# The non-machine learning approach. Calculates the appropriate threshold, with two distributions (dist 1 being the one
# that starts with the upper state, dist 2 lower state) and integration time taken as parameters.
# This uses a dual annealing (simulated annealing) method to increase processing speed.
def dual_annealing_threshold(dist1, dist2):
    threshold = np.array([5.0])
    # Make edges to the threshold lookup be 2 standard deviations away
    min = [-2 * np.std(dist2)]
    max = [2 * np.std(dist1)]
    min_threshold = dual_annealing(calculate_error, bounds=list(zip(min, max)), x0=threshold, args=(dist1, dist2))
    return min_threshold


# This can take in lab data of simultions instead of making the simulations itself.
def lab_data_dual_annealing_threshold(sim_upper, sim_lower):
    dist1 = []
    for sim in sim_upper:
        dist1.append(sum(sim))
    dist2 = []
    for sim in sim_lower:
        dist2.append(sum(sim))
    threshold = np.array([5.0])
    # Make edges to the threshold lookup be 2 standard deviations away
    min = [-2 * np.std(dist2)]
    max = [2 * np.std(dist1)]
    min_threshold = dual_annealing(calculate_error, bounds=list(zip(min, max)), x0=threshold, args=(dist1, dist2))
    return min_threshold


# This can take in lab data of simultions instead of making the simulations itself.
def lab_data_dual_annealing_threshold_cpp(sim_upper, sim_lower):
    dist1 = []
    for sim in sim_upper:
        dist1.append(sum(sim))
    dist2 = []
    for sim in sim_lower:
        dist2.append(sum(sim))
    threshold = np.array([5.0])
    # Make edges to the threshold lookup be 2 standard deviations away
    min = [-2 * np.std(dist2)]
    max = [2 * np.std(dist1)]
    min_threshold = dual_annealing(calculate_error_cpp, bounds=list(zip(min, max)), x0=threshold, args=(dist1, dist2))
    return min_threshold


# The non-machine learning approach. Calculates the appropriate threshold, with two distributions (dist 1 being the one
# that starts with the upper state, dist 2 lower state) and integration time taken as parameters.
# This uses a dual annealing (simulated annealing) method to increase processing speed.
def dual_annealing_threshold_cpp(dist1, dist2):
    threshold = np.array([5.0])
    # Make edges to the threshold lookup be 2 standard deviations away
    min = [-2 * np.std(dist2)]
    max = [2 * np.std(dist1)]
    min_threshold = dual_annealing(calculate_error_cpp, bounds=list(zip(min, max)), x0=threshold, args=(dist1, dist2))
    return min_threshold


# Same as DA_threshold_it, but uses C++
def DA_threshold_it_cpp(T1, upper_simulations, lower_simulations, threshold_guess):
    # tic = time.perf_counter()
    # print("here")
    if T1 < 10:
        threshold_it = np.array([threshold_guess, T1 / 2])
    else:
        threshold_it = np.array([threshold_guess, 5])
    min = [-3000, 0.1]
    max = [3000, 15]
    sim_size = len(upper_simulations) + len(lower_simulations)
    if len(upper_simulations[0]) > 3000:  # Necessary so the array is the right size for the c++ code
        for sim in upper_simulations:
            sim = sim[:3000]
        for sim in lower_simulations:
            sim = sim[:3000]
    elif len(upper_simulations[0]) < 3000:
        remaining = 3000 - len(upper_simulations[0])
        for sim in upper_simulations:
            sim += ([0] * remaining)
        for sim in lower_simulations:
            sim += ([0] * remaining)
    s1_array = (ctypes.c_float * len(upper_simulations[0]) * len(upper_simulations))(
        *(tuple(i) for i in upper_simulations))
    s2_array = (ctypes.c_float * len(lower_simulations[0]) * len(lower_simulations))(
        *(tuple(i) for i in lower_simulations))
    result = dual_annealing(advanced_calculate_error_cpp, bounds=list(zip(min, max)), x0=threshold_it,
                            args=(s1_array, s2_array, sim_size))
    return result


# Finds the best integration time and best threshold for the given T1. Does this by using the dual annealing algorithm
# with two varying parameters, calling advanced_calculate_error() and optimizing it. Takes in as parameters a best guess
# for the threshold as well as T1 (typical time for cubit to decay from upper to lower state). Returns the optimization
# result represented as a OptimizeResult object, Important attributes are: x the solution array (x[0] js threshold, x[1]
# is integration time), fun the value of the function at the solution (the fidelity, and message which describes the
# cause of the termination. Search up OptimizeResult for a description of other attributes.
def DA_threshold_it(T1, upper_simulations, lower_simulations, threshold_guess):
    # tic = time.perf_counter()
    if T1 < 10:
        threshold_it = np.array([threshold_guess, T1 / 2])
    else:
        threshold_it = np.array([threshold_guess, 5])
    min = [-3000, 0.1]
    max = [3000, 15]

    result = dual_annealing(advanced_calculate_error, bounds=list(zip(min, max)), x0=threshold_it,
                            args=(upper_simulations, lower_simulations), maxiter=200)
    # pool.close()
    toc = time.perf_counter()
    print("time at finish:", toc)

    return result


# NOT to be used for running by the user, this is a helper function for parallel_find_t1_to_optimal_thresh_and_it
def parallel_DA_threshold_it_cpp(T1):
    # tic = time.perf_counter()
    if T1 < 10:
        threshold_it = np.array([0, T1 / 2])
    else:
        threshold_it = np.array([0, 5])
    min = [-3000, 0.1]
    max = [3000, 13]
    upper_simulations, lower_simulations = noise_simulations(8000, 15, T1)  # Create 8000 cubit noise simulations, each
    sim_size = len(upper_simulations) + len(lower_simulations)
    s1_array = (ctypes.c_float * len(upper_simulations[0]) * len(upper_simulations))(
        *(tuple(i) for i in upper_simulations))
    s2_array = (ctypes.c_float * len(lower_simulations[0]) * len(lower_simulations))(
        *(tuple(i) for i in lower_simulations))
    result = dual_annealing(advanced_calculate_error_cpp, bounds=list(zip(min, max)), x0=threshold_it,
                            args=(s1_array, s2_array, sim_size))

    toc = time.perf_counter()
    print("time at finish:", toc)
    print(result)

    return result


# NOT to be used for running by the user, this is a helper function for parallel_find_t1_to_optimal_thresh_and_it
# Same as DA_threshold_it except this one is used for the parallelized functions, and does not take an initial
# threshold guess
def parallel_DA_threshold_it(T1):
    # tic = time.perf_counter()
    if T1 < 10:
        threshold_it = np.array([0, T1 / 2])
    else:
        threshold_it = np.array([0, 5])
    min = [-3000, 0.1]
    max = [3000, 13]
    upper_simulations, lower_simulations = noise_simulations(8000, 15, T1)  # Create 8000 cubit noise simulations, each
    result = dual_annealing(advanced_calculate_error, bounds=list(zip(min, max)), x0=threshold_it,
                            args=(upper_simulations, lower_simulations), maxiter=150)
    toc = time.perf_counter()
    print("time at finish:", toc)
    print(result)

    return result


# Takes in the result from DA_threshold_it (The optimization result represented as a OptimizeResult object) as a
# parameter as well as T1 (which should be the same as the one that was previously fed into DA_threshold_it), creates a
# brand new set of testing data, and calculates the fidelity of the result of DA_threshold_it on this new testing data.
def test_DA_threshold_it(result, T1):
    threshold = result.x[0]
    it = result.x[1]  # Takes the optimal integration time found from
    upper_dist, lower_dist = create_distribution(it, 10000, T1)  # Creates distribution of size 10000 w/ it from result
    fidelity = 1 - calculate_error(threshold, upper_dist, lower_dist)
    return fidelity


# Same as find_t1_to_optimal_thresh_and_it, but uses multiprocessing and C++
def find_T1_to_optimal_thresh_and_it_cpp():
    thresholds = []
    fidelity_list = []
    i_times = []
    t1_array = np.array([])
    t1s = np.linspace(0.1, 2, 20)
    t1_array = np.concatenate([t1_array, t1s])
    pool = Pool(cpu_count())
    results = pool.map(parallel_DA_threshold_it_cpp, [t for t in t1s])
    t1 = 0.1
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 0.1
        print("T1: ", t1)
        print(result)
    pool.close()
    pool2 = Pool(cpu_count())
    t1s = np.linspace(2, 10, 16)
    t1_array = np.concatenate([t1_array, t1s])
    results = pool2.map(parallel_DA_threshold_it_cpp, [t for t in t1s])
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 0.5
        print("T1: ", t1)
        print(result)
    pool2.close()
    pool3 = Pool(cpu_count())
    t1s = np.linspace(10, 100, 45)
    t1_array = np.concatenate([t1_array, t1s])
    results = pool3.map(parallel_DA_threshold_it_cpp, [t for t in t1s])
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 2
        print("T1: ", t1)
        print(result)
    pool3.close()
    with open("thresholds.txt", "wb") as fp:
        pickle.dump(thresholds, fp)
    with open("int_times.txt", "wb") as fp:
        pickle.dump(i_times, fp)
    with open("fidelities.txt", "wb") as fp:
        pickle.dump(fidelity_list, fp)
    with open("t1s.txt", "wb") as fp:
        pickle.dump(t1_array, fp)
    return thresholds, i_times, fidelity_list, t1_array


# Same as find_t1_to_optimal_thresh_and_it, but uses multiprocessing
def parallel_find_T1_to_optimal_thresh_and_it():
    thresholds = []
    fidelity_list = []
    i_times = []
    t1_array = np.array([])
    t1s = np.linspace(0.1, 2, 20)
    t1_array = np.concatenate([t1_array, t1s])
    pool = Pool(cpu_count())
    results = pool.map(parallel_DA_threshold_it, [t for t in t1s])
    t1 = 0.1
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 0.1
        print("T1: ", t1)
        print(result)
    pool.close()
    pool2 = Pool(cpu_count())
    t1s = np.linspace(2, 10, 16)
    t1_array = np.concatenate([t1_array, t1s])
    results = pool2.map(parallel_DA_threshold_it, [t for t in t1s])
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 0.5
        print("T1: ", t1)
        print(result)
    pool2.close()
    pool3 = Pool(cpu_count())
    t1s = np.linspace(10, 100, 45)
    t1_array = np.concatenate([t1_array, t1s])
    results = pool3.map(parallel_DA_threshold_it, [t for t in t1s])
    for result in results:
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1 += 2
        print("T1: ", t1)
        print(result)
    pool3.close()
    with open("thresholds.txt", "wb") as fp:
        pickle.dump(thresholds, fp)
    with open("int_times.txt", "wb") as fp:
        pickle.dump(i_times, fp)
    with open("fidelities.txt", "wb") as fp:
        pickle.dump(fidelity_list, fp)
    with open("t1s.txt", "wb") as fp:
        pickle.dump(t1_array, fp)
    return thresholds, i_times, fidelity_list, t1_array


# Note: Takes several hours,  returns arrays that can be used for plotting. This
# function runs DA_threshold_it several times for a multitude of different integration times. It then returns the
# thresholds, integration times, and fidelities that were found for each subsequent T1. To see how they fared,
# plot any of those 3 first arrays returned with the array of T1's (the last, fourth array returned). This way you can
# see how each of those variables changed as the T1 variable changed. Remember, the 1st value of the T1 array accounts
# for the same trial of the first value of the threshold array (or it/fidelity ones), the 2nd value of T1 array accounts
# for the same trial of the 2nd value of the other arrays, and so on. It saves all these values to some text files on
# your computer, so you don't have to repeat after running the function for hours.
def find_T1_to_optimal_thresh_and_it():
    t1 = 0.1
    thresholds = []
    i_times = []
    fidelity_list = []
    t1s = []
    previous_threshold = 0  # Saves the previous threshold chosen by the dual annealing to boost optimization
    # t1s = np.linspace(0.1, 2, 20)
    # results = pool.map(DA_threshold_it, [t for t in t1s])
    while t1 < 2:
        result = DA_threshold_it(t1, previous_threshold)
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1s.append(t1)
        print("t1:", t1)
        t1 += 0.1
        previous_threshold = result.x[0]
    while t1 < 10:
        result = DA_threshold_it(t1)
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1s.append(t1)
        print("t1:", t1)
        t1 += 0.5
    while t1 < 100:
        result = DA_threshold_it(t1)
        thresholds.append(result.x[0])
        i_times.append(result.x[1])
        error = result.fun
        fidelity_list.append(1 - error)
        t1s.append(t1)
        print("t1:", t1)
        t1 += 2
        print(result)
    with open("thresholds.txt", "wb") as fp:
        pickle.dump(thresholds, fp)
    with open("int_times.txt", "wb") as fp:
        pickle.dump(i_times, fp)
    with open("fidelities.txt", "wb") as fp:
        pickle.dump(fidelity_list, fp)
    with open("t1s.txt", "wb") as fp:
        pickle.dump(t1s, fp)
    return thresholds, i_times, fidelity_list, t1s


# Can plot the arrays returned by T1_to_optimal_thresh_and_it
def plot_T1_to_optimal_thresh_and_it(thresholds, i_times, fidelity_list, t1s):
    plt.plot(t1s, thresholds)
    plt.ylabel('Thresholds')
    plt.xlabel('T1')
    plt.show()
    plt.clf()
    plt.plot(t1s, i_times)
    plt.ylabel('Integration times')
    plt.xlabel('T1')
    plt.show()
    plt.clf()
    plt.plot(t1s, fidelity_list)
    plt.ylabel('Best fidelity')
    plt.xlabel('T1')
    plt.show()
    plt.clf()


# Trains a random forest binary classifier on the two distributions with different integration times.
def train_random_forest(distribution, spins):
    model = RandomForestClassifier()
    model.fit(distribution, spins)
    return model
    # filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))


# Runs DA_threshold_it number_iterations times, returning an array of the results for thresholds, it's and fidelity
# each time. Also saves these into a txt file so this doesn't have to be repeated several times over later, since this
# function can take a while. These arrays can then be plotted after to see how the variance between trials looked.
def find_variance(T1, number_iterations):
    thresholds = []
    its = []
    fidelities = []
    for i in range(0, number_iterations):
        result = DA_threshold_it(T1)
        thresholds.append(result.x[0])
        its.append((result.x[1]))
        fidelities.append(1 - result.fun)
        print("i", i)
        print("thresh", result.x[0])
        print("integration time", result.x[1])
        print("fidelity", 1 - result.fun)
    with open("variance_thresholds.txt", "wb") as fp:  # Unpickling
        pickle.dump(thresholds, fp)
    with open("variance_its.txt", "wb") as fp:  # Unpickling
        pickle.dump(its, fp)
    with open("variance_fidelities.txt", "wb") as fp:
        pickle.dump(fidelities, fp)
    return thresholds, its, fidelities


# MAIN FUNCTION, RUN CODE HERE
# Press the green button in the gutter to run the script. Here is where you may run all the functions above.
if __name__ == '__main__':
    # tic = time.perf_counter()
    # EXAMPLE:
    upper_dist, lower_dist = create_distribution(5, 10000, 10)
    thresh = dual_annealing_threshold_cpp(upper_dist, lower_dist)
    print(thresh)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
