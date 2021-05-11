

NOTE; For a more readable version of this, look at the end of the final project PDF

The following functions are not an exhaustive list of all the functions made in the project, but only the most important ones / the ones that should be called. Some of the functions that aren’t meant to be called (such as noise_simulator) are explained in the project document above. These are the only functions that are meant to be called by the user, do not call the others. 


def create_distribution(it, size)

This function makes a distribution of size size within a given integration time it, simulating the noise, adding trails up, and finalizing the distribution.The function returns two lists (each of size size/2) , the first of which represents the distributions that started in the upper state (M = 0.5) and the second of which represents the distributions that started in the lower state (M = -0.5). Example:
upper_dist, lower_dist = create_distribution(4, 10000)
upper_dist is the upper state list, lower_dist is the lower state list. Integration time is 4.

def create_multiple_distributions(low, high, size)

This function makes distributions of size size within a given low to high integration time range, simulating the noise, adding trails up, and finalizing the distribution.The function returns two lists (each of size size/2) , the first of which represents the distributions that started in the upper state (M = 0.5) and the second of which represents the distributions that started in the lower state (M = -0.5). If the range low, high is 5 to 6, then two lists will be returned, each of which were calculated with only integration time 5. If the range was 1 to 6, the function would return 2 lists still, but each of the lists would have integration times 1, 2, 3, 4, and 5, and the list would also be 5 times larger than the one calculated with the range of 5 to 6. This added functionality of multiple distributions for different integration times is added in case the user wants to train a model that understands multiple different integration times all at once. 
Example:
upper_dist, lower_dist = create_multiple_distributions(4, 9, 10000)
upper_dist the upper state list, lower_dist is the lower state list. Integration time is 4, 5, 6, 7, and 8 for this range, and the size of each is 5,000. 

def plot_distributions(distribution1, distribution2) 

Plots two different distributions into a histogram.  For most purposes, distribution1 will represent the first list returned by create_distributions (trials starting in upper state), and distribution2 will represent the second list (lower state), for only one integration time. 
Example: 
upper_dist, lower_dist = create_distribution(5, 10000)
plot_distributions(upper_dist, lower_dist)
Output: 



def linear_bins_threshold(dist1, dist2)

Returns the best threshold value using a linear step through histogram bins. First creates a histogram from distributions dist1 and dist2.  For most purposes, dist1 will represent the first list returned by create_distributions (trials starting in upper state), and dist2 will represent the second list (lower state), for only one integration time. Function then steps through each of the histogram bins and finds the bin where the error is smallest. Error is calculated as incorrect upper + incorrect lower / total. The numerator here corresponds to the total number of trials that were incorrectly classified due to them being at the wrong side of the threshold for both the trials that started on the lower state and those that started in the upper state. The denominator is the total number of trials. This error is measured for any given threshold using the function def calculate_error(threshold, d1, d2). Once the histogram bin with the smallest error is found, the corresponding x value is returned.
Example:
upper_dist, lower_dist = create_distribution(5, 10000)
threshold = linear_bins_threshold(upper_dist, lower_dist)
Output of print(threshold):
-201.48506047125466
Note: The fidelity (1 - calculate_error(threshold, upper_dist, lower_dist)) of this threshold for the above integration time and distributions is 0.9181


def dual_annealing_threshold(dist1, dist2):

Returns the best threshold value using a dual annealing approach. For most purposes, dist1 will represent the first list returned by create_distributions (trials starting in upper state), and dist2 will represent the second list (lower state), for only one integration time. Dual annealing is imported from scipy.optimize import dual_annealing, which attempts to minimize the def calculate_error(threshold, d1, d2) function, with threshold as the variable to be changed and the two distributions returned by create_distributions as constants. The function returns an object that describes the result obtained, including the minimum error found by dual annealing as well as the threshold value that obtained that error. 
Example:
upper_dist, lower_dist = create_distribution(5, 10000)
threshold = dual_annealing_threshold(upper_dist, lower_dist)
Output of print(threshold):
fun: 0.0801
 message: ['Maximum number of iteration reached']
    nfev: 2021
    nhev: 0
     nit: 1000
    njev: 10
  status: 0
 success: True
       x: array([-203.35351449])

In the above output, fun: 0.0801 means that the optimized error found was 0.0801, or a fidelity of 0.9199.  x: array([-203.35351449]) means that the threshold value that returned this error score was -203.35. 


def dual_annealing_threshold_cpp(dist1, dist2):

The same as dual_annealing_threshold, but runs in C++. This makes it more than 3x faster. Remember to run g++ -fPIC -shared -o fasterTest.so faster.cpp on your terminal before running this, else it will not work. 


def lab_data_dual_annealing_threshold(sim_upper, sim_lower)

The same as dual_annealing_threshold, but takes in two 2d arrays of simulations instead, making it possible to take in lab data. Assumes the lab data is an array of arrays of time series representing the life cycle of the qubit. 
Example
threshold = dual_annealing_threshold(upper_sim, lower_sim)
Where the first parameter is the upper state simulation and the lower state is the other


def lab_data_dual_annealing_threshold_cpp(sim_upper, sim_lower)

The same as lab_data_dual_annealing_threshold, so it can take in simulations/lab data, but runs in c++


def train_random_forest(distribution, spins)

This function trains a random forest machine learning model that can learn to classify trial points as either starting in the upper state or the lower state. Before running this function, you should first call the create_multiple_distributions function. If you want multiple different integration times to be trained go ahead, if not, still call that function but do so with a range of consecutive numbers (ex: create_multiple_distributions(1, 2, 10000) will create a distribution for integration time 1). Do NOT call the create_distribution function. Next, prepare the data by calling the prepare_data function with the two lists returned by create_multiple_distributions as well as whatever low, high integration times you made the distributions with. This will return 4 different lists; the training data, training labels, testing data, and testing labels. If you want to save this data (good thing to do to save time) call the save_data function with these 4 lists as parameters, and this function will save the lists into text files. Use the read_data functions to recover these lists from the text files in the future. You are now ready to run the train_random_forest function, with the training data as the distribution parameter and the training labels as the spins parameter. The output of this function will be a trained model, so use model.predict() to predict the labels (the starting qubit state, 0 or 1) of the testing data returned by prepare_data and then compare the predicted labels with the testing labels. For clarification look at the example below, which trains only on an integration time of 5. 
Example:
upper_dist, lower_dist = create_multiple_distributions(5, 6, 100000) 
training_data, training_labels, testing_data, testing_labels = prepare_data(upper_dist, lower_dist, 5, 6)
save_data(training_data, training_labels, testing_data, testing_labels)
model = train_random_forest(training_data, training_labels)
rf_predictions = model.predict(testing_data)
score = accuracy_score(rf_predictions, testing_labels)
Output of print(score): 0.8699
This output means that for an integration time of 5, the model correctly classified the trial correctly 87% of the time. 


def DA_threshold_it(T1, upper_simulations, lower_simulations, threshold_guess):


Finds the best integration time and best threshold for the given T1. Does this by using the dual annealing algorithm with two varying parameters, calling advanced_calculate_error() and optimizing it. Takes in as parameters a best guess for the threshold as well as T1 (typical time for qubit to decay from upper to lower state). Returns the optimization result represented as a OptimizeResult object, Important attributes are: x the solution array (x[0] js threshold, x[1] is integration time), fun the value of the function at the solution (the fidelity, and message which describes the cause of the termination. Search up OptimizeResult for a description of other attributes. Takes in T1 (The typical time that it takes for a qubit to go from upper to lower state) and your best guess for a threshold (any number, 0 is fine). Takes about 800 seconds to run, so recommend you use the C++ version if possible.  Can also take in lab data (upper_simulations and lower_simulations would be your 2d lab trial arrays containing an array of arrays of time series representing the life cycle of the qubit lab)
Example: 
upper_simulations, lower_simulations = noise_simulations(8000, 15, 10)  
result = DA_threshold_it(1, 5)
print(result)
Output on next page


Output of print(result): 
fun: 0.11500000208616257
 message: ['Maximum number of iteration reached']
    nfev: 4031
    nhev: 0
     nit: 1000
    njev: 10
  status: 0
 success: True
       x: array([-8.153239  ,  0.40447468])


def DA_threshold_it_cpp(T1, upper_simulations, lower_simulations, threshold_guess) 

The same as DA_threshold_it, but runs in C++. This makes it more than 3x faster. It is also significantly more accurate, since its increased speed allows it to run more iterations without taking too long.  Remember to run g++ -fPIC -shared -o fasterTest.so faster.cpp on your terminal before running this, else it will not work. Can also take in lab data (upper_simulations and lower_simulations would be your 2d lab trial arrays containing an array of arrays of time series representing the life cycle of the qubit lab)



def find_T1_to_optimal_thresh_and_it():

Note: Takes several hours,  returns arrays that can be used for plotting. This function runs DA_threshold_it several times for a multitude of different integration times. It then returns the thresholds, integration times, and fidelities that were found for each subsequent T1. To see how they fared, plot any of those 3 first arrays returned with the array of T1's (the last, fourth array returned). This way you can see how each of those variables changed as the T1 variable changed. Remember, the 1st value of the T1 array accounts for the same trial of the first value of the threshold array (or it/fidelity ones), the 2nd value of T1 array accounts for the same trial of the 2nd value of the other arrays, and so on. It saves all these values to some text files on your computer, so you don't have to repeat after running the function for hours.


def parallel_find_T1_to_optimal_thresh_and_it():

The same as find_T1_to_optimal_thresh_and_it but makes use of multiprocessing


def find_T1_to_optimal_thresh_and_it_cpp():

The same as find_T1_to_optimal_thresh_and_it but makes use of multiprocessing and C++, making it by far the fastest and best choice. Remember to run g++ -fPIC -shared -o fasterTest.so faster.cpp on your terminal before running this, else it will not work.


def read_T1_to_optimal_thresh_and_it_data():

Returns the saved arrays from either of the the T1_to_optimal_thresh_and_it_cpp functions by grabbing them from text files they were saved in. Use this whenever you want to get a previous result of T1_to_optimal_thresh_and_it_cpp without having to run it all over again 
Example:
thresholds, int_times, fidelities, t1s = read_T1_to_optimal_thresh_and_it_data()


def plot_T1_to_optimal_thresh_and_it(thresholds, i_times, fidelity_list, t1s):

Takes in the result returned by any of the T1_to_optimal_thresh_and_it functions (the 4 arrays returned by those functions) and creates 3 plots of threshold, integration times, and fidelity with respect to T1.  



