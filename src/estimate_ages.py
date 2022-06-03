from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
This file contains a set of functions used to estimate a continuous
value for the age of a subject based on a distribution of categorical
data.

The values returned by the function are inherently estimates since
they are being made based on 'aggregate' categories of ages. The
goal of these estimates is to provide a distribution of ages that
more closely resembles actual subject ages than the 6 'buckets'
provided in the initial dataset.

Note that this function never outputs an individual subject age that
is outside of the initial 'bucket' the dataset provides. For example,
if an individual is in the '36 - 45' age 'bucket', this function will
return an age within this range, under all circumstances. The probability
of an age of 36 is higher than an age of 45. This can be seen in the EDA
notebook and is based on a histogram of age 'buckets'. See functions
below for those calculations.

Parameters:
-----------
df : DataFrame containing categorical 'Subject Age Group' column

Returns:
--------
ages : Series containing estimates for Subject Ages

"""


"""
CONSTANT years_in_category
Simple dictionary providing a conversion from *text* ranges to
*integer* ranges in the form of a two-item list formatted
as [start_age, end_age]. Note that the first age group (1-17)
and the last age group (56 and above) are adjusted to start at
10 and end at 70, respectively.

For the young age group, this is because
a 2-year old is not likely to be stopped. Similarly, an 8-year old
is unlikely. 10 was determined to be a reasonable cutoff. The
function does not return any ages lower than 10.

In a similar fashion, for the old age group, no ages are estimated
above 70. This is based on the histogram of age 'bucket' and the
reasonable analysis of the trend that the number of stops
decreases by age.
"""
AGE_STRING_TO_LIST = {
    "1 - 17": [10, 17],
    "18 - 25": [18, 25],
    "26 - 35": [26, 35],
    "36 - 45": [36, 45],
    "46 - 55": [46, 55],
    "56 and Above": [56, 70],
}


# STEP 1
def estimate_age_from_categorical(df):
    """
    This function is the top-level function that calls each of
    the subsequent functions. It also plots the results by showing
    a histogram of the categorical ages overlaying a histogram
    of the continuous estimated ages.

    Parameters:
    -----------
    df : DataFrame containing categorical 'Subject Age Group' column

    Returns:
    --------
    ages : Series containing estimates for Subject Ages
    """

    # Call subfunction to return:
    # 1. Series containing estimated ages, continuous
    # 2. x within range 10:70
    # 3. number of subjects within each age 'bucket' at given x value
    subject_age_group = df["Subject Age Group"]
    ages_estimated, age_groups, age_group_cts = generate_age_estimates(
        subject_age_group
    )

    # Plot estimated ages (continous)
    plt.figure()
    sns.histplot(ages_estimated, bins=60, kde=True, label="Estimated Ages")

    # Plot age 'buckets' as scatter plot, with y representing
    # number of subjects within age 'bucket'
    plt.scatter(age_groups, age_group_cts, c="r", s=10, label="Age Bins")

    # Plot add-ons
    plt.legend()
    plt.title("Subject Ages (Est)")

    # Return age estimates (continuous)
    return ages_estimated


# STEP 2: called from STEP 1
def generate_age_estimates(subject_age_group):
    """
    Loop over each age subject's age, calling predict_age_from_category
    on each one to get an estimated continuous value

    Parameters:
    -----------
    subject_age_group : Series containing the data within the full
                        dataset's 'Subject Age Group' column

    Returns:
    --------
    ages_estimated : Series containing estimated ages, continuous (the
                     primary feature looking to be created by this file)
    age_groups : x within range 10:70, corresponding to every possible age
    age_group_cts : number of subjects within each age group defined above
    """

    # Calculate mean and std deviation of categorical distribution!
    avg, std, age_groups, age_group_cts = calculate_mean_and_std(
        subject_age_group
    )

    # Loop over each age group
    ages_estimated = []
    for age_categorical in subject_age_group:

        # call function to get estimated age from the category
        age_estimated = predict_age_from_category(age_categorical, avg, std)

        # append to list of estimated ages
        ages_estimated.append(int(round(age_estimated, 0)))

    return ages_estimated, age_groups, age_group_cts


# STEP 3: called from STEP 2
def calculate_mean_and_std(subject_age_group):
    """
    Calculate the mean and standard deviation of a normal distribution of
    based on the ages in the original dataset.

    Note that this gives each age within a given 'bucket' is equally
    as common. These values are seen as red dots in the histogram output by
    this file.  For example, for 100 individuals in the age range from 25-35,
    there are exactly 10 of age 25, 10 of age 26, 10 of age 27, etc.

    Parameters:
    -----------
    subject_age_group : Series containing the data within the full
                        dataset's 'Subject Age Group' column

    Returns:
    --------
    avg : average of the normal curve used to estimate subject age
    std : std deviation of the normal curve used to estimate subject age
    ages : list of all included ages (10 to 70)
    their_counts : number of subjects falling within each age (10 to 70)
                   under the assumption noted in this function's description
    """

    # Empty dictionary to store values that will be looped over
    ages_and_their_counts = {}

    # Count the number of subjects within each age 'bucket'
    age_group_counts = subject_age_group.value_counts()

    # Loop over each age 'bucket'
    for age_group in AGE_STRING_TO_LIST.keys():

        # Return the integer range corresponding to the 'bucket's string
        age_range = AGE_STRING_TO_LIST[age_group]

        # First year in age bucket
        start = age_range[0]

        # Last year in age bucket
        stop = age_range[1]

        # Create a list of values ranging from start to stop at intervals of 1
        age_range_list = np.linspace(start, stop, stop - start + 1)

        # Loop over each age within the 'bucket'
        for item in age_range_list:

            # Create dictionary item for each age (10 to 70)
            # and the number of subjects *in that age bucket*
            ages_and_their_counts[item] = int(
                age_group_counts[age_group] / len(age_range_list)
            )

    # extract values from dictionary
    ages = np.array(list(ages_and_their_counts.keys()))
    their_counts = np.array(list(ages_and_their_counts.values()))

    # calculate the mean and std deviation for the distribution
    avg = sum(ages * their_counts) / sum(their_counts)
    std = stdev(ages)
    print("Avg: {}, std: {}".format(avg, std))
    return avg, std, ages, their_counts


# STEP 4: called from step 2
def predict_age_from_category(bucket, avg, std):
    """
    The function performing the functionality core to this file.

    This function estimates the age of one subject based on the
    normal distribution calculated above, with the avg and std dev
    provided as inputs.

    This function extracts an age from that normal distribution under
    the incredibly important caveat that the age is resampled if the
    extracted value is not *within the known age 'bucket'*.

    For example, if an individual is in the '36 - 45' age 'bucket',
    this function will return an age within this range, under all
    circumstances. The probability of an age of 36 is higher than
    an age of 45. This can be seen in the EDA notebook and is
    based on a histogram of age 'buckets'. See functions below for
    those calculations.

    Parameters:
    -----------
    subject_age_group : Series containing the data within the full
                        dataset's 'Subject Age Group' column

    Returns:
    --------
    avg : average of the normal curve used to estimate subject age
    std : std deviation of the normal curve used to estimate subject age
    ages : list of all included ages (10 to 70)
    their_counts : number of subjects falling within each age (10 to 70)
                   under the assumption noted in this function's description
    """

    # Convert string to integers
    year_range = AGE_STRING_TO_LIST[bucket]
    min_age = year_range[0] - 0.5
    max_age = year_range[1] + 0.5

    # Loop until a random number *WITH THE DESIRED STD DEV AND AVG*
    # is found *WITHIN THE PREDETERMINED BUCKET*
    age = 0
    while (age <= min_age) or (age > max_age):
        age = np.random.normal(avg, std)
    return age
