import numpy as np

def bootstrap_pvalue(sample1, sample2):
    
    # Set Random Seed
    np.random.seed(20)
    
    # Assuming you have two paired sample datasets: sample1 and sample2
    # Calculate the observed test statistic for the original data (the mean of differences)
    observed_statistic = np.mean(sample1 - sample2)
    
    # Define the number of bootstrap iterations
    num_bootstrap_samples = 10000
    
    # Initialize an array to store bootstrap sample statistics
    bootstrap_sample_statistics = np.empty(num_bootstrap_samples)
    
    # Perform bootstrap iterations
    for i in range(num_bootstrap_samples):
        # Randomly sample with replacement from the differences between the paired samples
        differences = sample1 - sample2
        bootstrap_differences = np.random.choice(differences, size=len(differences), replace=True)
        
        # Create a new bootstrap sample by adding the bootstrap differences to sample2
        bootstrap_sample = sample2 + bootstrap_differences
        
        # Calculate the test statistic for the bootstrap sample (mean of differences)
        bootstrap_statistic = np.mean(bootstrap_sample - sample1)
        
        # Store the bootstrap sample statistic
        bootstrap_sample_statistics[i] = bootstrap_statistic
        
#     return observed_statistic, bootstrap_sample_statistics
#     Calculate the p-value for a two-tailed test
    extreme_count = np.sum(np.abs(bootstrap_sample_statistics) >= np.abs(observed_statistic))
    p_value = (extreme_count + 1) / (num_bootstrap_samples + 1)  # Adding 1 to include the observed statistic
    
    return round(p_value,4)