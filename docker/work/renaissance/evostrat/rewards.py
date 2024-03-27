from permetrics import RegressionMetric
import numpy as np
import multiprocess as mp
import pandas as pd

def reward_func(weights, mlp, k_names, model, n_samples, eig_partition, n_threads, reward_flag, n_consider):
    def calc_eig(gen_param):
        model.prepare_parameters(gen_param, k_names)
        max_eig = model.calc_eigenvalues()
        return max_eig

    pool = mp.Pool(n_threads)

    # Assuming mlp is accessible or passed as a parameter
    mlp.generator.set_weights(weights)
    gen_params = mlp.sample_parameters()
    gen_params = [[params] for params in gen_params]
    max_eig = []
    for gen_param in gen_params:
        max_eig.append(calc_eig(gen_param))
    # max_eig = pool.map(calc_eig, gen_params)
    max_eig = np.array([this_eig for eig in max_eig for this_eig in eig])

    if reward_flag == 0:
        max_neg_eig = np.min(max_eig)
        if max_neg_eig > eig_partition:
            this_reward = 0.01 / (1 + np.exp(max_neg_eig - eig_partition))
        else:
            this_reward = len(np.where(max_eig <= eig_partition)[0]) / n_samples
    
    elif reward_flag == 1:
        max_eig.sort()
        considered_avg = sum(max_eig[:n_consider]) / n_consider
        this_reward = np.exp(-0.1 * considered_avg) / 2

    pool.close()
    pool.join()
    return this_reward

def time_stamp_matching(N_STEPS, TOTAL_TIME, current_time):
    if current_time > TOTAL_TIME:
        return N_STEPS
    else:
        return int(current_time * (N_STEPS-1) / TOTAL_TIME)

def check_simulated_RMSE(solutions, experimental_data, scaling_factor, N_STEPS = 1000, TOTAL_TIME = 60, LAST_N_POINTS=1):
    """
    Calculate the RMSE for bioreactor simulations against experimental data.

    Parameters:
        solution (list): A list of solutions.
        experimental_data (dict): A dictionary containing experimental data.
        scaling_factor (dict): A dictionary containing scaling factors.
        n_steps (int): The number of steps.
        total_time (float): The total time.
        n_simulations (int): The number of simulations.
        last_n_points (int, optional): The number of last points. Defaults to 0.

    Returns:
        dict: A dictionary containing the full score for each simulation.
    """    
    

    all_solutions = experimental_data.copy()
    total_reward = 0
    for conc, scaling in scaling_factor.items():
        df = all_solutions[conc][['Time', 'mean']].iloc[-LAST_N_POINTS:].copy()
        df.rename(columns={'mean':'experimental'}, inplace=True)
        
        temp_df = pd.DataFrame()
        for i in range(len(solutions)):
            sim_conc = solutions[i].concentrations[conc]

            if sim_conc.shape[0]==N_STEPS:
                new_df = df.copy() 
                new_df['simulated'] = new_df['Time'].apply(lambda x: sim_conc[time_stamp_matching(N_STEPS, TOTAL_TIME, x)]*scaling)
                new_df['sq_err'] = (new_df['experimental'] - new_df['simulated']) ** 2
                new_df['reward'] = nrmse_score(new_df['sq_err'], 0.5)
                temp_df = temp_df.append(new_df, ignore_index=True)  # Append to temp_df
        all_solutions[conc] = temp_df

        total_reward += temp_df['reward'].mean() / 3
    return total_reward

def nrmse_score(errors, scaling_factor):
    nrmse = 1 / (np.exp(errors / scaling_factor))**2
    return nrmse