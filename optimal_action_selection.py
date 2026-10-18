import numpy as np
import matplotlib.pyplot as plt

q_star = [0.2, -0.8, 1.5, 0.4, 1.3, -1.5, -0.15, -1, 0.8, -0.4]
sigma = 1
n_runs = 2000
n_steps = 1000
a_opt = int(np.argmax(q_star))  

#-------------------------------greedy-----------------------------------------
def greedy(n_steps, n_runs):
    sum_rewards = np.zeros(n_steps)
    sum_opt = np.zeros(n_steps)              
    for run in range(n_runs):
        Q = np.zeros(10)
        N = np.zeros(10)
        rewards = np.zeros(n_steps)
        optimal = np.zeros(n_steps)          
        for t in range(n_steps):
            arm = np.argmax(Q)
            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
            optimal[t] = (arm == a_opt)
        sum_rewards += rewards
        sum_opt += optimal
    total_avg = sum_rewards / n_runs
    pct_opt = (sum_opt / n_runs) * 100 
    return total_avg, pct_opt

#-------------------------------epsilon greedy---------------------------------
def epsilon_greedy(n_steps, n_runs, epsilon):
    sum_rewards = np.zeros(n_steps)
    sum_opt = np.zeros(n_steps)
    for run in range(n_runs):
        Q = np.zeros(10); N = np.zeros(10)
        rewards = np.zeros(n_steps)
        optimal = np.zeros(n_steps)
        for t in range(n_steps):
            if np.random.rand() < epsilon:
                arm = np.random.randint(0, 10)
            else:
                arm = np.argmax(Q)
            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
            optimal[t] = (arm == a_opt) * 1.0
        sum_rewards += rewards
        sum_opt += optimal
    total_avg = sum_rewards / n_runs
    pct_opt = (sum_opt / n_runs) * 100.0
    return total_avg, pct_opt

#-------------------------------soft max---------------------------------------
def soft_max(n_steps, n_runs, T):
    sum_rewards = np.zeros(n_steps)
    sum_opt = np.zeros(n_steps)
    for run in range(n_runs):
        Q = np.zeros(10); N = np.zeros(10)
        rewards = np.zeros(n_steps)
        optimal = np.zeros(n_steps)
        for t in range(n_steps):
            exp_values = np.exp(Q / T)
            prob = exp_values / np.sum(exp_values)
            arm = np.random.choice(np.arange(10), p=prob)
            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
            optimal[t] = (arm == a_opt) * 1.0
        sum_rewards += rewards
        sum_opt += optimal
    total_avg = sum_rewards / n_runs
    pct_opt = (sum_opt / n_runs) * 100.0
    return total_avg, pct_opt


#-------------------------------plot-------------------------------------------
avg_greedy, opt_greedy = greedy(n_steps, n_runs)
avg_e01,   opt_e01    = epsilon_greedy(n_steps, n_runs, epsilon=0.1)
avg_e001,  opt_e001   = epsilon_greedy(n_steps, n_runs, epsilon=0.01)
avg_sm1,   opt_sm1    = soft_max(n_steps, n_runs, T=1.0)
#avg_sm_2,   opt_sm_2    = soft_max(n_steps, n_runs, T=0.2)


steps = np.arange(n_steps)
plt.figure(figsize=(8,5))
plt.plot(steps, opt_greedy, label='Greedy (ε=0)')
plt.plot(steps, opt_e01,   label='ε=0.1')
plt.plot(steps, opt_e001,  label='ε=0.01')
plt.plot(steps, opt_sm1,   label='Softmax (T=1)')
#plt.plot(steps, opt_sm_2,   label='Softmax (T=0.2)')

plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.title('Percentage of Optimal Action Selection vs Steps')
plt.legend()
plt.grid(True)
plt.show()
