import numpy as np
import matplotlib.pyplot as plt

#-------------------------------enviroment-------------------------------------
q_star = [0.2, -0.8, 1.5, 0.4, 1.3, -1.5, -0.15, -1, 0.8, -0.4]
sigma = 1
n_runs = 2000
n_steps = 1000
#-------------------------------greedy-----------------------------------------
def greedy(n_steps, n_runs):
    sum_rewards = np.zeros(n_steps)
    for run in range(n_runs):          
        Q = np.zeros(10)
        N = np.zeros(10)
        rewards = np.zeros(n_steps)
        for t in range(n_steps):      
            arm = np.argmax(Q)
            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
        sum_rewards += rewards        
    total_avg =sum_rewards / n_runs
    return total_avg

#-------------------------------epsilon greedy---------------------------------
def epsilon_greedy(n_steps,n_runs, epsilon):
    sum_rewards = np.zeros(n_steps)
    for run in range(n_runs):          
        Q = np.zeros(10)
        N = np.zeros(10)
        rewards = np.zeros(n_steps)
        for t in range(n_steps):      
            if np.random.rand() < epsilon:
                arm = np.random.randint(0, 10)         
            else:
                arm = np.argmax(Q)
            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
        sum_rewards += rewards        
    total_avg =sum_rewards / n_runs
    return total_avg

#-------------------------------soft max---------------------------------------
def soft_max(n_steps, n_runs, T):
    sum_rewards = np.zeros(n_steps)
    for run in range(n_runs):
        Q=np.zeros(10)
        N=np.zeros(10)
        rewards = np.zeros(n_steps)
        for t in range(n_steps):
            exp_values = np.exp(Q/T)
            prob = exp_values / np.sum(exp_values)
            arm = np.random.choice(np.arange(10), p=prob)

            reward = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (reward - Q[arm])
            rewards[t] = reward
        sum_rewards += rewards        
    total_avg =sum_rewards / n_runs
    return total_avg



#-------------------------------UCB (Upper Confidence Bound)-------------------
def ucb(n_steps, n_runs, c=2.0):
    sum_rewards = np.zeros(n_steps)

    for run in range(n_runs):
        Q = np.zeros(10)          
        N = np.zeros(10)          
        rewards = np.zeros(n_steps)

        t = 0
        for a in range(10):
            if t >= n_steps:
                break
            r = np.random.normal(q_star[a], sigma)
            N[a] += 1
            Q[a] += (1 / N[a]) * (r - Q[a])
            rewards[t] = r
            t += 1


        for t in range(t, n_steps):
            ucb_values = Q + c * np.sqrt(np.log(t + 1) / (N + 1e-12))
            arm = np.argmax(ucb_values)

            r = np.random.normal(q_star[arm], sigma)
            N[arm] += 1
            Q[arm] += (1 / N[arm]) * (r - Q[arm])
            rewards[t] = r

        sum_rewards += rewards

    total_avg = sum_rewards / n_runs
    return total_avg


#-------------------------------plot-------------------------------------------
avg_greedy = greedy(n_steps,n_runs)
avg_eps01  = epsilon_greedy(n_steps, n_runs, epsilon=0.1)
avg_eps001 = epsilon_greedy(n_steps, n_runs, epsilon=0.01)
avg_soft_max_1 = soft_max(n_steps, n_runs , T=1) 
avg_ucb_c2 = ucb(n_steps, n_runs, c=2.0)
#avg_soft_max_nim = soft_max(n_steps, n_runs , T=0.5) 
#avg_soft_max_yek_dahom = soft_max(n_steps, n_runs , T=0.1) 

steps = np.arange(n_steps)

plt.figure(figsize=(8,5))
plt.plot(steps, avg_greedy, label='Greedy')
plt.plot(steps, avg_eps01,  label='ε = 0.1')
plt.plot(steps, avg_eps001, label='ε = 0.01')
plt.plot(steps, avg_soft_max_1, label= 'soft max T=1' )
plt.plot(steps, avg_ucb_c2, label='UCB c=2.0')
#plt.plot(steps, avg_soft_max_nim, label= 'soft max T=0.5' )
#plt.plot(steps, avg_soft_max_yek_dahom, label= 'soft max T=0.1' )

plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Steps for Greedy and ε-Greedy and UCB Methods')
plt.legend()
plt.grid(True)
plt.show()

