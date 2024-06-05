import matplotlib.pyplot as plt
import numpy as np

# Data points for World Bank
world_bank_x = [5, 15, 30, 50, 70, 85, 95]
world_bank_y = [2460.678385, 3439.927946, 4532.167841, 6001.042183, 8424.057124, 11776.1037, 19961.62567]

# Data points for World Inequality Database
wid_x = [25, 70, 94.5, 99.5]
wid_y = [2289.937, 7984.650274, 26782.894, 78339.9649]

# Data point for Oxfam
oxfam_x = [99.95]
oxfam_y = [228993.7]

# Defining the model function
x = np.linspace(0, 100, 500)
y = 100000 * (100 - x) ** (-0.25) - 30000

# Plotting
plt.figure(figsize=(10, 6))

# Plot World Bank data
plt.scatter(world_bank_x, world_bank_y, color='red', marker='x', label='World Bank')
# Plot World Inequality Database data
plt.scatter(wid_x, wid_y, color='orange', marker='x', label='World Inequality Database')
# Plot Oxfam data
plt.scatter(oxfam_x, oxfam_y, color='green', marker='x', label='Oxfam')
# Plot the model curve
plt.plot(x, y, 'b--', label='Model')

plt.title('Distribution of Capital Across Percentiles')
plt.xlabel('Percentile')
plt.ylabel('Capital / USD')
plt.legend()

plt.grid(True)
plt.show()
