import matplotlib.pyplot as plt
import torch

# Given tensor values
values = torch.tensor([0.0000e+00, 2.3283e-10, 4.6566e-10, 9.3132e-10, 7.6289e-06, 7.6292e-06,
                       7.6294e-06, 1.5258e-05, 1.5259e-05, 3.0517e-05, 3.0517e-05, 3.0518e-05,
                       3.8147e-05, 4.5776e-05, 6.1034e-05, 6.1035e-05, 6.1035e-05, 6.1035e-05,
                       6.8664e-05, 7.6293e-05, 9.9994e-01, 9.9995e-01, 9.9996e-01, 9.9997e-01,
                       9.9998e-01, 9.9999e-01, 1.0000e+00, 1.0000e+00])

# Plotting the tensor values
plt.figure(figsize=(10, 5))
plt.plot(values.numpy(), marker='o')
plt.title('Plot of Tensor Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.savefig("Exploring.png")
