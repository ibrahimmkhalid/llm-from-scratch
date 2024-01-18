# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)


# %%
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.show()
