import numpy as np

kernel_sizes = [7, 7, 5, 3, 3, 3]
stride_sizes = [4, 2, 2, 2, 2]

print_all = False

for n in np.arange(100,351):
    
    if(print_all): print(f"** N = {n}\n")
    
    spatial_sizes = []
    spatial_sizes.append(n)

    ss1 = (n-3)/4
    spatial_sizes.append(ss1)

    ss2 = (ss1 - kernel_sizes[0])/stride_sizes[0] + 1
    spatial_sizes.append(ss2)

    ss3 = (ss2 - kernel_sizes[1])/stride_sizes[1] + 1
    spatial_sizes.append(ss3)

    ss4 = (ss3 - kernel_sizes[2])/stride_sizes[2] + 1
    spatial_sizes.append(ss4)

    ss5 = (ss4 - kernel_sizes[3])/stride_sizes[3] + 1
    spatial_sizes.append(ss5)

    ss6 = (ss5 - kernel_sizes[4])/stride_sizes[4] + 1
    spatial_sizes.append(ss6)

    works = True
    for o in [n]:
        for i, k in enumerate(kernel_sizes[:-1]):
            if(print_all): print(f"Spatial size {o}, kernel size {k}, stride {stride_sizes[i]}, at i = {i} gives {(o-k)/stride_sizes[i]+1}")
            o = (o - k)/stride_sizes[i] + 1
            if(o != int(o)): works = False
    if(works): print(f"{n} works!")
