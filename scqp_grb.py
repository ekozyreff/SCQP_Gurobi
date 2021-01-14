# SCQP instance generator
# by Ernee Kozyreff (@ekozyreff)

# This code generates a non-convex quadratic instance and uses Gurobi do solve it.
# Then it generates a piecewise linear approximation of the instance and again uses Gurobi to solve it.

#%%
# Import libraries

import os
import math
import time
import itertools
import numpy as np
import pandas as pd
from gurobipy import *


#%%
# Generate coefficients for SCQP instances

n = 100
m = 10

np.random.seed(0)

d = []
for j in range(n):
    d.append(-np.random.randint(1,6))

#print("d =", d)

c = []
for j in range(n):
    c.append(d[j] * np.random.randint(10,31) / 2)

#print("c =", c)

e = []
for i in range(m):
    e.append([])
    for j in range(n):
        e[i].append(np.random.randint(1,6))

#print("e =", e)

a = []
for i in range(m):
    a.append([])
    for j in range(n):
        a[i].append(e[i][j] * np.random.randint(10,31) / 2)

#print("a =", a)

b = []
for i in range(m):
    b.append(np.floor(0.03 * np.sum(a[i])))

#print("b =", b)


# %%
# Quadratic model

# Create model on Gurobi
model = Model()


# Create x_j variables
x = {}
for j in range(n):
    x[j] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x'+'_'+str(j))
model.update()


# Set objective function
model.setObjective(quicksum(d[j] * x[j] for j in range(n)) + quicksum(c[j] * x[j] * x[j] for j in range(n)))


# Add constraints
for i in range(m):
    model.addConstr(quicksum(e[i][j] * x[j] for j in range(n)) + quicksum(a[i][j] * x[j] * x[j] for j in range(n)) <= b[i], name='c1'+'_'+str(i))
model.update()


# Write model do disk (optional)
model.write("model_1.lp")


# %%
# Optimization of quadratic model

# Set NonConvex parameter so that Gurobi knows it is a qudratica non-convex problem
model.params.NonConvex = 2


# Optimize model
model.optimize()


# Show solution (optional) on the positive x_j variables
for j in range(n):
    if x[j].x > 0:
        print("x_"+str(j), x[j])


# %%
# Generate goefficients for PWL approximation with fixed number of breakpoints

# Number of breakpoints
T = 101


# Create breakpoints for the x_j variables
# The numbers t[k] are values uniformly distributed between 0 and 1
t = []
for k in range(T):
    t.append(k / (T-1))

#print("t =", t)


# Create objective function values for each breakpoint
# The numbers f[j][k] are the values of f_j evaluated at t[k]
f = []
for j in range(n):
    f.append([])
    for k in range(T):
        f[j].append(c[j]*t[k]*t[k] + d[j]*t[k])

#print("f =", f)


# Create constraints values for each breakpoint
# The numbers g[i][j][k] are the values of g_ij evaluated at t[k]
g = []
for i in range(m):
    g.append([])
    for j in range(n):
        g[i].append([])
        for k in range(T):
            g[i][j].append(a[i][j]*t[k]*t[k] + e[i][j]*t[k])

#print("g =", g)


#%%
# PWL approximation with fixed number of breakpoints model

# Create model on Gurobi
model = Model()


# Create x variables (continuous)
# Variable x[j,k] is associated with breakpoint t[k] of variable x_j
x = {}
for j in range(n):
    for k in range(T):
        x[j,k] = model.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='x'+'_'+str(j)+'_'+str(k))
model.update()


# Create y variables (binary)
# Variable y[j,k] is associated with the interval between t[k] and t[k+1] for x_j
y = {}
for j in range(n):
    for k in range(T-1):
        y[j,k] = model.addVar(vtype=GRB.BINARY, name='y'+'_'+str(j)+'_'+str(k))
model.update()


# Set objective function
model.setObjective(quicksum(f[j][k] * x[j,k] for j in range(n) for k in range(T)))


# Add "original" constraints of the qudratic problem using variables x[j,k]
for i in range(m):
    model.addConstr(quicksum(g[i][j][k] * x[j,k] for j in range(n) for k in range(T)) <= b[i], name='c1'+'_'+str(i))
model.update()


# Add constraints \sum_{k=0}^{T-1} x_j_k = 1, for j = 1, ..., n
for j in range(n):
    model.addConstr(quicksum(x[j,k] for k in range(T)) == 1, name='c2'+'_'+str(j))


# Add first part of SOS2 constraints for all variables x_j
for j in range(n):
    model.addConstr(x[j,0] - y[j,0] <= 0, name='c3'+'_'+str(j)+'_'+str(0))
    for k in range(1, T-1):
        model.addConstr(x[j,k] - y[j,k-1] - y[j,k] <= 0, name='c3'+'_'+str(j)+'_'+str(k))
    model.addConstr(x[j,T-1] - y[j,T-2] <= 0, name='c3'+'_'+str(j)+'_'+str(T-1))


# Add second part of SOS2 constraints: \sum_{k=0}^{T-2} y_j_k <= 1, for j = 1, ..., n
for j in range(n):
    model.addConstr(quicksum(y[j,k] for k in range(T-1)) <= 1, name='c4'+'_'+str(j))


# Write model do disk (optional)
model.write("model_2.lp")


# %%
# Optimization of PWL model with fixed number of breakpoints

# Optimize model
model.optimize()


# Show solution on the x[j,k] variables (optional)
for j in range(n):
    for k in range(T):
        if x[j,k].x > 0:
            print("x_"+str(j)+"_"+str(k), x[j,k])


# Show solution on the y[j,k] variables (optional)
for j in range(n):
    for k in range(T-1):
        if y[j,k].x > 0:
            print("y_"+str(j)+"_"+str(k), y[j,k])


# Create a vector x_orig to represent the "original" x_j variables
# and compute their values from the x[j,k] variables in the solution
x_orig = []
for j in range(n):
    x_orig.append(0)
    for k in range(T):
        x_orig[j] += x[j,k].x * t[k]

print("x_orig", x_orig)


# Compute obj_orig, the value of the "original" (quadratic) objective function
# using the original coefficients c_j and d_j
obj_orig = 0
for j in range(n):
    obj_orig += x_orig[j] * d[j] + x_orig[j] * x_orig[j] * c[j]

print("obj_orig", obj_orig)
