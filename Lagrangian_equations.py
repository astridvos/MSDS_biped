""" 
In this file, T, V, M and C are calculated symbolically with casadi and then converted to numerical functions.
Also the coordinates and there derivatives are calculated symbolically with casadi and converted to numerical functions.
"""

# imports
import casadi as ca
import numpy as np
import json
import matplotlib.pyplot as plt

# import parameters
with open('parameters.json') as f:
    data = json.load(f)
l1, l2, l3, l4, l5 = data['l1'], data['l2'], data['l3'], data['l4'], data['l5']
r1, r2, r3, r4, r5 = data['r1'], data['r2'], data['r3'], data['r4'], data['r5']
m1, m2, m3, m4, m5 = data['m1'], data['m2'], data['m3'], data['m4'], data['m5']
I1, I2, I3, I4, I5 = data['I1'], data['I2'], data['I3'], data['I4'], data['I5']


# create symbolic variables for the coordinates and there derivatives
q1, q2, q3, q4, q5 = ca.SX.sym('q1'), ca.SX.sym('q2'), ca.SX.sym('q3'), ca.SX.sym('q4'), ca.SX.sym('q5')
q1_d, q2_d, q3_d, q4_d, q5_d = ca.SX.sym('q1_d'), ca.SX.sym('q2_d'), ca.SX.sym('q3_d'), ca.SX.sym('q4_d'), ca.SX.sym('q5_d')
q = ca.vertcat(q1, q2, q3, q4, q5)
q_d = ca.vertcat(q1_d, q2_d, q3_d, q4_d, q5_d)

# coordinates of each link
# link 3
alpha_3 = q[0] + q[4] + q[2] + np.pi
x_G3 = (l3-r3)*ca.sin(-alpha_3)
y_G3 = (l3-r3)*ca.cos(-alpha_3)

x_B = l3*ca.sin(-alpha_3)
y_B = l3*ca.cos(-alpha_3)

# link 1
alpha_1 = np.pi*3/2 - q[0] - q[4]
x_G1 = x_B - (l1-r1)*ca.cos(alpha_1)
y_G1 = y_B + (l1-r1)*ca.sin(alpha_1)

x_C = x_B - l1*ca.cos(alpha_1)
y_C = y_B + l1*ca.sin(alpha_1)

# link 5
x_G5 = x_C + r5*ca.sin(-q[4])
y_G5 = y_C + r5*ca.cos(-q[4])

x_F = x_C + l5*ca.sin(-q[4])
y_F = y_C + l5*ca.cos(-q[4])

# link 2
alpha_2 = q[4] + q[1] - np.pi/2
x_G2 = x_C - r2*ca.cos(alpha_2)
y_G2 = y_C - r2*ca.sin(alpha_2)

x_D = x_C - l2*ca.cos(alpha_2)
y_D = y_C - l2*ca.sin(alpha_2)

# link 4
alpha_4 = q[4] + q[1] + q[3] - np.pi/2
x_G4 = x_D - r4*ca.cos(alpha_4)
y_G4 = y_D - r4*ca.sin(alpha_4)

x_E = x_D - l4*ca.cos(alpha_4)
y_E = y_D - l4*ca.sin(alpha_4)


# Kinetic energy 
# Ifo q_d -> easier to calculate the mass matrix
# chain rule: d/dt(x_G1(q)) = d/dq(x_G1)*d/dt(q) = d/dq(x_G1)*q_d
# ca.jtimes calculates the jacobian of x_G1 with respect to q and multiplies it with q_d
T = 0.5*m1*(ca.jtimes(x_G1, q, q_d)**2 + ca.jtimes(y_G1, q, q_d)**2) + 0.5*I1*ca.mtimes(q_d[0] + q_d[4], q_d[0] + q_d[4]) + \
    0.5*m2*(ca.jtimes(x_G2, q, q_d)**2 + ca.jtimes(y_G2, q, q_d)**2) + 0.5*I2*ca.mtimes(q_d[1] + q_d[4], q_d[1] + q_d[4]) + \
    0.5*m3*(ca.jtimes(x_G3, q, q_d)**2 + ca.jtimes(y_G3, q, q_d)**2) + 0.5*I3*ca.mtimes(q_d[2] + q_d[0] + q_d[4], q_d[2] + q_d[0] + q_d[4]) + \
    0.5*m4*(ca.jtimes(x_G4, q, q_d)**2 + ca.jtimes(y_G4, q, q_d)**2) + 0.5*I4*ca.mtimes(q_d[3] + q_d[1] + q_d[4], q_d[3] + q_d[1] + q_d[4]) + \
    0.5*m5*(ca.jtimes(x_G5, q, q_d)**2 + ca.jtimes(y_G5, q, q_d)**2) + 0.5*I5*ca.mtimes(q_d[4], q_d[4])

# Potential energy
V = m1*9.81*y_G1 + m2*9.81*y_G2 + m3*9.81*y_G3 + m4*9.81*y_G4 + m5*9.81*y_G5

# Lagrangian
L = T - V

# M matrix
M = ca.jacobian(ca.jacobian(T, q_d), q_d)

# C matrix 
C = ca.gradient(L, q) - ca.jtimes(ca.gradient(L, q_d), q, q_d)

# create a function of T, V, M and C to evaluate expressions
T = ca.Function('T', [q, q_d], [T])
V = ca.Function('V', [q], [V])
M = ca.Function('M', [q], [M])
C = ca.Function('C', [q, q_d], [C])
B = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [0,0,0,0]])

# create all differentials of the coordinates
x_G1_d = ca.jacobian(x_G1, q) @ q_d
y_G1_d = ca.jacobian(y_G1, q) @ q_d
x_G2_d = ca.jacobian(x_G2, q) @ q_d
y_G2_d = ca.jacobian(y_G2, q) @ q_d
x_G3_d = ca.jacobian(x_G3, q) @ q_d
y_G3_d = ca.jacobian(y_G3, q) @ q_d
x_G4_d = ca.jacobian(x_G4, q) @ q_d
y_G4_d = ca.jacobian(y_G4, q) @ q_d
x_G5_d = ca.jacobian(x_G5, q) @ q_d
y_G5_d = ca.jacobian(y_G5, q) @ q_d
x_B_d = ca.jacobian(x_B, q) @ q_d
y_B_d = ca.jacobian(y_B, q) @ q_d
x_C_d = ca.jacobian(x_C, q) @ q_d
y_C_d = ca.jacobian(y_C, q) @ q_d
x_D_d = ca.jacobian(x_D, q) @ q_d
y_D_d = ca.jacobian(y_D, q) @ q_d
x_E_d = ca.jacobian(x_E, q) @ q_d
y_E_d = ca.jacobian(y_E, q) @ q_d
x_F_d = ca.jacobian(x_F, q) @ q_d
y_F_d = ca.jacobian(y_F, q) @ q_d

# create a function of all coordinates and there derivatives
x_G1 = ca.Function('x_G1', [q], [x_G1])
y_G1 = ca.Function('y_G1', [q], [y_G1])
x_G2 = ca.Function('x_G2', [q], [x_G2])
y_G2 = ca.Function('y_G2', [q], [y_G2])
x_G3 = ca.Function('x_G3', [q], [x_G3])
y_G3 = ca.Function('y_G3', [q], [y_G3])
x_G4 = ca.Function('x_G4', [q], [x_G4])
y_G4 = ca.Function('y_G4', [q], [y_G4])
x_G5 = ca.Function('x_G5', [q], [x_G5])
y_G5 = ca.Function('y_G5', [q], [y_G5])
x_A = ca.Function('x_A', [q], [x_B])
y_A = ca.Function('y_A', [q], [y_B])
x_B = ca.Function('x_B', [q], [x_B])
y_B = ca.Function('y_B', [q], [y_B])
x_C = ca.Function('x_C', [q], [x_C])
y_C = ca.Function('y_C', [q], [y_C])
x_D = ca.Function('x_D', [q], [x_D])
y_D = ca.Function('y_D', [q], [y_D])
x_E = ca.Function('x_E', [q], [x_E])
y_E = ca.Function('y_E', [q], [y_E])
x_F = ca.Function('x_F', [q], [x_F])
y_F = ca.Function('y_F', [q], [y_F])
x_G1_d = ca.Function('x_G1_d', [q, q_d], [x_G1_d])
y_G1_d = ca.Function('y_G1_d', [q, q_d], [y_G1_d])
x_G2_d = ca.Function('x_G2_d', [q, q_d], [x_G2_d])
y_G2_d = ca.Function('y_G2_d', [q, q_d], [y_G2_d])
x_G3_d = ca.Function('x_G3_d', [q, q_d], [x_G3_d])
y_G3_d = ca.Function('y_G3_d', [q, q_d], [y_G3_d])
x_G4_d = ca.Function('x_G4_d', [q, q_d], [x_G4_d])
y_G4_d = ca.Function('y_G4_d', [q, q_d], [y_G4_d])
x_G5_d = ca.Function('x_G5_d', [q, q_d], [x_G5_d])
y_G5_d = ca.Function('y_G5_d', [q, q_d], [y_G5_d])
x_B_d = ca.Function('x_B_d', [q, q_d], [x_B_d])
y_B_d = ca.Function('y_B_d', [q, q_d], [y_B_d])
x_C_d = ca.Function('x_C_d', [q, q_d], [x_C_d])
y_C_d = ca.Function('y_C_d', [q, q_d], [y_C_d])
x_D_d = ca.Function('x_D_d', [q, q_d], [x_D_d])
y_D_d = ca.Function('y_D_d', [q, q_d], [y_D_d])
x_E_d = ca.Function('x_E_d', [q, q_d], [x_E_d])
y_E_d = ca.Function('y_E_d', [q, q_d], [y_E_d])
x_F_d = ca.Function('x_F_d', [q, q_d], [x_F_d])
y_F_d = ca.Function('y_F_d', [q, q_d], [y_F_d])




