import numpy as np

data = np.genfromtxt('sales.csv',delimiter=',')
print(data)

print("'%s'" % int(data[1][0]))

def h(t0, t1, x):
    return t0 + t1 * x

t0 = 0.0
t1 = 0.0
alpha = 0.2
m = len(data)

print('m=%s' % m)

for i in range(190):
    t0_sum = 0.0
    for z in range(m):
        t0_sum += h(t0,t1,data[z][0]) - data[z][1]

    t0_tmp = t0 - alpha / float(m) * t0_sum

    t1_sum = 0.0
    for z in range(m):
        t1_sum += (h(t0,t1,data[z][0]) - data[z][1]) * data[z][0]

    t1_tmp = t1 - alpha / float(m) * t1_sum

    t0 = t0_tmp
    t1 = t1_tmp

print("%.2f + %.2f * x" % (t0, t1))
