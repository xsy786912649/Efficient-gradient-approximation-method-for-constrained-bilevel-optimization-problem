import cvxpy as cp
import numpy as np


# dataset = diabete
# feature = 8
# number = 768

data_list=[]

f = open("diabete.txt",encoding = "utf-8")
a_list=f.readlines()
f.close()
for line in a_list:
    line1=line.replace('\n', '')
    line2=list(line1.split(' '))
    y=float(line2[0])
    x= [float(line2[i].split(':')[1]) for i in (1,2,3,4,5,6,7,8)]
    data_list.append(x+[y])

data_array=np.array(data_list)
print(data_array.shape)
np.random.seed(100)
np.random.shuffle(data_array)

z_train=data_array[:400, :8]
y_train=data_array[:400, -1]
z_val=data_array[400:580, :8]
y_val=data_array[400:580, -1]
z_test=data_array[580:, :8]
y_test=data_array[580:, -1]

c_array= np.ones_like(y_train)

print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
print(c_array.shape)

feature=8
w = cp.Variable(feature)
b = cp.Variable()
xi = cp.Variable(y_train.shape[0])
lambd = cp.Parameter(y_train.shape[0],nonneg=True)
loss =  0.5*cp.norm(w, 2)**2 + 0.5 * (cp.scalar_product(lambd, cp.power(xi,2)))

# Create two constraints.
constraints=[]
constraints_value=[]
for i in range(y_train.shape[0]):
    constraints.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i])+b) <= 0)
    constraints_value.append(1 - xi[i] - y_train[i] * (cp.scalar_product(w, z_train[i])+b) )

# Form objective.
obj = cp.Minimize(loss)

# Form and solve problem.
prob = cp.Problem(obj, constraints)

C=10
lambd.value=c_array*C
prob.solve(solver='ECOS', abstol=1e-20,reltol=1e-10,max_iters=10000000, warm_start=True)


dual_variables= np.array([ constraints[i].dual_value for i in range(len(constraints))])
constraints_value_1= np.array([ constraints_value[i].value for i in range(len(constraints))])
#print("dual variables", dual_variables)
#print("constraints_value ", constraints_value_1)
print("w value:", (w.value))
print("b value:", (b.value))
#print("xi value:", (xi.value))

number_equal=0
for i in range(len(y_train)):
    if constraints_value_1[i]<-0.0001:
        number_equal=number_equal+1
print(number_equal)

number_equal=0
for i in range(len(y_train)):
    if dual_variables[i]>0.0001*C:
        number_equal=number_equal+1
print(number_equal)

print("value:",(obj.value))

number_right=0
for i in range(len(y_train)):
    q=y_train[i] * (cp.scalar_product(w, z_train[i])+b)
    if q.value>0:
        number_right=number_right+1
print(number_right/len(y_train))

number_right=0
for i in range(len(y_test)):
    q=y_test[i] * (cp.scalar_product(w, z_test[i])+b)
    if q.value>0:
        number_right=number_right+1
print(number_right/len(y_test))

