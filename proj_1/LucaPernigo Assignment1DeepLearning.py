#1


import numpy as np
import matplotlib . pyplot as plt
def plot_polynomial( coeffs , z_range , color='b' ):
    z= np.linspace(z_range[0], z_range[1], 100)
    y=np.polynomial.polynomial.polyval(z, coeffs)
    plt.plot(z, y, color)

w=np.array([0, -5, 2, 1, 0.05])
z=(-3,3)



plot_polynomial(w, z)
plt.title("Polynomial")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#2


def create_dataset(w, z_range , sample_size , sigma, seed=42):
  random_state = np.random.RandomState(seed)
  z = random_state.uniform(z_range[0], z_range[1], (sample_size)) 
  x = np.zeros((sample_size , w.shape[0]))
  for i in range(sample_size):
    for j in range(5):
      x[i,j]= z[i]**j
  
  y = x.dot(w) 
  if sigma > 0:
    y += random_state.normal (0.0 , sigma , sample_size ) 
  return x, y



#3


training=create_dataset(w,z,500,0.5, seed=0)
#10
#training=create_dataset(w,z,10,0.5, seed=0) 
validation=create_dataset(w,z,500,0.5, seed=1)



#4


#Training
x=training[0][:,1]
y=training[1]
plt.scatter(x,y,color='b', alpha=0.2) 
plt.title("Training set")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


#Validation
x=validation[0][:,1]
y=validation[1]
plt.scatter(x,y,color='r', alpha=0.2) 
plt.title("Validation set")
plt.xlabel("x")
plt.ylabel("y")
plt.show()



#5 ->see report


#6

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE=torch.device("cuda:0" if torch.cuda.is_available()
                    else "cpu")
model=nn.Linear(5,1, bias=False)
model=model.to(DEVICE)
loss_fn=nn.MSELoss()

learning_rate=0.00122
optimizer=optim.SGD(model.parameters(), lr=learning_rate)


training_x=np.array(training[0])
training_x=torch.from_numpy(training_x)
training_x=training_x.float()
training_x=training_x.to(DEVICE)


val_x=np.array(validation[0])
val_x=torch.from_numpy(val_x)
val_x=val_x.float()
val_x=val_x.to(DEVICE)

training_y=torch.from_numpy(training[1])
training_y=training_y.float()
training_y=training_y.reshape([500,1])
#10 #training_y=training_y.reshape([10,1])
training_y=training_y.to(DEVICE)

y_val=torch.from_numpy(validation[1])
y_val=y_val.float()
y_val=y_val.reshape([500,1])
y_val=y_val.to(DEVICE)


num_steps=1600
loss_evo=list()
vloss_evo=list()

w0=list()
w1=list()
w2=list()
w3=list()
w4=list()


for step in range(num_steps):
  model.train()
  optimizer.zero_grad()
  
  y_=model(training_x)
  loss=loss_fn(y_, training_y)
  print(f"Step {step}: train loss: {loss}")
  
  loss_evo.append(loss.item())
  
  w0.append(model.weight[0][0].item())
  w1.append(model.weight[0][1].item())
  w2.append(model.weight[0][2].item())
  w3.append(model.weight[0][3].item())
  w4.append(model.weight[0][4].item())
  

  loss.backward()
  optimizer.step()

  model.eval()
  with torch.no_grad():
    y_=model(val_x)
    val_loss=loss_fn(y_, y_val)
    vloss_evo.append(val_loss.item())
  print(f"Step {step}: val loss: {val_loss}") 
  

#7 -> see report


#8


fig, ax= plt.subplots()

ax.set_yscale('log')
ax.plot(range(step+1), loss_evo, color="b")

plt.title("Training Loss vs Steps Log scale")
plt.xlabel("Steps",)
plt.ylabel("Training Loss");
plt.show()

fig, ax= plt.subplots()

ax.set_yscale('log')
ax.plot(range(step+1), vloss_evo, linewidth="1", color="r")

plt.title("Validation Loss vs Steps Log scale")
plt.xlabel("Steps")
plt.ylabel("Validation Loss");
plt.show()

fig, ax= plt.subplots()

ax.set_yscale('log')
plt.title("Losses vs Steps Log scale")
ax.plot(range(step+1), vloss_evo, ":", linewidth = '4', label="Validation Loss", color="r")
ax.plot(range(step+1), loss_evo,"--", linewidth = "2.1", label="Training Loss", color="b")

plt.legend()
plt.show()

#9


model.eval()

with torch.no_grad():
  y_=model(training_x)
  

fig, ax= plt.subplots()


ax.plot(training_x.cpu().numpy()[:,1], training_y.cpu().numpy(), ".", label="training points")

sorted, indices = torch.sort(training_x[:,1], 0)
i=indices.cpu().numpy()
ax.plot(training_x.cpu().numpy()[:,1][i], y_.cpu().numpy()[i], "-",linewidth=3.0, label="estimated polynomial")

plt.title("Polynomial regression")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
plt.show()


#10 -> see report



#11

fig, ax= plt.subplots()

ax.plot(range(step+1), w0, linewidth = '2', label="w0")
ax.plot(range(step+1), w1, linewidth = '2', label="w1")
ax.plot(range(step+1), w2, linewidth = '2', label="w2")
ax.plot(range(step+1), w3, linewidth = '2', label="w3")
ax.plot(range(step+1), w4, linewidth = '2', label="w4")

plt.title("Coefficients evolution")
plt.xlabel("Steps",)
plt.ylabel("Coefficients");

plt.legend()
plt.show()