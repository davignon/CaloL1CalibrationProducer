#!/usr/bin/env python
import jax.numpy as jnp
from jax.scipy import optimize
import numpy as np
from jax import grad, jacobian
import matplotlib.pyplot as plt
import glob

def f(x, y, z, w, v, p):

  ietas_index=x
  ihad = y
  ihad_index = z
  jets = w
  iem = v
  
  SFs=p.reshape(100,40)

  jet_energies = jets[:,3]
  l1_jet_energies = jnp.zeros_like(jet_energies)

  ihad_flat = ihad.flatten()
  ietas_index_flat = ietas_index.flatten()
  ihad_index_flat = ihad_index.flatten()
  SF_for_these_towers_flat = SFs[ietas_index_flat, ihad_index_flat]
  ihad_calib_flat = jnp.multiply(ihad_flat, SF_for_these_towers_flat)
  ihad_calib = ihad_calib_flat.reshape(len(ihad_index),81)
  l1_jet_energies = jnp.sum(ihad_calib[:], axis=1)
  l1_jet_em_energies = jnp.sum(iem[:], axis=1)

  DIFF = jnp.abs((l1_jet_energies + l1_jet_em_energies) - jet_energies)
  #print("sum ihad+iem:",(l1_jet_energies + l1_jet_em_energies))
  #print("jet energy:",jet_energies)
  MAPE = jnp.divide(DIFF, jet_energies)
  STD = jnp.std(MAPE)

  #print(STD)
  return MAPE

def f2(x, y, z, w, p):
    
  return p.sum()

def find_min(x, p):

  x0 = jnp.array([1.0])
  args = (a, b)
  return optimize.minimize(f, x0, (args,), method="BFGS")

eta_binning = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
et_binning = [i for i in range(1,101)]
SFs = jnp.ones(shape=(len(eta_binning),len(et_binning)))
SFs_flat = jnp.array([1. for i in range(0,len(eta_binning)*len(et_binning))])
print(SFs_flat)
#print(type(SFs))
#SFs_flat = SFs.flatten() 
#SFs[10, 42] = 5.
#SFs[11, 14] = 4.

list_towers_files = glob.glob("input_data/towers_*_0.npz")
list_jets_files = glob.glob("input_data/jets_*_0.npz")

towers = jnp.load(list_towers_files[0], allow_pickle=True)['arr_0']
jets = jnp.load(list_jets_files[0], allow_pickle=True)['arr_0']

#for ifile in range(1,len(list_towers_files)):
for ifile in range(1,10):
    #print("adding:",list_towers_files[ifile])
    #print("adding:",list_jets_files[ifile])
    x = jnp.load(list_towers_files[ifile], allow_pickle=True)['arr_0']
    y = jnp.load(list_jets_files[ifile], allow_pickle=True)['arr_0']
    towers = jnp.concatenate([towers,x])
    jets = jnp.concatenate([jets,y])


#towers = jnp.load("input_data/towers_556_0.npz", allow_pickle=True)['arr_0']
#jets = jnp.load("input_data/jets_556_0.npz", allow_pickle=True)['arr_0']


print("Number of events in file file =",len(towers))

ietas = jnp.argmax(towers[:, :, 3: ], axis=2) + 1
ietas_index = jnp.argmax(towers[:, :, 3: ], axis=2)

ihad = towers[:, :, 1]
iem = towers[:, :, 0]
ihad_index = towers[:, :, 1] - 1
ihad_index = ihad_index.at[ihad_index > 99].set(99)
ihad_index = ihad_index.at[ihad_index < 0].set(0)

#[ihad_index > 99] = 99
#ihad_index[ihad_index < 0] = 0

p = SFs_flat

test = f(ietas_index, ihad, ihad_index, jets, iem, p)
#print(test)

nb_epochs=1
lvals=[]
dvals=[]
lr=0.1

for ep in range(nb_epochs):
    print("epoch",ep)
    for i in range(0,len(ihad)):
        if i==len(ihad)-1: break
        if i%100==0: print(i)
        #xi = [ietas_index[i:i+1], ihad[i:i+1], ihad_index[i:i+1], jets[i:i+1]]
        #print("i=%.20f"%(i))
        #calculate the loss
        #print(type(x))
        #print(type(SFs_flat))
        #x=ihad
        jac = jacobian(f,argnums=5)(ietas_index[i:i+1], ihad[i:i+1], ihad_index[i:i+1], jets[i:i+1], iem[i:i+1], SFs_flat)[0]
        #print(len(jac))
        SFs_flat = SFs_flat - lr*jac
        #jac = jacobian(f)(xi, SFs_flat)
        #for ieta in range(0,len(SFs)):
        #    for iet in range(0,len(SFs[ieta])):
        #        a=SFs[ieta][iet]
        #        l=f(xi,SFs)
        #        #lvals.append(l)
        #        #print("l=%.20f"%(l))
        #        #calculate the partial derivative
        #        #d=grad_loss(l, (xi, a))
        #        d = grad(l)(xi, SFs)
        #        print("d=%.20f"%(d))
        #        #dvals.append(d)
        #        #upgrade the parameter
        #        SFs[ieta][iet]=a-lr*d
        #        aguess.append(a)
        #        #print("a=%.20f"%(a))
#for i in range(0,len(SFs_flat)):
#    print(SFs_flat[i])

SFs=SFs_flat.reshape(100,40)
print(SFs)
jnp.save('test.out', SFs)

#sol=optimize.minimize(f,
