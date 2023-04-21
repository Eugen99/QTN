import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
# from generate_full import *
import time

# generates a random quntum state for n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def random(n):
  x = np.random.rand(2**n)+1.j*np.random.rand(2**n)
  x = x/np.sqrt(np.vdot(x,x))
  x = np.reshape(x,list(np.repeat(2,n)))
  return x

# generates a random ground state of a classical 1D Ising model with n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def Is_GS(n):
  x = np.random.rand(2)+1.j*np.random.rand(2)
  x = x/np.sqrt(np.vdot(x,x))
  v = np.zeros(2**n,dtype=np.complex128)
  v[0] = x[0]
  v[2**n-1] = x[1]
  v = np.reshape(v,list(np.repeat(2,n)))
  return v

#generates an expectation value O of an operator o acting at a site s 
#O=psi_{i0,...,i_{s-1},j,i_{s+1},...,i_n}^* o_{j,k} psi_{i0,...,i_{s-1},k,i_{s+1},...,i_n} (Einstein's convention implied) 
def EV_1s(psi,s,op):
   n = len(psi.shape)
   psiop = np.tensordot(op,psi,axes=((1,),(s,)))
   psiop = np.transpose(np.reshape(psiop,(2,2**(s),2**(n-s-1))),(1,0,2))
   EV = np.vdot(np.reshape(psi,-1),np.reshape(psiop,-1))
   return np.real(EV)

#generates an expectation value O of an operator o acting at sites s,s+1 
#O=psi_{i0,...,i_{s-1},j,k,i_{s+2},...,i_n}^* o_{j,k,l,m} psi_{i0,...,i_{s-1},l,m,i_{s+2},...,i_n} (Einstein's convention implied)
def EV_2s(psi,s,op):
   n = len(psi.shape)
   psiop = np.tensordot(op,psi,axes=((2,3),(s,s+1)))
   psiop = np.transpose(np.reshape(psiop,(4,2**(s),2**(n-s-2))),(1,0,2))
   EV = np.vdot(np.reshape(psi,-1),np.reshape(psiop,-1))
   return np.real(EV)

def decompose1(psi, n):
    if n == 1:
        psi = np.reshape(psi, (2 ** (n - 1)))
        U, S, Vh = svd(psi, full_matrices=False)
        return U
    else:
        psi = np.reshape(psi, (psi.shape[0], 2 ** (n - 1)))
        U, S, Vh = svd(psi, full_matrices=False)
        D = S.size()
        psi = np.diag(S)@Vh
        psi = np.reshape(psi, (D, ))
        return U, psir

# performs the first step of a left canonical MPS construction
def to_left_can_first(psi):                         # takes as an argument psi_{i_0,i_1,..,i_{n-1}}
    sv = psi.shape                                  # returns dimensions of psi_{i_0,i_1,...,i_{n-1}} indices as a tuple (d_0,d_1,...,d_{n-1})
    n = psi.ndim                                    # returns psi rank (n)
    psi = np.reshape(psi,(sv[0], np.prod(sv[1:])))  # merges i_1,...,i_{n-1} indices returning a matrix M_{ij} of dimensions d_0 times d_1*....*d_{n-1}
    [u,s,vh] = svd(psi,full_matrices=False)         # performs svd of the matrix
    s = [x for x in s if x > 10 ** -15]             # neglecting a numerical singular values
    D = len(s)                                      # size of the new index created by svd
    newu = u[:, :D]                                 # redefinition of U matrix
    newvh = vh[:D, :]                               # redefinition of Vh matrix
    psir = np.diag(s)@newvh                         # multiplies a matrix of singular values by the right singular vectors
    psir = np.reshape(psir, ((D,)+sv[1:]))          # splits the second index of a matrix of right singular values  indices of the dimensions i_1,..,i_{n-1}
    return newu, psir                               # u here is the first MPS tensor, psir is a tensor which needs to be decomposed in the next step

# Exercise 1

def to_left_can_rest(psi):
    sv = psi.shape
    n = psi.ndim
    psi = np.reshape(psi,(sv[0] * sv[1], np.prod(sv[2:])))
    [u,s,vh] = svd(psi,full_matrices=False)
    s = [x for x in s if x > 10 ** -15]
    D = len(s)
    newu = u[:, :D]
    newvh = vh[:D, :]
    psir = np.diag(s)@newvh
    psir = np.reshape(psir, ((D,)+sv[2:]))
    newu = np.reshape(newu, (sv[0], sv[1], D))
    return newu, psir

def LCMPS(psi):
    n = psi.ndim
    U = [None] * n
    U[0], psir = to_left_can_first(psi)
    for i in range(1,n-1):
        U[i], psir = to_left_can_rest(psir)
    U[n-1] = psir
    return U

# Exercise 2

def bond_dim(psi):
    dim = []
    for x in psi[:-1]:
        bond = np.shape(x)
        dim.append(bond[-1])
    return dim

def to_right_can_first(psi):
    sv = psi.shape
    n = psi.ndim
    psi = np.reshape(psi, (np.prod(sv[:-1]),sv[-1]))
    [u,s,vh] = np.linalg.svd(psi,full_matrices=False)
    s = [x for x in s if x > 10 ** -15]
    D = len(s)
    newu = u[:,:D]
    newvh = vh[:D,:]
    psir = newu@np.diag(s)
    psir = np.reshape(psir,((sv[:-1]+(D,))))
    return psir, newvh

def to_right_can_rest(psi):
    sv = psi.shape
    n = psi.ndim
    psi = np.reshape(psi,(np.prod(sv[:-2]),sv[-1]*sv[-2]))
    [u,s,vh] = np.linalg.svd(psi,full_matrices=False)
    s = [x for x in s if x > 10 ** -15]
    D = len(s)
    newu = u[:, :D]
    newvh = vh[:D, :]
    psir = newu @ np.diag(s)
    psir = np.reshape(psir, ((sv[:-2] + (D,))))
    newvh = np.reshape(newvh,(D,sv[-2],sv[-1]))
    return psir, newvh

def RCMPS(psi):
    n = psi.ndim
    V = [None] * n
    psir, V[n-1] = to_right_can_first(psi)
    for i in range(1,n-1):
        psir, V[n-1-i] = to_right_can_rest(psir)
    V[0] = psir
    return V

# Exercise 3

def check(psi):
    U = decompose(psi)
    contractions = []
    identities = []
    for u in U[:-1]:
        x = np.tensordot(np.conjugate(u), u, ((0, 1),(0,1)))
        contractions.append(x)
    return contractions

# Exercise 4

def norm(psi):
    V = decompose(psi)[-1]
    norm = np.tensordot(np.conjugate(V), V, ((0, 1), (0, 1)))
    return norm

# Exercise 5

def truncate_svd(psi):
  [u,s,vh] = np.linalg.svd(psi)
  s = s[s>1.e-15]
  k = s.size
  psit = u[:,:k]*s*vh[:k,:]
  if np.allclose(psi,psit,atol=1.e-14,rtol=0):
    return k
  else:
    return -1

# Exercise 6

def MCMPS(psi, j):
    n = psi.ndim
    UV = [None] * n
    if j > n-1 or j < 0:
        raise ValueError("You are out of range!")
    else:
        UV[0], psi_l = to_left_can_first(psi)
        if j > 1:
            for i in range(1, j):
                UV[i], psi_l = to_left_can_rest(psi_l)
            psi_r, UV[n - 1] = to_right_can_first(psi_l)
        for i in range(1, n-(j+1)):
            psi_r, UV[n-1-i] = to_right_can_rest(psi_r)
        UV[j] = psi_r
        return UV

def MCMPScheck(psi, j):
    psi_mps = MCMPS(psi, j)
    n = psi.ndim
    norm = [None] * n
    norm[0] = psi_mps[0]@np.transpose(np.conjugate(psi_mps[0]))
    for i in range(1, len(psi) - 1):
        norm[i] = np.einsum('ijk,ijl->kl', psi_mps[i], np.conjugate(psi_mps[i]))
    for i in range(j+1, n-1):
        norm[i] = np.einsum('ijk,ljk->il', psi_mps[i], np.conjugate(psi_mps[i]))
    norm[n-1] = psi[len(psi)-1]@np.conjugate(psi[len(psi)-1])
    return norm

def MCMPSnorm(psi, j = 1):
    psi_mps = MCMPS(psi, j)
    return np.einsum('ijk,ijk', psi_mps[j], np.conjugate(psi_mps[j])).real

# Exercises 7, 8

s_z = [[1, 0], [0, -1]]
Z = np.einsum('ij,kl->ikjl', s_z, s_z)
Z2 = np.einsum('ikab,abjl->ikjl', Z, Z)
s_z2 = np.einsum('ij,jk->ik', s_z, s_z)

def E_1(psi, j, o):
    mps = MCMPS(psi, j)
    ps = mps[j]
    cps = np.conjugate(mps[j])
    o_expval = np.einsum('ikj,kl,ilj', ps, o, cps, optimize='greedy')
    return o_expval.real

def E_2(psi, j, o):
    mps = MCMPS(psi, j)
    ps = mps[j]
    ps1 = mps[j+1]
    cps = np.conjugate(mps[j])
    cps1 = np.conjugate(mps[j+1])
    if j == (len(mps) - 2):
        o_expval = np.einsum('iaj,ibk,acbd,jc,kd', ps, cps, o, ps1, cps1, optimize='greedy')
    else:
        o_expval = np.einsum('iaj,ibk,acbd,jcl,kdl', ps, cps, o, ps1, cps1, optimize='greedy')
    return o_expval.real

m = 7

# 2 random

tab_expval, tab_var, tab = [None] * (m-2), [None] * (m-2), [None] * (m-2)

for i in range(2, m):
    expval, expval2 = 0, 0
    for j in range(100):
        psi = random(2*i)
        expval += E_2(psi, i, Z)
        expval2 += E_2(psi, i, Z2)
    expval2 = expval2 / 100
    expval = expval / 100
    tab_expval[i-2] = expval
    tab_var[i-2] = expval2 - expval ** 2
    tab[i-2] = 2*i

graph1 = plt.plot(tab, tab_expval, color="blue")
plt.show()

graph2 = plt.plot(tab, tab_var, color="blue")
plt.show()

# 1 random

tab_expval1, tab_var1, tab1 = [None] * (m-2), [None] * (m-2), [None] * (m-2)

for i in range(2, m):
    expval, expval2 = 0, 0
    for j in range(100):
        psi = random(2*i)
        expval2 += E_1(psi, i, s_z2)
        expval += E_1(psi, i, s_z)
    expval2 = expval2 / 100
    expval = expval / 100
    tab_expval1[i - 2] = expval
    tab_var1[i - 2] = expval2 - expval ** 2
    tab1[i - 2] = 2 * i

graph3 = plt.plot(tab1, tab_var1, color="red")
plt.show()

graph4 = plt.plot(tab1, tab_expval1, color="red")
plt.show()

# 2 Ising

tab_expvalI, tab_varI, tabI = [None] * (m-2), [None] * (m-2), [None] * (m-2)

for i in range(2, m):
    expval, expval2 = 0, 0
    for j in range(100):
        psi = Is_GS(2*i)
        expval += E_2(psi, i, Z)
        expval2 += E_2(psi, i, Z2)
    expval2 = expval2 / 100
    expval = expval / 100
    tab_expvalI[i - 2] = expval
    tab_varI[i - 2] = expval2 - expval ** 2
    tabI[i - 2] = 2 * i

graph5 = plt.plot(tabI,tab_varI, color="green")
plt.show()

graph6 = plt.plot(tabI,tab_expvalI, color="green")
plt.show()

# 1 Ising

tab_expvalI1, tab_varI1, tabI1 = [None] * (m-2), [None] * (m-2), [None] * (m-2)

for i in range(2, m):
    expval, expval2 = 0, 0
    for j in range(100):
        psi = Is_GS(2*i)
        expval += E_1(psi, i, s_z)
        expval2 += E_1(psi, i, s_z2)
    expval2 = expval2 / 100
    expval = expval / 100
    tab_expvalI1[i - 2] = expval
    tab_varI1[i - 2] = expval2 - expval ** 2
    tabI1[i - 2] = 2 * i

graph7 = plt.plot(tabI1, tab_varI1, color="black")
plt.show()

graph8 = plt.plot(tabI1, tab_expvalI1, color="black")
plt.show()