import numpy as np
from numpy.linalg import svd
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import functools

Z = np.diag([1, -1])
X = np.array([[0., 1.], [1., 0.]])


# generates a random quntum state for n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def random(n):
    x = np.random.rand(2 ** n) + 1.j * np.random.rand(2 ** n)
    x = x / np.sqrt(np.vdot(x, x))
    x = np.reshape(x, list(np.repeat(2, n)))
    return x

# generates a random ground state of a classical 1D Ising model with n spins 1/2 as
# a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def Is_GS(n):
    x = np.random.rand(2) + 1.j * np.random.rand(2)
    x = x / np.sqrt(np.vdot(x, x))
    v = np.zeros(2 ** n, dtype=np.complex128)
    v[0] = x[0]
    v[2 ** n - 1] = x[1]
    v = np.reshape(v, list(np.repeat(2, n)))
    return v


# apply one-dimensional quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
# H = -\sum_{i=0}^{n-2} Z_i Z_j - h \sum_{i=0}^{n-1} X_i - h_Z \sum_{i=0}^{n-1} Z_i.
# to a state given by  a rank n tensor of probability amplitudes psi_{0,1,2,\dots,n-1}
def apply_H(psi, n, h, hz):
    sv = np.shape(psi)
    # print(sv)
    psio = np.zeros(sv)
    # print(np.shape(psio))
    op2 = np.transpose(np.tensordot(-Z, Z, axes=((), ())), (0, 2, 1, 3))
    op = -h * X - hz * Z
    for i in range(n - 1):
        psit = np.tensordot(op2, psi, axes=((2, 3), (i, i + 1)))
        psit = np.transpose(np.reshape(psit, (4, 2 ** (i), 2 ** (n - i - 2))), (1, 0, 2))
        psit = np.reshape(psit, sv)
        psio = psio + psit
    for i in range(n):
        psit = np.tensordot(op, psi, axes=((1,), (i,)))
        psit = np.transpose(np.reshape(psit, (2, 2 ** (i), 2 ** (n - i - 1))), (1, 0, 2))
        psit = np.reshape(psit, sv)
        psio = psio + psit
    return psio


# the same as apply_H but psi_{0,1,2,\dots,n-1} reshaped into a vector (required for the ground state computation)
def apply_H_wrap(psi, n, h, hz):
    sv = [2 for i in range(n)]
    # print(sv)
    psir = np.reshape(psi, sv)
    psio = apply_H(psir, n, h, hz)
    return np.reshape(psio, -1)

# compute energy of  the quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
def en_H(psi, n, h, hz):
    psio = apply_H(psi, n, h, hz)
    EV = np.vdot(np.reshape(psi, -1), np.reshape(psio, -1))
    return np.real(EV)


# compute the ground state energy of  the quantum transverse Ising model  (QTIM) with open boundary conditions Hamiltonian
def en_GS(n, h, hz):
    apply_H_hand = lambda x: apply_H_wrap(x, n, h, hz)
    dmx = 2 ** n
    A = LinearOperator((dmx, dmx), matvec=apply_H_hand)
    evals_small, evecs_small = eigsh(A, 1, which='SA')
    return evals_small[0]


n = 7  # the qubit number
h = 3  # the transverse field
hz = 0.01  # the longitudinal fields
psi = random(n)  # the random state
en = en_H(psi, n, h, hz)
enGS = en_GS(n, h, hz)

print("energy of a random state is " + str(en))
print("the ground state energy is " + str(enGS))

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


def to_left_can_first(psi):
    sv = psi.shape
    n = psi.ndim
    psi = np.reshape(psi,(sv[0], np.prod(sv[1:])))
    [u,s,vh] = svd(psi,full_matrices=False)
    s = [x for x in s if x > 10 ** -15]
    D = len(s)
    newu = u[:, :D]
    newvh = vh[:D, :]
    psir = np.diag(s)@newvh
    psir = np.reshape(psir, ((D,)+sv[1:]))

    return newu, psir

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

def bond_dim(psi):
    dim = []
    for x in psi[:-1]:
        bond = np.shape(x)
        dim.append(bond[-1])
    return dim

###########################################################################



###########################################################################

def MCMPS(psi, j):
    n = psi.ndim
    UV = [None] * n
    if j > n-1 or j < 0:
        raise ValueError("You are out of range!")
    elif j == 0:
        return RCMPS(psi)
    elif j == (n-1):
        return LCMPS(psi)
    else:
        UV[0], psi_l = to_left_can_first(psi)

        for i in range(1, j):
            UV[i], psi_l = to_left_can_rest(psi_l)
        psi_r, UV[n - 1] = to_right_can_first(psi_l)

        for i in range(1, n - (j + 1)):
            psi_r, UV[n - 1 - i] = to_right_can_rest(psi_r)

        UV[j] = psi_r

        return UV

Z = np.diag([1, -1])
X = np.array([[0, 1], [1, 0]])


def H_MPO(psi, h, hz):

    H_0 = np.zeros((3, 2, 2))
    H_0[0, :, :] = -h * X - hz * Z
    H_0[1, :, :] = -Z
    H_0[2, :, :] = np.identity(2)

    H_j = np.zeros((3, 3, 2, 2))
    H_j[0, 0, :, :] = np.identity(2)
    H_j[0, 1, :, :] = np.zeros((2, 2))
    H_j[0, 2, :, :] = np.zeros((2, 2))
    H_j[1, 0, :, :] = Z
    H_j[1, 1, :, :] = np.zeros((2, 2))
    H_j[1, 2, :, :] = np.zeros((2, 2))
    H_j[2, 0, :, :] = -h * X - hz * Z
    H_j[2, 1, :, :] = -Z
    H_j[2, 2, :, :] = np.identity(2)

    H_n = np.zeros((3, 2, 2))
    H_n[0, :, :] = np.identity(2)
    H_n[1, :, :] = Z
    H_n[2, :, :] = -h * X - hz * Z

    n = len(psi.shape)

    H = [None] * n
    H[0] = H_0

    for j in range(1, n - 1):
        H[j] = H_j

    H[n-1] = H_n

    return H

def SLH_contraction(L, mps, mpo, s):

    if s <= 0:
        return L

    L = np.einsum("ijk, inm -> mnjk", L, mps[s])
    L = np.einsum("ijkl, knjm -> inml", L, mpo[s])
    L = np.einsum("ijkl, nkl -> ijn", L, mps[s].conj().T)

    return L

def LH_contraction(mps, mpo, s):

    L = np.tensordot(mps[0], mpo[0], axes=[-2, -2])
    L = np.tensordot(L, mps[0].conj().T, axes=[-1, -1])

    for i in range(1, s):
        L = SLH_contraction(L, mps, mpo, i)

    return L

def SRH_contraction(R, mps, mpo, s):

    if s >= (len(mps) - 1):
        return R

    R = np.einsum("ijk, nmi -> nmjk", R, mps[s], optimize='greedy')
    R = np.einsum("ijkl, nkjm -> inml", R, mpo[s], optimize='greedy')
    R = np.einsum("ijkl, lkr -> ijr", R, mps[s].conj().T, optimize='greedy')

    return R

def RH_contraction(mps, mpo, s):

    R = np.tensordot(mps[-1], mpo[-1], axes=[-1, -2])
    R = np.tensordot(R, mps[-1].conj().T, axes=[-1, -2])

    n = len(mps)
    for i in reversed(range(s+1, n-1)):
        R = SRH_contraction(R, mps, mpo, i)

    return R

def H_expval(psi, mpo):

    s = len(psi.shape) // 2

    mps = MCMPS(psi, s)

    L = LH_contraction(mps, mpo, s)
    R = RH_contraction(mps, mpo, s-1)

    expval = np.einsum('ijk, ijk', L, R)

    return expval.real

def HMPS_expval(mps, mpo, s):

    L = LH_contraction(mps, mpo, s)

    if s == 0 or s == len(mps) - 1:
        R = RH_contraction(mps, mpo, s)
    else:
        R = RH_contraction(mps, mpo, s-1)

    expval = np.einsum('ijk, ijk', L, R)

    return expval.real

psi = random(6)
H = H_MPO(psi, 1, 1)

print("Energy of the state from H_expval function: ", H_expval(psi, H))
print("Reference energy: ", en_H(psi, len(psi.shape),1,1))


def DMRG(L, R, mps, mpo, s, vec):
    n = len(mps)

    tensor = np.reshape(vec, mps[s].shape)

    L = LH_contraction(mps, mpo, s)
    R = RH_contraction(mps, mpo, s)

    if s == 0:
        DMRG = np.einsum("il, ljk -> ijk", tensor, R)
        DMRG = np.einsum('kli, lkj -> ij', mpo[s], DMRG)

    elif s == (n-1):
        DMRG = np.einsum('ljk, li -> ijk', L, tensor)
        DMRG = np.einsum("kli, lkj -> ij", DMRG, mpo[s])

    else:
        DMRG = np.einsum('nij, nkl -> lkij', L, tensor)
        DMRG = np.einsum('inmj, mknl -> iklj', DMRG, mpo[s])
        DMRG = np.einsum("nmij, nmk -> jik", DMRG, R)

    return np.reshape(DMRG, -1)


def minE(L, R, mps, H_MPO, s):

    n_A = functools.reduce(lambda x, y: x * y, mps[s].shape)
    apply = lambda x: DMRG(L, R, mps, H_MPO, s, x)
    operator = LinearOperator((n_A, n_A), matvec = apply)
    E, x = eigsh(operator, 1, which='SA')
    x = np.reshape(x, mps[s].shape)
    mps[s] = x

    return mps

def prepare_L(mps, mpo):
    n = len(mps)

    L = LH_contraction(mps, mpo, 0)
    tensor_list = []
    tensor_list.append(1)  # add an element for L to have the same length as MPS
    tensor_list.append(L)

    for i in range(1, n - 1):
        L = SLH_contraction(L, mps, mpo, i)
        tensor_list.append(L)

    return tensor_list

def prepare_R(mps, mpo):
    n = len(mps)

    R = RH_contraction(mps, mpo, n)
    tensor_list = []
    tensor_list.append(R)

    for i in reversed(range(1, n - 1)):
        R = SRH_contraction(R, mps, mpo, i)
        tensor_list.append(R)

    tensor_list.append(1)

    return tensor_list

def move_center(mps, center, new):
    if center < new:
        for i in range(center, new):

            L_shape = mps[i].shape

            A = np.reshape(mps[i], (-1, L_shape[-1]))
            U, S, Vh = np.linalg.svd(A, full_matrices=False )

            mps[i] = np.reshape(U, L_shape)
            SVh = np.diag(S) @ Vh
            mps[i+1] = np.tensordot(SVh, mps[i+1], axes=(-1, 0))
    elif center > new:
        for i in range(center, new, -1):

            R_shape= mps[i].shape
            L_shape = mps[i-1].shape

            A = np.reshape(mps[i], (L_shape[-1], -1))
            U, S, Vh = np.linalg.svd(A, full_matrices=False )

            mps[i] = np.reshape(Vh, R_shape)
            newU = mps[i-1] @ U @ np.diag(S)
            mps[i-1] = np.reshape(newU, L_shape)
    else:
        return mps

    return mps

def minE_total(mps, mpo, reps):
    n = len(mps)

    for j in range(reps):

        R = prepare_R(mps, mpo)
        L = LH_contraction(mps, mpo, 0)

        for i in range(n-1):
            L = SLH_contraction(L, mps, mpo, i - 1)
            mps = minE(L, R[i], mps, mpo, i)
            mps = move_center(mps, i, i + 1)

        L = prepare_L(mps, mpo)
        R = RH_contraction(mps, mpo, n)

        for i in reversed(range(1, n)):
            R = SRH_contraction(R, mps, mpo, i)
            mps = minE(L[i], R, mps, mpo, i)
            mps = move_center(mps, i, i - 1)

    return mps

s = 0
psi = random(6)
H = H_MPO(psi, 1, 1)

mps = MCMPS(psi, s)

enH = en_H(psi, len(psi.shape), 1, 1)
enGS = en_GS(len(psi.shape), 1, 1)

# print("\nReference energies")

print("Hamiltonian energy:\n", enH)
print("\nGround state energy:\n", enGS)

reps = 2
print("\nMPS energy after %d repetitions." % reps)
mps = minE_total(mps, H, reps)
print("Final energy:", HMPS_expval(mps, H, 0))

E_ref = []
E_obtained = []

col = ['b', 'y', 'g', 'r', 'm']

m = 5
for i in range(m):
    n = i + 3
    E_ref.append([])
    E_obtained.append([])

    h_values = np.arange(0, 10, .1)

    for h in h_values:

        psi = random(n)
        mps = MCMPS(psi, s)
        H = H_MPO(psi, h, 0)

        mps = minE_total(mps, H, 1)

        E_ref[i].append(en_GS(n, h, 0))
        E_obtained[i].append(HMPS_expval(mps, H, 0))
        energy = HMPS_expval(mps, H, 0)

    plt.plot(h_values, E_ref[i], lw=6, label=f"n={i + 3} ground state energy")
    plt.plot(h_values, E_obtained[i], '--', color=col[i], label=f"n={i + 3} minimum found")

plt.legend()
plt.show()

h_values = [0, 0.1, 1, 10]
n_values = np.arange(3, 11, 1)

E_ref = []
E_obtained = []

i = 0
for h in h_values:
    E_ref.append([])
    E_obtained.append([])

    for n in n_values:

        psi = random(n)
        mps = MCMPS(psi, s)
        H = H_MPO(psi, h, 0)

        mps = minE_total(mps, H, 1)

        E_ref[i].append(en_GS(n, h, 0))
        E_obtained[i].append(HMPS_expval(mps, H, 0))
        energy = HMPS_expval(mps, H, 0)

    plt.plot(n_values, E_ref[i], '--', label=f"h={h_values[i]} order parameter - referenced energies", color=col[i])
    plt.plot(n_values, E_obtained[i], 'o', label=f"h={h_values[i]} order parameter - obtained energies", color=col[i])

    i += 1

plt.legend()
plt.show()

def SS_expval(mps, s, op):
    m = mps[s]
    mt = m.T.conj()

    if(s == 0):
        m = np.einsum('ij,ik->jk', m, op)
        expval = np.einsum('ij,ij', m, mt)
    elif(s == len(mps)-1):
        m = np.einsum('ij,jk->ik', m, op)
        expval = np.einsum('ij,ji', m, mt)
    else:
        m = np.einsum('ijk,jl->ilk', m, op)
        expval = np.einsum('ijk,kji', m, mt)

    return expval.real

n = 6
hz_values = [10**-4, 10**-3, 10**-2]
psi = random(n)

h_values = np.arange(0, 2, 0.02)

for hz in hz_values:
    ferroparameter = []
    for h in h_values:

        mps = MCMPS(psi, 0)
        H = H_MPO(psi, h, hz)
        mps_en = minE_total(mps, H, 1)
        parameter = 0

        for s in range(n):
            parameter += SS_expval(mps_en, s, Z)
        ferroparameter.append(parameter / n)

    plt.plot(h_values[1:], ferroparameter[1:], label=f"n={n} ferromagnetic order")

    plt.legend()
    plt.show()