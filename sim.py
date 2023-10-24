import numpy as np
from toric_code import ToricCode
from ca import CA, Center
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def setup(L, Q, U, fC, fN):
    assert L % Q == 0
    d = int(np.log(L) / np.log(Q)) - 1 # 0: no colonies, 1: colonies, 2: supercolonies, ...
    c = int(np.floor(Q/2)) # linear center idx of colony

    # 1.Generate CAs
    CAs = np.zeros((L,L), dtype=object)
    for i in range(L):
        for j in range(L):
            addr = (i,j) # absolute address
            if (i-c)%Q == 0 and (j-c)%Q == 0: # center cell
                CAs[i,j] = Center(addr, Q, U, d, fC, fN)
            else:
                CAs[i,j] = CA(addr, Q, U, d)

    # 2. Assign neighbors
    for i in range(L):
        for j in range(L):
            neighbors = []
            for i_ in [-1,0,1]:
                for j_ in [-1,0,1]:
                    if (i_,j_) == (0,0): continue
                    neighbors.append( CAs[(i+i_)%L][(j+j_)%L] )
            CAs[i,j].neighbors = neighbors

    return CAs

def global_step(CAs, toric_code):

    L = toric_code.L

    def flip(addr, flip_dir):
        """Flip qubit in `toric_code` located at `flip_dir` relative to stabilizer `addr`"""
        r, c = addr
        r_, c_, d = {'W': (0,0,0), 'N': (0,0,1), 'E': (0,1,0), 'S': (1,0,1)}[flip_dir]
        toric_code.qubits[(r+r_)%L,(c+c_)%L,d] ^= 1
        toric_code.update_stabs()

    # 1. store syndromes
    for i in range(L):
        for j in range(L):
            syndrome = toric_code.syndrome(i,j)
            CAs[i,j].syndrome = syndrome

    # 2. copy data
    for i in range(L):
        for j in range(L):
            CAs[i,j].acquire()

    # 3. update CA, synchronized
    for i in range(L):
        for j in range(L):
            CAs[i,j].update()

    # 4. local rule
    for i in range(L):
        for j in range(L):
            flip_dir = CAs[i,j].rule()
            if flip_dir:
                addr = (i,j)
                flip(addr, flip_dir)

    # 5. coordinated flip between colonies of all levels
    for i in range(L):
        for j in range(L):
            flip_dirs = CAs[i,j].flip()
            for flip_dir in flip_dirs:
                addr = (i,j)
                flip(addr, flip_dir)

# model params
L = 27
Q = 3
p = 0.1

fC = 0.7
fN = 0.2
U = 4 # Note U > Q

t = ToricCode(L)
CAs = setup(L,Q,U,fC,fN)

fig, ax = plt.subplots(figsize=(L,L))
ax.set(xlim=[0,L], ylim=[0,L])
ax.invert_yaxis()

# plot grid
for i in range(L+1):
    plt.axvline(i)
    plt.axhline(i)
    if i % Q == 0: # colony bounds
        plt.axvline(i, c='tab:blue', lw=3)
        plt.axhline(i, c='tab:blue', lw=3)
    if i % Q**2 == 0: # colony bounds
        plt.axvline(i, c='tab:blue', lw=6)
        plt.axhline(i, c='tab:blue', lw=6)
        
inact_qbs = plt.plot([], [], 'o', ms=15, markeredgecolor="tab:blue", markerfacecolor="w")[0]
act_qbs = plt.plot([], [], 'o', ms=15, markeredgecolor="tab:blue", markerfacecolor="tab:blue")[0]
inact_stabs = ax.plot([], [], 'o', ms=15, markeredgecolor="tab:orange", markerfacecolor="w")[0]
act_stabs = ax.plot([], [], 'o', ms=15, markeredgecolor="tab:orange", markerfacecolor="tab:orange")[0]
act_cors = ax.plot([], [], 'o', ms=7, markeredgecolor="tab:red", markerfacecolor="tab:red")[0]

flip_sigs1 = ax.plot([], [], '*', ms=7, markeredgecolor="tab:red", markerfacecolor="tab:red")[0]
flip_sigs2 = ax.plot([], [], '*', ms=7, markeredgecolor="tab:pink", markerfacecolor="tab:pink")[0]

def init():
    
    t.__init__(L)
    
    # remove any filling
    for grid in [act_stabs, act_qbs, act_cors, flip_sigs1, flip_sigs2]:
        grid.set_data([],[])
    
    inact_stabs.set_xdata([i+0.5 for i in range(L) for _ in range(L)])
    inact_stabs.set_ydata([i+0.5 for _ in range(L) for i in range(L)])
    
    inact_qbs.set_xdata([0.5*i for i in range(2*L+1) for _ in range(L+1)])
    inact_qbs.set_ydata([(i % 2)*0.5 + j for i in range(1,2*L+2) for j in range(L+1)])
    
    ax.set_facecolor('w')
    return (act_stabs, act_qbs, act_cors, inact_stabs, inact_qbs, flip_sigs1, flip_sigs2)


def update(i):
    
    plt.title(f'age: {i}, U={U}, Q={Q}, d={int(np.log(L) / np.log(Q)) - 1}')
    
    if i == 0:
        return init()
    if i == 1:
        t.step(p)
    
    # t.step(p)
    pad_widths = [(0,1),(0,1),(0,0)]
    qubits = np.pad(t.qubits, pad_widths, 'wrap')

    # stabs
    coords = np.argwhere(t.stabs == 1)
    act_stabs.set_xdata([c[1]+0.5 for c in coords])
    act_stabs.set_ydata([c[0]+0.5 for c in coords])

    # qubits
    coords = np.argwhere(qubits == 1)
    act_qbs.set_xdata([c[1]+0.5*c[2] for c in coords])
    act_qbs.set_ydata([c[0]+0.5*(1-c[2]) for c in coords])

    # CA update
    global_step(CAs, t)

    # check for logical error (MUST CHANGE: ChECK VIA MWPM OF COPY)
    if np.all(t.stabs == 0):
        if t.check_log_err():
            ax.set_facecolor('#e2c7b0')

    # flip signal 1
    X,Y = [], []
    for i in range(L):
        for j in range(L):
            for n in np.where( CAs[i,j].flip_sig[0] )[0]:
                X.append( j + 0.5 + [0,-0.5,0.5,0][n] ) # N,W,E,S
                Y.append( i + 0.5 + [-0.5,0,0,0.5][n] ) # N,W,E,S (flipped yaxis)
    flip_sigs1.set_data(X,Y)
    
    # flip signal 2
    X,Y = [], []
    for i in range(L):
        for j in range(L):
            for n in np.where( CAs[i,j].flip_sig[1] )[0]:
                X.append( j + 0.5 + [0,-0.5,0.5,0][n] ) # N,W,E,S
                Y.append( i + 0.5 + [-0.5,0,0,0.5][n] ) # N,W,E,S (flipped yaxis)
    flip_sigs2.set_data(X,Y)

    # correction
    qubits_ = np.pad(t.qubits, pad_widths, 'wrap')
    cors = qubits ^ qubits_
    coords = np.argwhere(cors == 1)
    act_cors.set_xdata([c[1]+0.5*c[2] for c in coords])
    act_cors.set_ydata([c[0]+0.5*(1-c[2]) for c in coords])
        
    return (act_stabs, act_qbs, act_cors, inact_stabs, inact_qbs, flip_sigs1, flip_sigs2)


ani = animation.FuncAnimation(fig, update, init_func=init, frames=50, blit=True)
writervideo = animation.FFMpegWriter(fps=2)
ani.save(f'CA_toric_code_L={L}_Q={Q}_U={U}_p={p}.mp4', writer=writervideo)
