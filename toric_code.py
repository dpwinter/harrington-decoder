import numpy as np

class ToricCode:
    """Minimal toric code"""

    def __init__(self, L):
        self.L = L # lattice size
        self.stabs = np.zeros((L,L)).astype(np.int8)
        self.qubits = np.zeros((L,L,2)).astype(np.int8)

    def syndrome(self, i, j):
        """Stabiliser measurement at (i,j)"""
        l = self.qubits[i,j,0]
        u = self.qubits[i,j,1]
        r = self.qubits[i,(j+1)%self.L,0]
        d = self.qubits[(i+1)%self.L,j,1]
        return (l+r+u+d) % 2

    def update_stabs(self):
        """Parity of neighboring qubits"""
        for i in range(self.L):
            for j in range(self.L):
                self.stabs[i,j] = self.syndrome(i,j)

    def apply_errors(self, p):
        """Apply errors to qubits"""
        self.qubits ^= np.random.binomial(n=1, p=p, size=self.qubits.shape).astype(np.int8)

    def check_log_err(self):
        """Logical error when odd horizental or vertical crossings"""
        return self.qubits[0,:,1].sum() % 2 == 1 or self.qubits[:,0,0].sum() % 2 == 1

    def step(self, p):
        self.apply_errors(p)
        self.update_stabs()
