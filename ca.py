import numpy as np

def rule(addr, Q, syndrome, syndromes):
    """Decide which qubit N,S,E,W (or None) corresponding
    to CA at `addr` should be flipped as a function of its
    `syndrome` and the 8 `syndromes` of its neighboring CAs
    """

    y,x = Q - 1 - addr[0], addr[1] # x,y coordinates
    c = int(np.floor( Q / 2 )) # center coords x,y=c,c

    NW,N,NE,W,E,SW,S,SE = syndromes # syndromes of neighbor CAs
    Nothing = not any(syndromes) # True if no neighbor syndromes

    # colony center or no syndrome
    if (x,y) == (c,c) or syndrome == 0:
        return None # No flip

    # W border
    ### If self and any left neighbor has syndrome -> left
    if x == 0:
        if W or NW or SW: return 'W'

    # S border
    ### If self and any lower neighbor has syndrome -> down
    if y == 0:
        if S or SW or SE: return 'S'

    # N corridor
    ### If self or self and any lower neighbor has syndrome -> down
    if x == c and y > c:
        if S or SW or SE or Nothing: return 'S'

    # E corridor
    ### If self or self and any left neighbor has syndrome -> left
    if x > c and y == c:
        if W or NW or SW or Nothing: return 'W'

    # S corridor
    ### If self or self and any upper neighbor has syndrome -> up
    if x == c and y < c:
        if N or NE or NW or Nothing: return 'N'

    # W corridor
    ### If self or self and any right neighbor has syndrome -> right
    if x < c and y == c:
        if E or NE or SE or Nothing: return 'E'

    # SW quadrant
    ### If self or self and any upper neighbor has syndrome -> up
    ### Else If self and any lower-right neighbor has syndrome -> right
    if x < c and y < c:
        if N or NE or NW or Nothing: return 'N'
        elif E or SE: return 'E'

    # NW quadrant
    ### If self or self and any right neighbor has syndrome -> right
    ### Else If self and any lower-left neighbor has syndrome -> down
    if x < c and y > c:
        if E or NE or SE or Nothing: return 'E'
        elif S or SW: return 'S'

    # NE quadrant
    ### If self or self and any lower neighbor has syndrome -> down
    ### Else If self and any upper-left neighbor has syndrome -> left
    if x > c and y > c:
        if S or SW or SE or Nothing: return 'S'
        elif W or NW: return 'W'

    # SE quadrant
    ### If self or self and any left neighbor has syndrome -> left
    ### Else If self and any upper-right neighbor has syndrome -> up
    if x > c and y < c:
        if W or NW or SW or Nothing: return 'W'
        elif N or NE: return 'N'

    return None # No flip

class CA:

    def __init__(self, addr, Q, U, d):

        self.age = 0 # current time step
        self.neighbors = [] # List of neighbor CA instances
        self.addr = (addr[0] % Q, addr[1] % Q) # rel. address within colony
        self.Q = Q # linear size of colony
        self.d = d # total hierarchy depth >= 1
        self.syndrome = 0 # self syndrome
        self.syndromes = np.zeros(8).astype(np.int8) # syndrome of 8 neighbors

        if d > 0: # 2 fields (count, flip) per hierarchy level
            self.U = [U**i for i in range(1,d+1)] # hierarchy levels working period
            # signal to communicate occurrences of neighbor CA's syndromes in same hierarchy
            self.count_sig = np.zeros((d,8)).astype(np.int8)
            self.n_count_sig = np.zeros((d,8)).astype(np.int8) # temp. storage
            # signal to communicate synchronized bit-flips at each hierarchy level
            self.flip_sig = np.zeros((d,4)).astype(np.int8)
            self.n_flip_sig = np.zeros((d,4)).astype(np.int8) # temp. storage

    def acquire(self):
        """Acquire data from neighbors; store in temp n_*_sig"""

        self.syndromes = [n.syndrome for n in self.neighbors]

        NWES_neighbors = [self.neighbors[i] for i in [1,3,4,6]]
        for d in range(self.d): # signals at each hierarchy level
            flip_sigs = [n.flip_sig[d] for n in NWES_neighbors]
            count_sigs = [n.count_sig[d] for n in self.neighbors]
            self.n_count_sig[d] = np.diag(count_sigs[::-1]) # order: propagation direction
            self.n_flip_sig[d] = np.diag(flip_sigs[::-1]) # order: propagation direction

    def update(self):
        """Move from temp to real storage"""
        self.age += 1
        self.count_sig = np.copy(self.n_count_sig) # `copy` important, otherwise will equate memory locations
        self.flip_sig = np.copy(self.n_flip_sig)

    def flip(self):
        """Flip directions at current time step for all levels"""
        flips = []
        for d in range(self.d):
            if self.age % self.U[d] == self.Q**(d+1):
                flips += [k for k,i in zip(['N','W','E','S'],self.flip_sig[d]) if i]
                # reset flip signals
                self.flip_sig[d] = np.zeros(4).astype(np.int8)
                self.n_flip_sig[d] = np.zeros(4).astype(np.int8)
        return set(flips)

    def rule(self):
        """Level-0 local rule"""
        return rule(self.addr, self.Q, self.syndrome, self.syndromes)

class Center(CA):
    """Representation of k-colony center"""

    def __init__(self, addr, Q, U, d, fC, fN):

        super().__init__(addr,Q,U,d) # propagate signals not meant for self, absorb/overwrite only at level k.

        def get_k(i):
            """Calculate hierarchy level index for lattice index `i`"""
            for k in range(d+1):
                if not ( (i - np.floor(Q**(k+1)/2)) / Q**(k+1) ).is_integer():
                    break
            return k - 1

        self.k = min(get_k(addr[0]), get_k(addr[1])) # own hierarchy level
        self.kaddr = ( (addr[0] // Q**(self.k+1)) % Q, (addr[1] // Q**(self.k+1)) % Q) # relative address within own hierarchy level

        self.fN = fN # threshold for neighbor Center syndromes
        self.fC = fC # threshold for own syndrome

        # count only events at own hierarchy level
        self.b = int(np.sqrt(self.U[self.k])) # counter interval for own hierarchy level
        self.counts = np.zeros((2,8)).astype(int) # neighbor counts
        self.count = np.zeros(2).astype(int) # self count

    def update(self):

        self.age += 1
        self.count_sig = self.n_count_sig

        self.count_sig[self.k][:] = self.syndrome # broadcast center syndrome
        self.flip_sig[self.k+1:] = self.n_flip_sig[self.k+1:] # let higher-level flip signals pass

        # update first counter field
        self.counts[0,:] += self.n_count_sig[self.k][::-1] # order: direction signals came from
        self.count[0] += self.syndrome

        # update second counter field, reset first field
        if self.age % self.b == 0:
            self.count[1] += int(self.count[0] >= self.fC * self.b)
            self.counts[1,:] += (self.counts[0,:] >= self.fN * self.b).astype(int)
            self.count[0] = 0
            self.counts[0,:] = 0

    def rule(self):
        """Level-k local rule"""

        if self.age % self.U[self.k] == 0:

            syndromes = (self.counts[1,:] >= self.fN * self.b).astype(np.int8)
            syndrome = int(self.count[1] >= self.fC * self.b)
            flip_dir = rule(self.kaddr, self.Q, syndrome, syndromes)

            if flip_dir:
                self.flip_sig[self.k][['N','W','E','S'].index(flip_dir)] = 1

            self.count[1] = 0 # reset counters at end of work period
            self.counts[1,:] = 0

        return super().rule() # execute level-0 local rule
