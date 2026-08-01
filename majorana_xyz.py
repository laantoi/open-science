from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
import numpy as np

__all__ = ["XYZ_Hamiltonian", "rank_f2", "center_check_matrix", "find_extra_center_element", "solve_left_coeffs_f2"]

class XYZ_Hamiltonian:
    def __init__(self, rows:int, cols:int,bc='periodic'):
        """
        :param rows: Number of rows
        :param cols: Number of cols
        :param bc: Boundary conditions. 'open' or 'periodic'.

        Our node indexing is 
            index=x+y⋅rows, 
        where x is the row coordinate and y is the column coordinate.
                index(x,y)
                2(0,1)---3(1,1)
                /     \  /
                0(0,0)---1(1,0)
        """
        self.rows = rows
        self.cols = cols
        self.bc = bc
        return

    def print_lattice_indices(self):
        for y in range(self.rows-1,-1,-1):
            print("   " * y, end="") 
            for x in range(self.cols):
                i = self.idx(x,y)
                print(f"{i:3d}"f"{x:1d}"f"{y:1d}", end=" ")
            print()

    def idx(self, x:int, y:int): 
        return (x % self.cols) + (y % self.rows)* self.cols 
    


    def logical_matrix(self) -> np.ndarray:
        """
            Returns logical operators' check matrix of shape (n_logicals, 2*n_qubits) in [X|Z] form.
        """
        def x_logical_matrix():
            LX = np.zeros((self.cols, self.rows*self.cols), dtype=np.uint8)
            for x in range(self.cols):          # logical index
                for y in range(self.rows):
                    j1 = self.idx(x,y)          # qubits in column x
                    LX[x, j1] = 1
            return LX
        
        def y_logical_matrix():
            LY = np.zeros((self.cols, self.rows*self.cols), dtype=np.uint8)
            for x in range(self.cols):          # logical index
                for y in range(self.rows):
                    j1 = self.idx(x-y,y)        # qubits in column x
                    LY[x, j1] = 1
            return LY  
        
        def z_logical_matrix():
            LZ = np.zeros((self.rows, self.rows*self.cols), dtype=np.uint8)
            for y in range(self.rows):          # logical index
                for x in range(self.cols):
                    j1 = self.idx(x,y)          # qubits in row y
                    LZ[y, j1] = 1
            return LZ  

        Hx = x_logical_matrix()
        Hy = y_logical_matrix()
        Hz = z_logical_matrix()
        return np.block([[Hx, np.zeros_like(Hx)],
                         [Hy, Hy],
                         [np.zeros_like(Hz),Hz]]).astype(np.uint8)


    
    def double_loop_matrix(self) -> np.ndarray:
        """
            Returns double loops' check matrix of shape (n_double_loops, 2*n_qubits) in [X|Z] form.
            For L x L unit cell with odd L this construction is directly the centre of the gauge group. 
            For L x L unit cell with even L this construction lacks one independent generator.
        """
        def X_check_matrix() -> np.ndarray:
            if (self.bc == 'open'):
                c = 1
            elif (self.bc=='periodic'): 
                c = 0           
            HX = np.zeros((self.cols-c, self.rows*self.cols), dtype=np.uint8)
            for r in range(self.cols-c):        # stabiliser index (pair of columns r and r+1)
                x1 = r
                x2 = (r + 1) % (self.cols-c)
                for y in range(self.rows):
                    j1 = self.idx(x1,y)         # qubits in column x1
                    j2 = self.idx(x2,y)         # qubits in column x2
                    HX[r, j1] = 1
                    HX[r, j2] = 1
            return HX

        def Y_check_matrix() -> np.ndarray:
            if (self.bc == 'open'):
                c = 1
            elif (self.bc=='periodic'):
                c = 0           
            HY = np.zeros((self.cols-c, self.rows*self.cols), dtype=np.uint8)
            for r in range(self.cols-c):          # stabiliser index (pair of Y-type columns r and r+1)
                x1 = r
                x2 = (r + 1) % (self.cols-c)
                for y in range(self.rows):
                    j1 = self.idx(x1-y,y)         # qubits in Y-type column x1
                    j2 = self.idx(x2-y,y)         # qubits in Y-type column x2
                    HY[r, j1] = 1
                    HY[r, j2] = 1
            return HY    

        def Z_check_matrix() -> np.ndarray:
            if (self.bc == 'open'): 
                c = 1
            elif (self.bc=='periodic'): 
                c = 0     
            HZ = np.zeros((self.rows-c, self.rows*self.cols), dtype=np.uint8)
            for r in range(self.rows-c):          # stabiliser index (pair of rows r and r+1)
                y1 = r
                y2 = (r + 1) % (self.rows-c)
                for x in range(self.cols):
                    j1 = self.idx(x,y1)           # qubits in row y1
                    j2 = self.idx(x,y2)           # qubits in row y2
                    HZ[r, j1] = 1
                    HZ[r, j2] = 1
            return HZ    

        Hx = X_check_matrix()
        Hy = Y_check_matrix()
        Hz = Z_check_matrix()
        return np.block([[Hx, np.zeros_like(Hx)],
                         [Hy, Hy],
                         [np.zeros_like(Hz),Hz]]).astype(np.uint8)

    
    def stabiliser_matrix(self) -> np.ndarray:
        """
            Returns stabilisers' check matrix of shape (n_stabilisers, 2*n_qubits) in [X|Z] form.
            The stabiliser check matrix is defined as the centre of the gauge group.
            For L even, intentionally, the number of rows is rank_f2_rowrank+1. 
            To get independent stabiliser generators, use row_space_basis_f2().
        """
        H_stab = self.double_loop_matrix() 
        centre_of_G = center_check_matrix(self.updown_triangles_check_matrix())
        extra_vec = find_extra_center_element(centre_of_G, H_stab)
        if(extra_vec is not None): 
            H_stab = np.vstack((H_stab,extra_vec))
        return H_stab



    #full up triangular plaquette decomposition
    def up_triangles(self):
        triangles = []
        if (self.bc == 'open'): 
            c = 1
        elif (self.bc=='periodic'): 
            c = 0 
        for x in range(self.cols-c):
            for y in range(self.rows-c):
                tri = (
                    self.idx(x, y), #Y
                    self.idx(x+1, y), #X
                    self.idx(x , y + 1), #Z
                )
                triangles.append(tuple(tri))
        return sorted(set(triangles))

    #full down triangular plaquette decomposition
    def down_triangles(self):
        triangles = []
        if (self.bc == 'open'): 
            c = 1
        elif (self.bc=='periodic'): 
            c = 0         
        for x in range(self.cols-c):
            for y in range(self.rows-c):
                tri = (
                    self.idx(x, y), #X
                    self.idx(x+1, y), #Y
                    self.idx(x+1, y - 1), #Z
                )
                triangles.append(tuple(tri))
        return sorted(set(triangles))
    
    def up_terms(self):
        up_triangles = self.up_triangles() 
        up_terms = []
        for i, j, k in up_triangles:
            p = ["I"] * self.rows*self.cols
            p[i] = "Y"
            p[j] = "X"
            p[k] = "Z"
            up_terms.append(("".join(reversed(p)), 1.0)) 
        return up_terms
    
    def down_terms(self):
        down_triangles = self.down_triangles()
        down_terms = []
        for i, j, k in down_triangles:
            p = ["I"] * self.rows*self.cols
            p[i] = "X"
            p[j] = "Y"
            p[k] = "Z"
            down_terms.append(("".join(reversed(p)), 1.0))
        return down_terms

    def triangles_check_matrix(self,t = "up") -> np.ndarray:
        """
            :param t: Triangle type "up" or "down".
            Returns gauge triangles' check matrix of shape (n_triangles, 2*n_qubits) in [X|Z] form.
        """
        ttablex = str.maketrans({"I": "0", "X": "1", "Y": "0", "Z": "0"})
        ttabley = str.maketrans({"I": "0", "X": "0", "Y": "1", "Z": "0"})
        ttablez = str.maketrans({"I": "0", "X": "0", "Y": "0", "Z": "1"})

        if t == "up":
            lt= [item[0] for item in self.up_terms()]
        elif t == "down":
            lt= [item[0] for item in self.down_terms()]            
        ltx = [text.translate(ttablex) for text in lt]
        lty = [text.translate(ttabley) for text in lt]
        ltz = [text.translate(ttablez) for text in lt]
        ltxy = [
            f"{int(b1, 2) | int(b2, 2):0{len(b1)}b}" 
            for b1, b2 in zip(ltx, lty)
        ]
        ltzy = [
            f"{int(b1, 2) | int(b2, 2):0{len(b1)}b}" 
            for b1, b2 in zip(ltz, lty)
        ]
        ltxz = [b1 + b2 for b1, b2 in zip(ltxy, ltzy)] #stacked [X|Z] where list elements are rows
        t_checkmatrix = np.array([[int(b) for b in s] for s in ltxz])
        return t_checkmatrix
    
    def updown_triangles_check_matrix(self) -> np.ndarray:
        """ 
            Returns gauge triangles' check matrix of shape (n_uptriangles+n_downtriangles, 2*n_qubits) in [X|Z] form.
        """
        return np.vstack((self.triangles_check_matrix("up"),self.triangles_check_matrix("down")))

    #Hamiltonians
    def H_up(self):
        return SparsePauliOp.from_list(self.up_terms())
    
    def H_down(self):
        return SparsePauliOp.from_list(self.down_terms())
    
    def H_up_param(self,parameter_prefix='u'):
        up_opterms, up_coefs = zip(*self.up_terms())
        return SparsePauliOp(up_opterms,np.array(ParameterVector(parameter_prefix, len(up_opterms))))

    def H_down_param(self,parameter_prefix='d'): 
        down_opterms, down_coefs = zip(*self.down_terms())
        return SparsePauliOp(down_opterms,np.array(ParameterVector(parameter_prefix, len(down_opterms))))
    

# ---------------------------
# Binary symplectic utilities
# ---------------------------

def gf2(A):
    return (np.array(A, dtype=np.uint8) & 1).copy()


def rref_f2(A):
    """
    Reduced row echelon form over F2.
    Returns (R, pivots), where pivots are pivot column indices.
    """
    M = gf2(A)
    m, n = M.shape
    pivots = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        pivot = next((r for r in range(row, m) if M[r, col]), None)
        if pivot is None:
            continue

        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]

        for r in range(m):
            if r != row and M[r, col]:
                M[r] ^= M[row]

        pivots.append(col)
        row += 1

    return M, pivots


def rank_f2(A):
    R, _ = rref_f2(A)
    return int(np.count_nonzero(np.any(R, axis=1)))


def row_space_basis_f2(A):
    R, _ = rref_f2(A)
    return R[np.any(R, axis=1)]


def nullspace_basis_f2(A):
    """
    Basis for {x : A x = 0} over F2.
    Returns basis vectors as rows.
    """
    M = gf2(A)
    R, pivots = rref_f2(M)
    m, n = R.shape
    pivot_set = set(pivots)
    free_cols = [j for j in range(n) if j not in pivot_set]

    if not free_cols:
        return np.zeros((0, n), dtype=np.uint8)

    pivot_row_for_col = {c: i for i, c in enumerate(pivots)}
    basis = []

    for free in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[free] = 1

        for pc in reversed(pivots):
            r = pivot_row_for_col[pc]
            s = 0
            for j in free_cols:
                if R[r, j] and x[j]:
                    s ^= 1
            x[pc] = s

        basis.append(x)

    return np.array(basis, dtype=np.uint8)


def solve_f2(A, b):
    """
    Solve A x = b over F2.
    Returns one solution x, or None if inconsistent.
    Free variables are set to 0.
    """
    A = gf2(A)
    b = gf2(b).reshape(-1, 1)
    Aug = np.concatenate([A, b], axis=1)
    R, pivots = rref_f2(Aug)

    m, n1 = R.shape
    n = n1 - 1

    for r in range(m):
        if not np.any(R[r, :n]) and R[r, n]:
            return None

    x = np.zeros(n, dtype=np.uint8)
    pivot_row_for_col = {c: i for i, c in enumerate(pivots)}
    free_cols = [j for j in range(n) if j not in set(pivots)]

    for pc in reversed(pivots):
        r = pivot_row_for_col[pc]
        s = R[r, n]
        for j in free_cols:
            if R[r, j] and x[j]:
                s ^= 1
        x[pc] = s

    return x


def in_row_span_f2(M, v):
    M = gf2(M)
    v = gf2(v).reshape(1, -1)
    return solve_f2(M.T, v.T) is not None


def symplectic_product_matrix_J(n_qubits):
    I = np.eye(n_qubits, dtype=np.uint8)
    Z0 = np.zeros((n_qubits, n_qubits), dtype=np.uint8)
    return np.block([[Z0, I],
                     [I,  Z0]]).astype(np.uint8)


def center_check_matrix(H_G):
    """
    Rows of H_G are gauge generators in [X|Z] form over F2.
    Returns a basis for the center: all combinations commuting with every gauge row.
    """
    H = gf2(H_G)
    m, w = H.shape
    assert w % 2 == 0, "H_G must have 2n columns"
    n = w // 2

    if m == 0:
        return np.zeros((0, 2 * n), dtype=np.uint8)

    J = symplectic_product_matrix_J(n)
    C = (H @ J @ H.T) & 1

    # Find u such that C^T u = 0
    N = nullspace_basis_f2(C.T)

    # Map each u to u^T H
    H_S = (N @ H) & 1
    return row_space_basis_f2(H_S)


def find_extra_center_element(H_S_true, H_S_given):
    """
    Returns one vector in Span(H_S_true) not in Span(H_S_given), or None.
    """
    B_true = row_space_basis_f2(H_S_true)
    for vec in B_true:
        if not in_row_span_f2(H_S_given, vec):
            return vec
    return None


def solve_left_coeffs_f2(H_G, v):
    """
    Find u such that u^T H_G = v over F2.
    Returns one solution u, or None if impossible.
    """
    H = gf2(H_G)
    v = gf2(v)
    A = H.T
    u = solve_f2(A, v)
    return u
