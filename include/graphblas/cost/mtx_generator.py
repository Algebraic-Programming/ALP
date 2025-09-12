import numpy as np
import sys
import os
from scipy.io import mmwrite
from scipy.sparse import diags

def create_banded_diag_mtx(n: int, band_size: int, output_path=None):
    """
    Create a sparse banded diagonal matrix of size n x n with random values
    on the diagonals and save it as a .mtx file.

    Parameters
    ----------
    n : int
        Number of rows/columns (matrix is n x n)
    band_size : int
        Half-bandwidth of the matrix (total width = 2*band_size+1)
    output_path : str, optional
        Directory where the matrix file should be saved. If None, saves in current directory.
    """
    # Generate the diagonals
    diagonals = []
    offsets = []
    
    # Main diagonal and surrounding bands (total of 2*band_size+1 diagonals)
    for k in range(-band_size, band_size + 1):
        # Determine length of this diagonal
        if k < 0:
            length = n + k
        elif k > 0:
            length = n - k
        else:
            length = n
            
        # Create random values for this diagonal
        diagonal_values = np.random.rand(length)
        
        # For the main diagonal (k=0), ensure values are positive and larger
        if k == 0:
            diagonal_values = diagonal_values * 10 + 1.0
            
        diagonals.append(diagonal_values)
        offsets.append(k)
    
    # Create a sparse diagonal matrix
    A = diags(diagonals, offsets, shape=(n, n), format='csr')

    # Create filename
    filename = f"banded_diag_{n}x{n}_band_{band_size}.mtx"
    
    # If output_path is provided, join it with the filename
    if output_path:
        # Make sure the directory exists
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)
    else:
        full_path = filename

    # Write to Matrix Market format
    mmwrite(full_path, A)

    print(f"Created {full_path} with size {n}x{n}, band size {band_size} (total width: {2*band_size+1})")

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python mtx_generator.py <matrix_size> <band_size> [output_path]")
        print("Example: python mtx_generator.py 100 2")
        print("Example with path: python mtx_generator.py 100 2 /path/to/save/directory")
        print("Note: band_size defines half the bandwidth (total width = 2*band_size+1)")
        sys.exit(1)
    
    try:
        # Parse the matrix size and band size from command line
        n = int(sys.argv[1])
        band_size = int(sys.argv[2])
        
        # Check for valid input
        if n <= 0:
            print("Error: Matrix size must be a positive integer")
            sys.exit(1)
            
        if band_size < 0 or band_size >= n:
            print(f"Error: Band size must be between 0 and {n-1}")
            sys.exit(1)
        
        # Check if output path is provided
        output_path = None
        if len(sys.argv) >= 4:
            output_path = sys.argv[3]
            
        # Create the matrix
        create_banded_diag_mtx(n, band_size, output_path)
        
    except ValueError:
        print("Error: Both matrix size and band size must be integers")
        sys.exit(1)

if __name__ == "__main__":
    main()
