import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def is_stable(A):
    """Check if all eigenvalues have negative real parts."""
    eigenvalues = np.linalg.eigvals(A)
    return np.all(np.real(eigenvalues) < 0)

def construct_system_matrix(M_inv, D, Kp, Ki, Kd):
    """Construct the system matrix A using the gains."""
    A = np.block([
        [np.zeros((3,3)), np.eye(3), np.zeros((3,3))],
        [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)],
        [-M_inv@Ki, -M_inv@Kp, -M_inv@(D+Kd)]
    ])
    return A

def random_gain_matrix(min_val=1, max_val=10):
    """Generate a 3x3 diagonal gain matrix with the same integer on all diagonal elements."""
    # Generate a single random integer
    value = np.random.randint(min_val, max_val+1)
    # Create a diagonal matrix with this same value on all diagonal elements
    return value * np.eye(3)

def generate_stable_gains(M, D, max_iterations=10000, min_val=1, max_val=20):
    """
    Generate random K-gains until a stable system is found.
    
    Parameters:
    -----------
    M : array_like
        Mass matrix
    D : array_like
        Damping matrix
    max_iterations : int
        Maximum number of iterations to try
    min_val : int
        Minimum integer value for gain elements
    max_val : int
        Maximum integer value for gain elements
        
    Returns:
    --------
    tuple
        (Kp, Ki, Kd, eigenvalues, iterations, history)
    """
    M_inv = np.linalg.inv(M)
    
    # Keep track of all attempted gains and their stability
    history = []
    
    for i in range(max_iterations):
        # Generate random diagonal gain matrices
        Kp = random_gain_matrix(min_val, max_val)
        Ki = random_gain_matrix(min_val, max_val)
        Kd = random_gain_matrix(min_val, max_val)
        
        # Construct system matrix
        A = construct_system_matrix(M_inv, D, Kp, Ki, Kd)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(A)
        
        # Check stability
        stable = is_stable(A)
        
        # Store attempt in history
        history.append({
            'Kp': Kp[0,0],  # Just store the value since all diagonal elements are the same
            'Ki': Ki[0,0],
            'Kd': Kd[0,0],
            'max_real': np.max(np.real(eigenvalues)),
            'stable': stable
        })
        
        if stable:
            print(f"Found stable gains after {i+1} iterations!")
            return Kp, Ki, Kd, eigenvalues, i+1, history
    
    print(f"Failed to find stable gains after {max_iterations} iterations.")
    return None, None, None, None, max_iterations, history

def visualize_search(history):
    """Visualize the search process."""
    # Extract data from history
    iterations = len(history)
    max_reals = [entry['max_real'] for entry in history]
    stable_points = [i for i, entry in enumerate(history) if entry['stable']]
    
    # Plot the maximum real part of eigenvalues over iterations
    plt.figure(figsize=(12, 6))
    plt.plot(range(iterations), max_reals, 'b-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.7)
    
    if stable_points:
        plt.scatter(stable_points, [max_reals[i] for i in stable_points], 
                   color='g', s=50, label='Stable configurations')
    
    plt.xlabel('Iteration')
    plt.ylabel('Maximum real part of eigenvalues')
    plt.title('Search for Stable Configuration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # If stable solutions were found, plot a 3D visualization of gains for stable configurations
    if stable_points:
        stable_entries = [history[i] for i in stable_points]
        
        # Get the gain values (already single values since all diagonal elements are the same)
        kp_vals = [entry['Kp'] for entry in stable_entries]
        ki_vals = [entry['Ki'] for entry in stable_entries]
        kd_vals = [entry['Kd'] for entry in stable_entries]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(kp_vals, ki_vals, kd_vals, c='g', marker='o', s=50, alpha=0.6)
        
        ax.set_xlabel('Kp[0,0]')
        ax.set_ylabel('Ki[0,0]')
        ax.set_zlabel('Kd[0,0]')
        ax.set_title('Stable Gain Configurations')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # System parameters
    D = np.array([[0.66, 0.0, 0.0], [0.0, 1.3, 2.8], [0.0, 0.0, 1.9]])
    M = np.array([[16.0, 0.0, 0.0], [0.0, 24.0, 0.53], [0.0, 0.53, 2.8]])
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Searching for stable gains...")
    start_time = time.time()
    
    # Run the algorithm
    Kp, Ki, Kd, eigenvalues, iterations, history = generate_stable_gains(M, D)
    
    end_time = time.time()
    
    if Kp is not None:
        print(f"\nStable configuration found in {end_time - start_time:.2f} seconds after {iterations} iterations.")
        print("\nStable Gains:")
        print("Kp =")
        print(Kp)
        print("\nKi =")
        print(Ki)
        print("\nKd =")
        print(Kd)
        
        print("\nEigenvalues:")
        for i, eig in enumerate(eigenvalues):
            print(f"λ{i+1} = {eig:.4f}")
        
        # Visualize the search process
        visualize_search(history)
    else:
        print("No stable configuration found within the iteration limit.")
        
        # Visualize the search process anyway
        visualize_search(history)