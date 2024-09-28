import os, sys
sys.path.insert(0, os.path.abspath('..'))
import math
from tqdm import tqdm
import numpy as np
import torch
import gc

def low_cache_mode(X):
    X = X.to('cpu')
    torch.cuda.empty_cache()
    gc.collect()
    return X

def gpu_speed_mode(X):
    return X.to('cuda')

def check_memory(desc=""):
    reserved_memory = torch.cuda.memory_allocated()
    _, total_memory = torch.cuda.mem_get_info()
    
    # Convert the values from bytes to megabytes (MB)
    free_memory_GB = (total_memory - reserved_memory) / (1024 ** 3)
    total_memory_GB = total_memory / (1024 ** 3)
    print(f">>>>>>>> {desc} <<<<<<<<")
    print(f"Free memory: {free_memory_GB:.2f} GB")
    print(f"Used memory: {total_memory_GB - free_memory_GB:.2f} GB")
    print(f"Total memory: {total_memory_GB:.2f} GB")
    
    # torch.cuda.empty_cache()
    # gc.collect()
    # torch.cuda.empty_cache()
    # free_memory, total_memory = torch.cuda.mem_get_info()

    # # Convert the values from bytes to megabytes (MB)
    # free_memory_MB = free_memory / (1024 ** 3)
    # total_memory_MB = total_memory / (1024 ** 3)
    # print(f"Free memory: {free_memory_MB:.2f} GB")
    # print(f"Used memory: {total_memory_MB - free_memory_MB:.2f} GB")
    # print(f"Total memory: {total_memory_MB:.2f} GB")

class TSNE():
    def __init__(self, n_components=2, n_iter=1000, learning_rate=200.0):
        self.n_components = n_components
        self.n_iter = n_iter
        self.learning_rate = learning_rate
    
    def get_chunk_size(self, X: torch.tensor, rate: float):
        reserved_memory = torch.cuda.memory_allocated()
        _, total_memory = torch.cuda.mem_get_info()
        free_memory = total_memory - reserved_memory
        row_size = X.element_size()*X.size(1)
        chunk_size = int((free_memory*rate) / row_size)
        chunk_size = max(min(X.size(0), chunk_size), 1)
        
        return chunk_size
    
    def sum_x_square(self, X):
        sum_X_matrix = torch.zeros(X.size(0), device='cpu')
        chunk_size = self.get_chunk_size(X, 0.3) if X.size(0) != 5 else 2
        with tqdm(total=math.ceil(X.size(0)/chunk_size), desc="executing sum_x_square...") as pbar:
            for i in range(0, X.size(0), chunk_size):
                chunk = gpu_speed_mode(X[i:i+chunk_size])
                chunk_squared = chunk*chunk
                sum_chunk = torch.sum(chunk_squared, dim=1)
                sum_X_matrix[i:i + chunk_size] = low_cache_mode(sum_chunk)
                pbar.update(1)
            del chunk, chunk_squared, sum_chunk
            torch.cuda.empty_cache()
        return sum_X_matrix
    
    def dot(self, X, Y):
        dot_product_matrix = torch.zeros((X.size(0), Y.size(1)), device='cpu')
        chunk_size = self.get_chunk_size(X, 0.02) if X.size(0) != 5 else 2
        total = math.ceil(X.size(0)/chunk_size)*math.ceil(Y.size(1)/chunk_size)
        with tqdm(total=total, desc="executing dot...") as pbar:
            for i in range(0, X.size(0), chunk_size):
                chunk_i = gpu_speed_mode(X[i:i+chunk_size, :])
                for j in range(0, Y.size(1), chunk_size):
                    chunk_j = gpu_speed_mode(Y[:, j:j+chunk_size])
                    # Compute the dot product with chunks
                    dot_product = torch.matmul(chunk_i, chunk_j)
                    dot_product_matrix[i:i + chunk_size, j:j + chunk_size] = low_cache_mode(dot_product)
                    pbar.update(1)
            del chunk_i, chunk_j, dot_product
            torch.cuda.empty_cache()
        return dot_product_matrix
    
    def sum(self, X, Y):
        """
        Perform element-wise sum of tensors X and Y in chunks.
        If Y is 1D, expand it to 2D to match the shape of X.
        """
        # Ensure Y is 2D
        if Y.dim() == 1:
            Y = Y.unsqueeze(0).expand(X.size(0), -1)
        # Ensure both tensors have the same size after expansion
        assert X.size(0) == Y.size(0), "Tensors must have the same number of rows"
        
        # Initialize result as a list to hold chunk results
        sum_matrix = torch.zeros(X.size(), device='cpu')
        chunk_size = self.get_chunk_size(X, 0.2)
        total = math.ceil(X.size(0)/chunk_size)*math.ceil(Y.size(1)/chunk_size)
        with tqdm(total=total, desc="executing sum...") as pbar:
            # Process X and Y in chunks
            for i in range(0, X.size(0), chunk_size):
                # Initialize a list to hold the sum of the current chunk
                for j in range(0, X.size(1), chunk_size):
                    # Get the current chunk of X
                    X_chunk = gpu_speed_mode(X[i:i + chunk_size, j:j + chunk_size])
                    # Get the current chunk of Y
                    Y_chunk = gpu_speed_mode(Y[i:i + chunk_size, j:j + chunk_size])

                    # Perform element-wise sum for the current chunks
                    chunk_sum = X_chunk + Y_chunk
                    # Append the chunk sum to the results
                    sum_matrix[i:i + chunk_size, j:j + chunk_size] = low_cache_mode(chunk_sum)
                    pbar.update(1)

            del X_chunk, Y_chunk, chunk_sum
            torch.cuda.empty_cache()
        return sum_matrix
       
    def neg_squared_euc_dists(self, X):
        """
        Compute pairwise distances in the dataset.
        """
        sum_squareX = self.sum_x_square(X)
        dot_XXt = self.dot(X, X.T)
        sum_xxt_sum_squareX = self.sum(-2* dot_XXt, sum_squareX)
        D = self.sum(sum_xxt_sum_squareX.T, sum_squareX)
        if (X.shape[0] == 5):
            print(torch.allclose(sum_squareX, torch.sum(torch.square(X), dim=1), rtol=1e-05, atol=1e-06))
            diff = torch.abs(sum_squareX - torch.sum(torch.square(X), dim=1))
            print("Element-wise absolute difference:", diff)
            
            print(torch.allclose(dot_XXt, torch.matmul(X, X.T), rtol=1e-05, atol=1e-08))
            
            print(f"sum_xxt_sum_squareX:\n{sum_xxt_sum_squareX}")
            sum1 = -2 * dot_XXt + sum_squareX
            print(f"sum_xxt_sum_squareX correct:\n{sum1}")
            
            print(f"D:\n{D}")
            print(f"D correct:\n{(sum1).T + sum_squareX}")
            
            diff = torch.abs(D - ((sum1).T + sum_squareX))
            print("Element-wise absolute difference:\n", diff)
        
        return -D
    
    def softmax(self, X, diag_zero_id):
        """Take softmax of each row of matrix X."""
        # Initialize a list to hold the results
        X = gpu_speed_mode(X)
        print(f"e_x: {X}")
        print(f"max_x: {torch.max(X, axis=1).values.reshape(-1,1)}")
        e_x = torch.exp(X - torch.max(X, axis=1).values)
        print(f"updated e_x: {e_x}")

        # We usually want diagonal probailities to be 0.
        e_x[0, diag_zero_id] = 0
            

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability
        
        softmax = low_cache_mode(e_x / e_x.sum(axis=1).reshape([-1,1]))
        del X, e_x
        torch.cuda.empty_cache()
        print(f"softmax: {softmax}")

        return softmax
        
        # softmax_matrix = torch.zeros(X.size(), device='cpu')

        # chunk_size = self.get_chunk_size(X, 0.2) if X.size(1) != 5 else 2
        # # Process in chunks
        # with tqdm(total=math.ceil(X.size(0)/chunk_size), desc="executing softmax...") as pbar:
        #     for i in range(0, X.size(0), chunk_size):
        #         # Get the current chunk
        #         X_chunk = gpu_speed_mode(X[i:i + chunk_size])

        #         # Compute the max for the current chunk along axis 1
        #         max_vals = torch.max(X_chunk, axis=1).values.reshape([-1, 1])

        #         # Subtract the max values and compute the exponential
        #         e_x_chunk = torch.exp(X_chunk - max_vals)
                
        #         if diag_zero:
        #             e_x_chunk[torch.arange(e_x_chunk.size(0)), torch.arange(i, min(i+chunk_size, X.size(0)))] = 0.
        #         e_x_chunk += 1e-8
        #         e_x_chunk_sum = e_x_chunk.sum(dim=1).reshape(-1, 1)
        #         e_x_over_e_x_sum = e_x_chunk / e_x_chunk_sum
        #         softmax_matrix[i:i + chunk_size] = low_cache_mode(e_x_over_e_x_sum)
        #         pbar.update(1)
        #     del X_chunk, max_vals, e_x_chunk, e_x_chunk_sum, e_x_over_e_x_sum
        #     torch.cuda.empty_cache()
    
        # if X.size(1) == 5:
        #     correct_e_x = torch.exp(X - torch.max(X, axis=1).values.reshape([-1, 1]))
        #     if diag_zero:
        #         correct_e_x[torch.arange(correct_e_x.size(0)), torch.arange(correct_e_x.size(0))] = 0.
        #     correct_e_x = correct_e_x + 1e-8
        #     correct_e_x_sum = correct_e_x.sum(axis=1).reshape([-1, 1])
        #     correct_softmax = correct_e_x / correct_e_x_sum
        #     print(f"softmax:\n{softmax_matrix}")
        #     print(f"softmax correct:\n{correct_softmax}")
        #     print(torch.allclose(softmax_matrix, correct_softmax, rtol=1e-05, atol=1e-08))
        # return softmax_matrix
        
    def calc_prob_matrix(self, distances, diag_zero_id, sigmas=None):
        """Convert a distances matrix to a matrix of probabilities."""
        if sigmas is not None:
            two_sig_sq = 2 * (sigmas ** 2)
            print(f"distances:\n{distances}")
            print(f"two_sig_sq:{two_sig_sq}")
            return self.softmax(distances / two_sig_sq, diag_zero_id)
        else:
            return self.softmax(distances, diag_zero_id)
    
    def log2(self, X):
        log2_X = torch.zeros(X.size(), device='cpu')
        chunk_size = self.get_chunk_size(X, 0.2)
        total = math.ceil(X.size(0)/chunk_size)
        # with tqdm(total=X.size(0), desc="executing log2...") as pbar:
        for i in range(0, X.size(0), chunk_size):
            # Get the current chunk of X
            chunk = gpu_speed_mode(X[i:i + chunk_size])
            # Apply log2 to the chunk
            chunk_log2 = torch.log2(chunk)
            # Store the result back in the pre-allocated tensor
            log2_X[i:i + chunk_size] = low_cache_mode(chunk_log2)
                # Update progress bar
                # pbar.update(1)
                # Cleanup for memory management
        del chunk, chunk_log2
        torch.cuda.empty_cache()
            
        return low_cache_mode(log2_X)
    
    def calc_perplexity(self, prob_matrix):
        """Calculate the perplexity of each row 
        of a matrix of probabilities."""
        # log2_prob = self.log2(prob_matrix)
        prob_matrix = gpu_speed_mode(prob_matrix)
        log2_prob = torch.log2(prob_matrix)
        print(f"prob_matrix: {prob_matrix}")
        print(f"log2_prob: {log2_prob}")
        prob_dot_log2_prob = prob_matrix * log2_prob
        entropy = -torch.sum(prob_dot_log2_prob, dim=1)
        print(f"entropy: {entropy}")
        perplexity = low_cache_mode(2 ** entropy)
        if (prob_matrix.size() == 5):
            print(torch.allclose(torch.log2(prob_matrix), log2_prob, rtol=1e-05, atol=1e-06))
            diff = torch.abs(torch.log2(prob_matrix) - log2_prob)
            print("Element-wise absolute difference:", diff)
        return perplexity

    def perplexity(self, distances, diag_zero_id, sigmas):
        """Wrapper function for quick calculation of 
        perplexity over a distance matrix."""
        prob_matrix = self.calc_prob_matrix(distances, diag_zero_id,sigmas)
        return self.calc_perplexity(prob_matrix)

    def binary_search(self, eval_fn, target, tol=1e-10, lower=1000, upper=1e5):
        """Perform a binary search over input values to eval_fn.
        
        # Arguments
            eval_fn: Function that we are optimising over.
            target: Target value we want the function to output.
            tol: Float, once our guess is this close to target, stop.
            max_iter: Integer, maximum num. iterations to search for.
            lower: Float, lower bound of search range.
            upper: Float, upper bound of search range.
        # Returns:
            Float, best input value to function found during search.
        """
        i = 0
        while lower+1 < upper:
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            print(val)
            if abs(val - target) <= tol:
                return guess
            elif val > target:
                upper = guess
            else:
                lower = guess
            print(f"[{i}] sigmas: {guess} | diff: {abs(val - target)}")
            i += 1
        input()
        return guess

    def find_optimal_sigmas(self, distances, target_perplexity):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        sigmas = []
        # For each row of the matrix (each point in our dataset)
        for i in range(distances.size(0)):
            # eval_fn = lambda sigma: \
            #     self.perplexity(distances[i:i+1], i, torch.tensor([sigma]))
            # correct_sigma = self.binary_search(eval_fn, target_perplexity)
            # sigmas.append(correct_sigma)
            print(self.perplexity(distances[i:i+1], i, torch.tensor([50])))
        input()
        return sigmas
    
    def p_conditional_to_joint(self, P):
        """Given conditional probabilities matrix P, return
        approximation of joint distribution probabilities."""
        return (P + P.T) / (2. * P.shape[0])
    
    def p_joint(self, X, target_perplexity):
        """Given a data matrix X, gives joint probabilities matrix.

        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        # Get the negative euclidian distances matrix for our data
        distances = self.neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self.find_optimal_sigmas(distances, target_perplexity)
        exit()
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = self.calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        P = self.p_conditional_to_joint(p_conditional)
        return P
    
    def symmetric_sne_grad(P, Q, Y, _):
        """Estimate the gradient of the cost with respect to Y"""
        pq_diff = P - Q  # NxN matrix
        pq_expanded = np.expand_dims(pq_diff, 2)  #NxNx1
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  #NxNx2
        grad = 4. * (pq_expanded * y_diffs).sum(1)  #Nx2
        return grad
    
    def q_joint(Y):
        """Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        # Get the distances from every point to every other
        distances = neg_squared_euc_dists(Y)
        # Take the elementwise exponent
        exp_distances = np.exp(distances)
        # Fill diagonal with zeroes so q_ii = 0
        np.fill_diagonal(exp_distances, 0.)
        # Divide by the sum of the entire exponentiated matrix
        return exp_distances / np.sum(exp_distances), None
 
    def q_tsne(self, Y):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self.neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    
    
    def fit_transform(self, X: torch.tensor, momentum=False):
        """Estimates a SNE model.
        # Arguments
            X: Input data matrix.
        # Returns:
            Y: Matrix, low-dimensional representation of X.
        """
        P = self.p_joint(X, 25) # target perplexity suppose 25
        
        # Initialise our 2D representation
        Y = torch.normal(0., 0.0001, [X.shape[0], 2], device='cpu')

        # Initialise past values (used for momentum)
        if momentum:
            Y_m2 = Y.detach().clone()
            Y_m1 = Y.detach().clone()

        # Compute pairwise distances in the high-dimensional space
        # Start gradient descent loop
        # for i in range(self.num_iters):
        #     # Get Q and distances (distances only used for t-SNE)
        #     Q, distances = q_fn(Y)
        #     # Estimate gradients with respect to Y
        #     grads = grad_fn(P, Q, Y, distances)

        #     # Update Y
        #     Y = Y - learning_rate * grads
        #     if momentum:  # Add momentum
        #         Y += momentum * (Y_m1 - Y_m2)
        #         # Update previous Y's for momentum
        #         Y_m2 = Y_m1.copy()
        #         Y_m1 = Y.copy()

        #     # Plot sometimes
        #     if plot and i % (num_iters / plot) == 0:
        #         categorical_scatter_2d(Y, y, alpha=1.0, ms=6,
        #                             show=True, figsize=(9, 6))

        # return Y

# Example usage
if __name__ == "__main__":
    torch.manual_seed(42)
    tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0)
    data = torch.rand(5, 5)
    transformed_data = tsne.fit_transform(data)
    
    tsne = TSNE(n_components=2, n_iter=1000, learning_rate=200.0)
    print("[large data] Generating data ...", end='')
    data = torch.rand(45834*4, 4096) # 45834 * 42 # 45834*4
    print("Done")
    transformed_data = tsne.fit_transform(data)
    print("Transformed data:\n", transformed_data)