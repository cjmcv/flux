import torch


def svd_compression(matrix: torch.Tensor, compression_ratio: float = 0.3):

    K, N = matrix.t().shape
    max_rank = min(K, N)
    
    L= max(1, int(max_rank * compression_ratio))
    matrix_cpu = matrix.cpu().double()
    U,S,Vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
    
    C = (U[:,:L] @ torch.diag(torch.sqrt(S[:L]))).to(matrix.dtype).to(matrix.device).t()
    D = (torch.diag(torch.sqrt(S[:L])) @ Vh[:L,:]).to(matrix.dtype).to(matrix.device).t()
    
    return C, D

if __name__ == "__main__":
    M = 1024
    N = 1024
    K = 1024
    
    A = torch.ones((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.ones((N, K), device="cuda", dtype=torch.bfloat16)
    C = torch.nn.functional.linear(A , B)
    print(C)
    
    E,F = svd_compression(B)
    print(E, F)
    G = torch.nn.functional.linear(torch.nn.functional.linear(A , E), F)
    print(G)