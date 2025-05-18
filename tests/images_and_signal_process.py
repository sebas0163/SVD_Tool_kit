from PIL import Image
import os
import numpy as np
import SVD_Tool_Kit
import matplotlib.pyplot as plt
"""
    Compress a grayscale image using a truncated (compact) Singular Value Decomposition (SVD).

    Inputs:
        image_name (str): Filename of the grayscale image located in the "assets" folder.
        k (int): Number of singular values/components to keep for the compressed approximation.

    Process:
        - Load the image and convert it to a grayscale matrix.
        - Display the original image.
        - Compute the full SVD decomposition of the image matrix: mat = U * S * V^T.
        - Truncate U to keep the first k columns.
        - Truncate S to keep the top-left k x k block (singular values).
        - Truncate V to keep the first k rows (not columns).
        - Reconstruct the image using the truncated matrices: U_k * S_k * V_k^T.
        - Display the compressed/reconstructed image.

    Output:
        - Two plots: the original grayscale image and the reconstructed image with rank k approximation.
    """
def compressImage_copact_SVD(image_name,k):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "assets", image_name)
    img = Image.open(img_path).convert('L')
    mat = np.array(img, dtype=np.float64)
    plt.imshow(mat, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    plt.show()
    u,s,v = SVD_Tool_Kit.svd(mat, False)
    u = u[:, :k]
    s = s[:k, :k]
    v = v[:, :k]
    reconst = u@s@v.T
    plt.imshow(reconst, cmap='gray')
    plt.title(f"Reconstruction with k={k}")
    plt.axis('off')
    plt.show()
"""
    Test function to perform Generalized Singular Value Decomposition (GSVD) on two grayscale images.

    Inputs:
        image_1 (str): Filename of the first grayscale image located in the "assets" folder.
        image_2 (str): Filename of the second grayscale image located in the "assets" folder.

    Process:
        - Load the two images and convert them to grayscale matrices of type float64.
        - Check that both images have the same shape.
        - Compute the GSVD of the two matrices using the `SVD_Tool_Kit.GSVD` function.
        - Reconstruct the original matrices from the GSVD factors.
        - Plot the reconstructed images side by side.
        - Plot the absolute difference between the two reconstructed images as a heatmap.

    Output:
        - A matplotlib figure showing:
          1) Reconstructed image A,
          2) Reconstructed image B,
          3) Heatmap of the absolute difference between the two reconstructed images.

    Notes:
        - The reconstruction uses the formula: reconstructed = U * D * inv(X).
        - Assumes the `SVD_Tool_Kit.GSVD` returns the matrices (U_1, U_2, D_a, D_b, X) needed for reconstruction.
        - This function helps visualize how the GSVD separates two datasets (images) and their differences.
    """
def test_img_GSVD(image_1, image_2):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path_1 = os.path.join(script_dir, "assets", image_1)
    img_1 = Image.open(img_path_1).convert('L')
    mat_1 = np.array(img_1, dtype=np.float64)
    img_path_2 = os.path.join(script_dir, "assets", image_2)
    img_2 = Image.open(img_path_2).convert('L')
    mat_2 = np.array(img_2, dtype=np.float64)
    assert mat_2.shape == mat_1.shape
    u_1, u_2, d_a, d_b, x = SVD_Tool_Kit.GSVD(mat_1,mat_2)
    reconst_1 = u_1 @d_a @ np.linalg.inv(x)
    reconst_2 = u_2 @d_b @ np.linalg.inv(x)
    plt.figure(figsize=(6, 4))
    plt.subplot(2, 3, 1)
    plt.title(f"A reconstructed")
    plt.imshow(reconst_1, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 3, 2)
    plt.title(f"B reconstructed")
    plt.imshow(reconst_2, cmap="gray")
    plt.axis("off")
    plt.subplot(2, 3, 3)
    plt.title("Difference")
    plt.imshow(np.abs(reconst_1 - reconst_2), cmap="hot")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
"""
    Reconstruct a tensor from its HOSVD factors by truncating to the top k components.

    Inputs:
        matrix_u_list (list of np.ndarray): List of U factor matrices from HOSVD for each mode.
        sigma_list (list of np.ndarray): List of diagonal singular value matrices (core tensors) for each mode.
        matrix_v (np.ndarray): V matrix shared across modes.
        k (int): Number of leading components to retain for reconstruction (controls compression level).

    Process:
        - For each mode, take the first k columns of U (U_k) and first k singular values in S (S_k).
        - Take the first k columns of V (V_k).
        - Reconstruct the approximation matrix A_rec = U_k @ S_k @ V_k.T for each mode.
        - Collect all reconstructed mode matrices into a tensor.

    Output:
        np.ndarray: Reconstructed tensor with compressed rank k approximation for each mode.

    Note:
        This function assumes the input factor matrices and singular value matrices are consistent
        with the output of a high-order SVD function and that matrix_v is shared for all modes.
    """
def hosvd_reconstruct(matrix_u_list, sigma_list, matrix_v, k):
    V_k = matrix_v[:, :k]
    tensor_rec = []
    for U, S in zip(matrix_u_list, sigma_list):
        U_k = U[:, :k]
        S_k = S[:k, :k]
        A_rec = U_k @ S_k @ V_k.T
        tensor_rec.append(A_rec)
    return np.array(tensor_rec)
"""
    Compress an RGB image using High-Order Singular Value Decomposition (HOSVD) and reconstruct it
    by keeping only the top k components for each mode.

    Inputs:
        image_name (str): Filename of the RGB image located in the "assets" subfolder relative to the script.
        k (int): Number of components to retain in the reconstruction (controls compression level).

    Process:
        1. Load the image and split it into its R, G, B channels.
        2. Convert each channel into a float64 numpy array and combine into a 3D tensor of shape (3, height, width).
        3. Perform HOSVD on the tensor using the provided function `high_Order_SVD`, obtaining:
            - matrix_u_list: List of factor matrices (U matrices for each mode).
            - sigma_list: List of core tensors (singular values matrices).
            - matrix_v: Matrix V (shared across modes).
        4. Reconstruct the tensor using the top k components by calling `hosvd_reconstruct`.
        5. Clip reconstructed pixel values to valid [0,255] range and convert back to uint8.
        6. Convert reconstructed tensor channels back into PIL images and merge into an RGB image.
        7. Plot the original and the compressed-reconstructed image side by side.

    Outputs:
        - Displays a matplotlib figure comparing the original image and the compressed-reconstructed image.

    Dependencies:
        - numpy as np
        - matplotlib.pyplot as plt
        - PIL.Image for image loading and processing
        - SVD_Tool_Kit with `high_Order_SVD` function
        - A function `hosvd_reconstruct` which reconstructs the tensor from the HOSVD factors and k components
        - os module for path handling
    """
def test_compre_hosvd(image_name,k):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "assets", image_name)
    img = Image.open(img_path).convert('RGB')
    r, g, b = img.split()
    R = np.asarray(r, dtype=np.float64)
    G = np.asarray(g, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    tensor = np.array([R, G, B])
    matrix_u_list, sigma_list, matrix_v = SVD_Tool_Kit.high_Order_SVD(tensor)
    tensor_rec = hosvd_reconstruct(matrix_u_list, sigma_list, matrix_v, k)
    tensor_rec = np.clip(tensor_rec, 0, 255).astype(np.uint8)
    r_rec, g_rec, b_rec = [Image.fromarray(tensor_rec[i]) for i in range(3)]
    img_rec = Image.merge("RGB", (r_rec, g_rec, b_rec))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title(f"Compressed HOSVD (k={k})")
    plt.imshow(img_rec)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
"""
    Perform anomaly detection on a color image using tensor Singular Value Decomposition (t-SVD).

    This function loads an image, converts it to a tensor (3D array) with dimensions corresponding
    to color channels and spatial dimensions, applies tensor SVD (t-SVD), performs truncated
    reconstruction by keeping only the top singular values in the Fourier domain, and calculates
    an error map highlighting potential anomalies based on reconstruction differences.

    Inputs:
        image_name (str): Filename of the image located in the "assets" subfolder relative to the script.

    Process:
        1. Load and resize the image to 128x128 pixels.
        2. Convert the image to a tensor of shape (channels, height, width) with float64 precision.
        3. Apply the tensor SVD method (T_SVD) to decompose into U, S, and V.
        4. Transform U, S, and V to the Fourier domain along the first dimension.
        5. Truncate singular values S in the Fourier domain by zeroing out smaller components beyond rank r=30.
        6. Reconstruct the tensor in the Fourier domain using truncated S.
        7. Inverse FFT to return to the spatial domain and clip pixel values to [0,255].
        8. Calculate absolute reconstruction error tensor and reduce to an error map by averaging across color channels.
        9. Plot the original image, the reconstructed image, and the heatmap of the anomaly error map.

    Outputs:
        - Displays plots of the original image, reconstructed image, and anomaly map.
        - No return value.

    Dependencies:
        - numpy as np
        - matplotlib.pyplot as plt
        - PIL.Image for image loading
        - SVD_Tool_Kit with T_SVD function for tensor SVD decomposition
        - os module for path handling
    """
def test_anomalous_tensorSVD(image_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "assets", image_name)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))  # Redimensionar si es necesario
    tensor_img = np.transpose(np.array(img), (2, 0, 1)).astype(np.float64)
    U, S, V = SVD_Tool_Kit.T_SVD(tensor_img)
    r = 30
    k, m, n = S.shape
    U_fou = np.fft.fft(U, axis=0)
    S_fou = np.fft.fft(S, axis=0)
    V_fou = np.fft.fft(V, axis=0)
    S_fou_trunc = np.copy(S_fou)
    S_fou_trunc[:, r:, :] = 0
    S_fou_trunc[:, :, r:] = 0
    recon_fou = np.zeros((k, m, n), dtype=complex)
    for i in range(k):
        recon_fou[i] = U_fou[i, :, :r] @ S_fou_trunc[i, :r, :r] @ V_fou[i, :, :r].conj().T
    recon = np.fft.ifft(recon_fou, axis=0).real
    tensor_recon = np.clip(recon, 0, 255).astype(np.uint8)
    img_recon = np.transpose(tensor_recon, (1, 2, 0))
    error = np.abs(tensor_img - recon)
    error_map = np.mean(error, axis=0)  
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.array(img).astype(np.uint8))
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(img_recon)
    axs[1].set_title(f"Reconstruction (r = {r})")
    axs[1].axis('off')
    im = axs[2].imshow(error_map, cmap='hot')
    axs[2].set_title("Anomalies Map")
    axs[2].axis('off')
    plt.colorbar(im, ax=axs[2], shrink=0.7)
    plt.tight_layout()
    plt.show()
"""
    Test function for the joint SVD algorithm using synthetic data.

    This function generates a set of K synthetic matrices that share a common
    underlying structure defined by orthogonal bases U and V but differ in
    diagonal singular value matrices with added noise. It then applies the
    joint SVD algorithm to find a common decomposition (U, Dk, V) that
    approximates all matrices simultaneously.

    The function prints convergence information, the estimated diagonal matrices,
    reconstruction errors for each matrix, and plots the original and reconstructed
    version of the first matrix for visual comparison.

    Inputs:
        None (all parameters are internally defined)

    Process:
        1. Fix random seed for reproducibility.
        2. Define K (number of matrices), and their dimensions (m x n).
        3. Generate true orthogonal bases U_true and V_true.
        4. Create K diagonal singular value matrices D_true with random positive values.
        5. Construct each matrix in matrix_set as U_true @ D_true[k] @ V_true.T plus small noise.
        6. Call the joint SVD function (join_SVD) to find U, Dk, V.
        7. Print convergence details and diagonal matrices Dk.
        8. Compute and print Frobenius norm reconstruction errors for each matrix.
        9. Plot the original and reconstructed first matrix.

    Returns:
        None (prints outputs and shows plots)

    Dependencies:
        - numpy as np
        - matplotlib.pyplot as plt
        - join_SVD function from SVD_Tool_Kit module (or local scope)
    """
def test_data_analisys_joint_svd():
    np.random.seed(42)
    K = 5
    m = 5
    n = 5
    U_true = np.linalg.qr(np.random.randn(m,m))[0]
    V_true = np.linalg.qr(np.random.randn(n,n))[0]
    D_true = np.array([np.diag(np.random.uniform(1,5,m)) for _ in range(K)])
    matrix_set = np.zeros((K,m,n))
    for k in range(K):
        noise = 0.1 * np.random.randn(m,n)  # ruido pequeño
        matrix_set[k] = U_true @ D_true[k] @ V_true.T + noise
    U,Dk,V,err,iteraciones,val_final = SVD_Tool_Kit.join_SVD(matrix_set)
    print(f"Join SVD convergió en {iteraciones} iteraciones con error {err:.4e}")
    print("Matrices diagonales Dk encontradas:")
    for i, D in enumerate(Dk):
        print(f"D[{i}]:\n", D)
    for k in range(K):
        rec = U @ Dk[k] @ V.T
        err_fro = np.linalg.norm(matrix_set[k] - rec, 'fro')
        print(f"Error Frobenius matriz {k}: {err_fro:.4f}")
    plt.subplot(1,2,1)
    plt.title("Original Matriz 0")
    plt.imshow(matrix_set[0], cmap='viridis')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.title("Reconstrucción Matriz 0")
    plt.imshow(U @ Dk[0] @ V.T, cmap='viridis')
    plt.colorbar()
    plt.show()
"""
You can use this examples or try with your owns
test_anomalous_tensorSVD("images_2.jpeg")
test_compre_hosvd("images_2.jpeg",170)
test_data_analisys_joint_svd()
compressImage_copact_SVD('grayscale_img.jpg',410)
test_img_GSVD("images_2.jpeg","images_2_cop.jpg")
"""