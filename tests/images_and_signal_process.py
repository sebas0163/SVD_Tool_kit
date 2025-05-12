from PIL import Image
import os
import numpy as np
import SVD_Tool_Kit
import matplotlib.pyplot as plt

def compressImage_copact_SVD(image_name,k):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, "assets", image_name)
    img = Image.open(img_path).convert('L')
    mat = np.array(img, dtype=np.float64)
    plt.imshow(mat, cmap='gray')
    plt.title("Imagen original")
    plt.axis('off')
    plt.show()
    u,s,v = SVD_Tool_Kit.svd(mat, False)
    u = u[:, :k]
    s = s[:k, :k]
    v = v[:, :k]
    reconst = u@s@v.T
    plt.imshow(reconst, cmap='gray')
    plt.title(f"Reconstrucción con k={k}")
    plt.axis('off')
    plt.show()
compressImage_copact_SVD('grayscale_img.jpg',410)
