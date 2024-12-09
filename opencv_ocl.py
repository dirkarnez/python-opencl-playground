
import cv2
import time

# CPU 版本的 Canny 邊緣檢測
def canny_edge_cpu(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(image, 60, 150)

# CUDA 版本的 Canny 邊緣檢測，使用 opencv_cudaimgproc 模組
def canny_edge_cuda(image):
    gpu_img = cv2.cuda_GpuMat()
    
    # 如果需要，將圖片轉換為灰階
    if len(image.shape) == 3:  # 檢查圖片是否為彩色 (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 將圖片上傳到 GPU 記憶體
    gpu_img.upload(image)
    
    # 創建 CUDA 版本的 Canny 邊緣檢測器
    canny = cv2.cuda.createCannyEdgeDetector(60, 150)
    
    # 執行 Canny 邊緣檢測
    edges_gpu = canny.detect(gpu_img)
    
    # 將結果從 GPU 下載回 CPU
    result = edges_gpu.download()
    
    return result

cv2.ocl.setUseOpenCL(False)
print("Has OpenCL been Enabled?: ", end='')
print(cv2.ocl.useOpenCL())
# 載入圖片
image = cv2.imread('C:/Users/Administrator/Downloads/resize.png')
# 測量 CPU 版本 Canny 邊緣檢測的時間
start_time = time.time()
edges_cpu = canny_edge_cpu(image)
cpu_time = time.time() - start_time
print(f"CPU Canny 邊緣檢測時間：{cpu_time} 秒")

# 測量 CUDA 版本 Canny 邊緣檢測的時間

# edges_cuda = canny_edge_cuda(image)
# cuda_time = time.time() - start_time
# print(f"CUDA Canny 邊緣檢測時間：{cuda_time} 秒")

print("=============================\n")
def tryOpenCL():
    try:
        # Returns True if OpenCL is present
        ocl = cv2.ocl.haveOpenCL()
        # Prints whether OpenCL is present
        print("OpenCL Supported?: ", end='')
        print(ocl)
        print()
        # Enables use of OpenCL by OpenCV if present
        if ocl == True:
            print('Now enabling OpenCL support')
            cv2.ocl.setUseOpenCL(True)
            print("Has OpenCL been Enabled?: ", end='')
            print(cv2.ocl.useOpenCL())

    except cv2.error as e:
        print('Error:')
        
tryOpenCL()
start_time = time.time()
edges_cpu = canny_edge_cpu(image)
cpu_time = time.time() - start_time
print(f"CPU with OpenCL Canny 邊緣檢測時間：{cpu_time} 秒")

# 保存結果以進行比較
cv2.imwrite('edges_cpu.jpg', edges_cpu)
# cv2.imwrite('edges_cuda.jpg', edges_cuda)