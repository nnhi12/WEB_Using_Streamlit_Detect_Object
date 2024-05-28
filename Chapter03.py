import numpy as np
import cv2

L = 256

def Negative(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        # Convert color image to grayscale
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            s = L - 1 - r
            imgout[x, y] = s
    return imgout

def Logarit(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L - 1) / np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c * np.log(1 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

# Add similar modifications for the remaining functions...

# (omitting Power function for brevity)
def Power(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    gama = 5.0
    c = np.power(L - 1.0, 1.0 - gama)

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x,y]
            if r == 0:
                r = 1
            s = c*np.power(1.0*r, gama)
            imgout[x,y] = np.uint8(s)
    return imgout 


def PiecewiseLinear(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    rmin, rmax, _, _ = cv2.minMaxLoc(imgin)
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r < r1:
                s = s1 / r1 * r
            elif r < r2:
                s = (s2 - s1) / (r2 - r1) * (r - r1) + s1
            else:
                s = (L - 1 - s2) / (L - 1 - r2) * (r - r2) + s2
            imgout[x, y] = np.uint8(s)
    return imgout

# (omitting Histogram function for brevity)
def Histogram(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M,L), np.uint8) + 255
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range (0, N):
            r = imgin[x,y]
            h[r] = h[r] + 1
    p =h/(M*N)
    scale = 3000
    for r in range(0 , L):
        cv2.line(imgout, (r, M-1), (r, M-1-int(scale*p[r])), (0,0,0))
    return imgout

def HistEqual(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    h = np.zeros(L, np.int32)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            h[r] = h[r] + 1
    p = h / (M * N)

    s = np.zeros(L, np.float64)
    for k in range(0, L):
        for j in range(0, k + 1):
            s[k] = s[k] + p[j]

    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    return imgout

# (omitting HistEqualColor, LocalHist, HistStat, MyBoxFilter, BoxFilter, and Threshold functions for brevity)
def LocalHist(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) + 255
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)

    a = m//2
    b = n//2
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s  in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a, t+b] = imgin[x+s, y+t]
            w = cv2.equalizeHist(w)
            imgout[x, y] = w[a,b]
    return imgout

def HistStat(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8) + 255
    '''sum = 0.0
    for x in range(0, M):
        for y in range(0, N):
            sum = sum + imgin[x,y]
    mean = sum/(M*N)
    variance = 0.0
    for x in range(0, M):
        for y in range(0,N):
            variance = variance + (imgin[x,y] - mean)**2
    variance = variance/(M*N)
    std_dev = np.sqrt(variance)
    print('trung bình: ', mean)
    print('do lech chuan: ', std_dev)'''
    mean, std_dev = cv2.meanStdDev(imgin)
    mG = mean[0,0]
    sigmaG = std_dev[0,0]
    m = 3
    n = 3
    w = np.zeros((m,n), np.uint8)

    a = m//2
    b = n//2
    C=22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1
    for x in range(a, M-a):
        for y in range(b, N-b):
            for s  in range(-a, a+1):
                for t in range(-b, b+1):
                    w[s+a, t+b] = imgin[x+s, y+t]
            mean, std_dev = cv2.meanStdDev(w)
            msxy = mean[0,0]
            sigmasxy = std_dev[0,0]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):
                imgout[x,y] = np.uint8(C*imgin[x,y])
            else:
                imgout[x,y] = imgin[x,y]
    return imgout

def Smooth_box(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    m = 35
    n = 35
    w = np.ones((m,n), np.float32)/(m*n)
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout
def Smooth_gauss(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    m = 43
    n = 43
    sigma = 7.0

    #Tạo bộ lọc Gauss
    a = m // 2
    b = n // 2
    w = np.zeros((m,n), np.float32)
    for s in range(-a, a+1):
        for t in range(-b, b+1):
            w[s+a, t+b] = np.exp(-(s**2 + t**2)/(2*sigma**2))
    K = np.sum(w)
    w = w/K
    imgout  = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout

def Median_filter(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8) + 255
    m = 5 
    n = 5
    w = np.zeros((m, n), np.uint8)
    a = m // 2
    b = n // 2
    for x in range(0, M):
        for y in range(0, N):
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    x_moi = x+s
                    y_moi = y+t
                    x_moi = x_moi % M
                    y_moi = y_moi % N
                    w[s+a, t+b] = imgin[x_moi, y_moi]

            w_moi = np.reshape(w, (m*n,))
            w_moi = np.sort(w_moi)
            imgout[x, y] = w_moi[m*n//2] 

    return imgout
# Update the rest of the functions similarly...

def Sharpen(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    w = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    temp = cv2.filter2D(imgin, cv2.CV_32FC1, w)
    imgout = imgin - temp
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

def Gradient(imgin):
    if len(imgin.shape) == 3 and imgin.shape[2] == 3:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    mygx = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_x)
    mygy = cv2.filter2D(imgin, cv2.CV_32FC1, sobel_y)

    gx = cv2.Sobel(imgin, cv2.CV_32FC1, dx=1, dy=0)
    gy = cv2.Sobel(imgin, cv2.CV_32FC1, dx=0, dy=1)

    imgout = abs(gx) + abs(gy)
    imgout = np.clip(imgout, 0, L - 1)
    imgout = imgout.astype(np.uint8)
    return imgout

def HistEqualColor(imgin):
        B = imgin[:,:,0]
        G = imgin[:,:,1]
        R = imgin[:,:,2]
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        R = cv2.equalizeHist(R)
        imgout = np.array([B, G, R])
        imgout = np.transpose(imgout, axes = [1,2,0]) 
        return imgout


def MyBoxFilter(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M,N), np.uint8)
    m = 11
    n = 11
    w = np.ones((m,n))
    w = w/(m*n)

    a = m // 2
    b = n // 2
    for x in range(a, M-a):
        for y in range(b, M-b):
            r = 0.0
            for s in range(-a, a+1):
                for t in range(-b, b+1):
                    r = r + w[s+a,t+b]*imgin[x+s,y+t]
            imgout[x,y] = np.uint8(r)
    return imgout

def BoxFilter(imgin):
    m = 21
    n = 21
    w = np.ones((m,n))
    w = w/(m*n)
    imgout = cv2.filter2D(imgin,cv2.CV_8UC1,w)
    return imgout

def Threshold(imgin):
    temp = cv2.blur(imgin, (15,15))
    retval, imgout = cv2.threshold(temp,64,255,cv2.THRESH_BINARY)
    return imgout
