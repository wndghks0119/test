import matplotlib.pyplot as plt
import numpy as np
import random as rd
from sklearn.linear_model import LinearRegression

# 데이터 생성
n = 100
m = 10
epoches = 200
batch_size = 70
alpha = 0.0001
total_iteration = 10000

x_values = np.random.randint(1, 50, size=(n, m))
thetas = np.random.randint(1, 11, size=(m, 1))
y_values = np.dot(x_values,thetas) + np.random.randn(n,1)

# ★★★★★★★★★★★★★★★★★★★★★ 배치학습 적용 ★★★★★★★★★★★★★★★★★★★★★
total_epoch = []
loss_m1 = []
loss_m2 = []
loss_m3 = []
loss_m4 = []
for epoch in range(1,epoches+1):
    # 배치학습을 위한 데이터셋 생성
    random_indices = np.random.choice(n, batch_size, replace=False)
    x_batch = x_values[random_indices]
    y_batch = y_values[random_indices]
    X_batch_big = np.c_[np.ones((batch_size,1)), x_batch]
    print(f'Epoch {epoch}')
    total_epoch.append(epoch)
    
    # 방법 1) 정규방정식을 활용하는 방법
    theta_best = np.linalg.inv(np.dot(X_batch_big.T,X_batch_big)).dot(X_batch_big.T).dot(y_batch)
    print(f'정규방정식 사용 ) 절편 : {theta_best[0,0]}, 파라미터 : {theta_best[1:,0]}')
    y1_x_batch = np.dot(X_batch_big, theta_best)
    error_1 = sum((y1_x_batch - y_batch)**2)/batch_size
    print(f'Loss 1 : {error_1[0]}')
    loss_m1.append(error_1)
    
    # 방법 2) 사이킷런 linear regression 활용하기
    lin_reg = LinearRegression()
    lin_reg.fit(x_batch, y_batch)
    weights = lin_reg.coef_[0].tolist()
    bias = lin_reg.intercept_.tolist()
    print(f'사이킷런 사용 ) 절편 : {bias[0]}, 파라미터 : {weights}')
    thetas_new_list = bias + weights
    thetas_new = np.array(thetas_new_list)
    thetas_new = thetas_new.reshape(m+1,1) # 행렬로 만들어주는 작업
    y2_x_batch = np.dot(X_batch_big, thetas_new)
    error_2 = sum((y2_x_batch - y_batch)**2)/batch_size
    print(f'Loss 2 : {error_2[0]}')
    loss_m2.append(error_2)
    
    # 방법 3) L1 Loss 경사하강법 적용하기
    theta_method3 = 0.5*np.ones((m+1,1))
    for iteration in range(1,total_iteration+1):
        for parameter in range(0,m+1):
            gradients3 = 2 / batch_size * X_batch_big.T.dot(X_batch_big.dot(theta_method3) - y_batch)
            theta_method3[parameter] = theta_method3[parameter] - alpha * gradients3[parameter]   
    
    print(f'L1 경사하강법 사용 ) 절편 : {theta_method3[0,0]}, 파라미터 : {theta_method3[1:,0]}')
    y3_x_batch = np.dot(X_batch_big, theta_method3)
    error_3 = sum((y3_x_batch - y_batch)**2)/batch_size
    print(f'Loss 3 : {error_3[0]}')
    loss_m3.append(error_3)
    
    # 방법 4) L2 Loss 경사하강법 적용하기
    theta_method4 = 0.5*np.ones((m+1,1))
    for iteration in range(1,total_iteration+1):
        gradients4 = 2 / batch_size * X_batch_big.T.dot(X_batch_big.dot(theta_method4) - y_batch)
        theta_method4 = theta_method4 - alpha * gradients4  
    
    print(f'L2 경사하강법 사용 ) 절편 : {theta_method4[0,0]}, 파라미터 : {theta_method4[1:,0]}')
    y4_x_batch = np.dot(X_batch_big, theta_method4)
    error_4 = sum((y4_x_batch - y_batch)**2)/batch_size
    print(f'Loss 4 : {error_4[0]}')
    loss_m4.append(error_4)
    
loss_m1_numpy = np.array(loss_m1)
loss_m2_numpy = np.array(loss_m2)
loss_m1_m2_numpy = np.array(loss_m2) - np.array(loss_m1)
loss_m1_m2 = loss_m1_m2_numpy.tolist()

# 첫 번째 그래프 : 정규방정식 사용
plt.subplot(2, 2, 1)  # 2행 2열 중 첫 번째
plt.plot(total_epoch, loss_m1, color="red")
plt.ylabel('Loss')
plt.title('The Graph of Epoch - Loss : method 1st')
plt.legend()

# 두 번째 그래프 : 사이킷런 사용
plt.subplot(2, 2, 2)  # 2행 2열 중 두 번째
plt.plot(total_epoch, loss_m2, color="green")
plt.title('The Graph of Epoch - Loss : method 2nd')
plt.legend()

# 세 번째 그래프 : L1 경사하강법 사용
plt.subplot(2, 2, 3)  # 2행 2열 중 세 번째
plt.plot(total_epoch, loss_m3, color="blue")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('The Graph of Epoch - Loss : method 3rd')
plt.legend()

# 네 번째 그래프 : L2 경사하강법 사용
plt.subplot(2, 2, 4)  # 2행 2열 중 네 번째
plt.plot(total_epoch, loss_m4, color="purple")
plt.xlabel('Epoch')
plt.title('The Graph of Epoch - Loss : method 4th')
plt.legend()

# 그래프 보여주기
plt.show()
