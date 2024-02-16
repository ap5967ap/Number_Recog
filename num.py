import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt


def init_parameters():
    w1=np.random.rand(10,784)-0.5
    w2=np.random.rand(10,10)-0.5
    b1=np.random.rand(10,1)-0.5
    b2=np.random.rand(10,1)-0.5
    return w1,b1,w2,b2

def ReLU(a): return np.maximum(a,0)

def softmax(a): return np.exp(a)/sum(np.exp(a))

def forward_propagation(w1,b1,w2,b2,inp):
    z1=w1.dot(inp)+b1
    a1=ReLU(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(z2)
    return z1,a1,z2,a2

def one_hot(x):
    oh=np.zeros((x.size,x.max()+1))
    oh[np.arange(x.size),x]=1
    oh=oh.T
    return oh

def deri_relu(x):return x>0

def back_prop(z1,a1,z2,a2,w2,x,y,m):
    y=one_hot(y)
    dz2=a2-y
    dw2=1/m*dz2.dot(a1.T)
    db2=1/m*np.sum(dz2)
    dz1=w2.T.dot(dz2)*(deri_relu(z1))
    dw1=1/m*dz1.dot(x.T)
    db1=1/m*np.sum(dz1)
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1=w1-alpha*dw1
    b1=b1-alpha*db1    
    w2=w2-alpha*dw2  
    b2=b2-alpha*db2    
    return w1,b1,w2,b2

def get_pred(x):
    return np.argmax(x,0)

def get_correctness(pred, y):
    print(pred,y)
    return np.sum(pred == y)/y.size

def gradient_descent(x,y,alpha,it,m):
    w1,b1,w2,b2=init_parameters()
    for i in range(it):
        z1,a1,z2,a2=forward_propagation(w1,b1,w2,b2,x)
        dW1,db1,dw2,db2=back_prop(z1,a1,z2,a2,w2,x,y,m)
        w1,b1,w2,b2=update_params(w1,b1,w2,b2,dW1,db1,dw2,db2,alpha)
        if i%10==0:
            print("Iteration: ",i)
            predictions=get_pred(a2)
            print(get_correctness(predictions, y))
    return w1,b1,w2,b2


def make_predictions(X,w1,b1,w2,b2):
    _,_,_,a2=forward_propagation(w1,b1,w2,b2,X)
    pred=get_pred(a2)
    return pred

def test_prediction(index,w1,b1,w2,b2,X_train,Y_train):
    current_image=X_train[:,index, None]
    prediction=make_predictions(X_train[:, index, None], w1, b1, w2, b2)
    label=Y_train[index]
    print("Prediction: ",prediction)
    print("Label: ",label)
    
    current_image=current_image.reshape((28, 28))-255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def main():
    data=pd.read_csv('./train.csv')
    data=np.array(data) # 42000x785
    np.random.shuffle(data)
    m,n=data.shape # 42000,785
    data_dev=data[:1000].T # 785x1000
    Y_dev=data_dev[0] # 1x1000
    X_dev=data_dev[1:n] # 784x1000
    X_dev=X_dev/255 # Normalization
    data_train=data[1000:m].T # 785x41000
    Y_train=data_train[0]   # 1x41000
    X_train=data_train[1:n] # 784x41000
    X_train=X_train/255 # Normalization
    _,m_train=X_train.shape 
    w1,b1,w2,b2=gradient_descent(X_train,Y_train,0.10,500,m)
    dev_predictions = make_predictions(X_dev,w1,b1,w2,b2)
    print(get_correctness(dev_predictions,Y_dev))
        
    
if __name__ == "__main__":
    main()