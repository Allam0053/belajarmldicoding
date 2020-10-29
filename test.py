import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# load iris dataset
iris = datasets.load_iris()

# pisahkan atribut dan label pada iris dataset
x=iris.data
y=iris.target

# membagi dataset menjadi training dan testing 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# menghitung panjang/jumlah data pada x_test
len(x_test)
print(len(x_test))