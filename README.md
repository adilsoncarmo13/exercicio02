
# 8 . Usando NumPy, importe o [dataset iris](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) mantendo o seu conteúdo intacto.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='object', encoding='ascii')
iris[:3]

```




    array([[b'5.1', b'3.5', b'1.4', b'0.2', b'Iris-setosa'],
           [b'4.9', b'3.0', b'1.4', b'0.2', b'Iris-setosa'],
           [b'4.7', b'3.2', b'1.3', b'0.2', b'Iris-setosa']], dtype=object)



# 9 . Extraia a coluna *species* do dataset importado na questão anterior para um vetor unidimensional.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='object', encoding='ascii')
iris[:,4]
```




    array([b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-setosa', b'Iris-setosa',
           b'Iris-setosa', b'Iris-setosa', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-versicolor', b'Iris-versicolor',
           b'Iris-versicolor', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica',
           b'Iris-virginica', b'Iris-virginica', b'Iris-virginica'],
          dtype=object)



# 10 . Calcule a média, mediana e desvio padrão da primeira coluna (sepallength) do dataset Iris.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='float', usecols=[0])

media = np.mean(iris)
mediana = np.median(iris)
desvio = np.std(iris)

print('Média = ' + str(media))
print('Mediana = ' + str(mediana))
print('Desvio Padrão = ' + str(desvio))
```

    Média = 5.843333333333334
    Mediana = 5.8
    Desvio Padrão = 0.8253012917851409
    

# 11 . Normalize a coluna (sepallength) do dataset iris de tal forma que o valor mínimo seja zero (0) e o valor máximo seja 1.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='float', usecols=[0])

minimo = min(iris)
maximo = max(iris)

for i in range(len(iris)):
    iris[i] = (iris[i] - minimo) / (maximo - minimo)

print(iris)
```

    [0.22222222 0.16666667 0.11111111 0.08333333 0.19444444 0.30555556
     0.08333333 0.19444444 0.02777778 0.16666667 0.30555556 0.13888889
     0.13888889 0.         0.41666667 0.38888889 0.30555556 0.22222222
     0.38888889 0.22222222 0.30555556 0.22222222 0.08333333 0.22222222
     0.13888889 0.19444444 0.19444444 0.25       0.25       0.11111111
     0.13888889 0.30555556 0.25       0.33333333 0.16666667 0.19444444
     0.33333333 0.16666667 0.02777778 0.22222222 0.19444444 0.05555556
     0.02777778 0.19444444 0.22222222 0.13888889 0.22222222 0.08333333
     0.27777778 0.19444444 0.75       0.58333333 0.72222222 0.33333333
     0.61111111 0.38888889 0.55555556 0.16666667 0.63888889 0.25
     0.19444444 0.44444444 0.47222222 0.5        0.36111111 0.66666667
     0.36111111 0.41666667 0.52777778 0.36111111 0.44444444 0.5
     0.55555556 0.5        0.58333333 0.63888889 0.69444444 0.66666667
     0.47222222 0.38888889 0.33333333 0.33333333 0.41666667 0.47222222
     0.30555556 0.47222222 0.66666667 0.55555556 0.36111111 0.33333333
     0.33333333 0.5        0.41666667 0.19444444 0.36111111 0.38888889
     0.38888889 0.52777778 0.22222222 0.38888889 0.55555556 0.41666667
     0.77777778 0.55555556 0.61111111 0.91666667 0.16666667 0.83333333
     0.66666667 0.80555556 0.61111111 0.58333333 0.69444444 0.38888889
     0.41666667 0.58333333 0.61111111 0.94444444 0.94444444 0.47222222
     0.72222222 0.36111111 0.94444444 0.55555556 0.66666667 0.80555556
     0.52777778 0.5        0.58333333 0.80555556 0.86111111 1.
     0.58333333 0.55555556 0.5        0.94444444 0.55555556 0.58333333
     0.47222222 0.72222222 0.66666667 0.72222222 0.41666667 0.69444444
     0.66666667 0.66666667 0.55555556 0.61111111 0.52777778 0.44444444]
    

# 12 . Encontre o 5º e o 95º percentil do comprimento das pétalas (sepallength) do dataset Iris.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='float', usecols=[0])

print(np.percentile(iris, 95))
print(np.percentile(iris, 5))
```

    7.254999999999998
    4.6
    

# 13 . Encontre a quantidade de valores ausentes, bem como a posição deles da primeira coluna da coluna comprimento das pétalas (sepallength), dentro do vetor abaixo:


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

lista_pos = []
quant = 0

x = np.isnan(iris_2d[:,0])

for i in range(len(x)):
    if x[i] == True:
        quant += 1
        lista_pos.append(i)
print('Existem ' + str(quant) + ' valores ausentes')
print('Posições de valores ausentes')
print(lista_pos)

```

    Existem 4 valores ausentes
    Posições de valores ausentes
    [36, 46, 120, 143]
    

# 14 . Filtre o vetor iris_2d do exercício anterior para que ele tenha: (terceira coluna (petallength) > 1.5) && (primeira coluna (sepallength) < 5.0)


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float')
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

terceira = iris_2d[:,2]
primeira = iris_2d[:,0]

for i in range(len(iris_2d)):
    if terceira[i] <= 1.5:
        terceira[i] = 0
    if primeira[i] >= 5:
        primeira[i] = 0

print('Terceira')
for i in range(len(iris_2d)):
    if terceira[i] != 0 and np.isnan(terceira[i]) == False:
        print(str(terceira[i]) + ', ')

print('Primeira')
for i in range(len(iris_2d)):
    if primeira[i] != 0 and np.isnan(primeira[i]) == False:
        print(str(primeira[i]) + ', ')
```

    Terceira
    1.7, 
    1.6, 
    1.7, 
    1.7, 
    1.7, 
    1.9, 
    1.6, 
    1.6, 
    1.6, 
    1.6, 
    1.6, 
    1.9, 
    1.6, 
    4.7, 
    4.5, 
    4.9, 
    4.0, 
    4.6, 
    4.5, 
    4.7, 
    3.3, 
    4.6, 
    3.9, 
    3.5, 
    4.0, 
    4.7, 
    3.6, 
    4.4, 
    4.5, 
    4.1, 
    4.5, 
    3.9, 
    4.8, 
    4.0, 
    4.9, 
    4.7, 
    4.3, 
    4.4, 
    4.8, 
    5.0, 
    3.5, 
    3.8, 
    3.7, 
    3.9, 
    5.1, 
    4.5, 
    4.5, 
    4.7, 
    4.4, 
    4.1, 
    4.0, 
    4.4, 
    4.6, 
    4.0, 
    3.3, 
    4.2, 
    4.2, 
    4.2, 
    4.3, 
    3.0, 
    4.1, 
    6.0, 
    5.1, 
    5.9, 
    5.6, 
    5.8, 
    6.6, 
    4.5, 
    6.3, 
    5.8, 
    6.1, 
    5.1, 
    5.3, 
    5.5, 
    5.0, 
    5.1, 
    5.3, 
    5.5, 
    6.7, 
    6.9, 
    5.0, 
    5.7, 
    4.9, 
    6.7, 
    4.9, 
    5.7, 
    6.0, 
    4.8, 
    4.9, 
    5.6, 
    5.8, 
    6.4, 
    5.6, 
    5.1, 
    5.6, 
    6.1, 
    5.6, 
    5.5, 
    4.8, 
    5.4, 
    5.6, 
    5.1, 
    5.9, 
    5.7, 
    5.2, 
    5.0, 
    5.2, 
    5.4, 
    5.1, 
    Primeira
    4.9, 
    4.7, 
    4.6, 
    4.6, 
    4.4, 
    4.9, 
    4.8, 
    4.8, 
    4.3, 
    4.6, 
    4.8, 
    4.7, 
    4.8, 
    4.9, 
    4.9, 
    4.4, 
    4.5, 
    4.4, 
    4.8, 
    4.6, 
    4.9, 
    4.9, 
    

# 15 . Selecione as linhas do vetor iris_2d que não contenham valores nan (Not a Number).


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

for i in range(len(iris_2d)):
    if np.isnan(iris_2d[i]).all() == False:
        print(iris_2d[i])

```

    [5.1 3.5 1.4 0.2]
    [4.9 3.  1.4 0.2]
    [4.7 3.2 1.3 0.2]
    [4.6 3.1 1.5 0.2]
    [5.  3.6 1.4 0.2]
    [5.4 3.9 1.7 0.4]
    [4.6 3.4 1.4 0.3]
    [5.  3.4 1.5 0.2]
    [4.4 2.9 1.4 0.2]
    [4.9 3.1 1.5 0.1]
    [5.4 3.7 1.5 0.2]
    [4.8 3.4 1.6 0.2]
    [4.8 3.  1.4 0.1]
    [4.3 3.  1.1 0.1]
    [5.8 4.  1.2 0.2]
    [5.7 4.4 1.5 0.4]
    [5.4 3.9 1.3 0.4]
    [5.1 3.5 1.4 0.3]
    [5.7 3.8 1.7 0.3]
    [5.1 3.8 1.5 0.3]
    [5.4 3.4 1.7 0.2]
    [5.1 3.7 1.5 0.4]
    [4.6 3.6 1.  0.2]
    [5.1 3.3 1.7 0.5]
    [4.8 3.4 1.9 0.2]
    [5.  3.  1.6 0.2]
    [5.  3.4 1.6 0.4]
    [5.2 3.5 1.5 0.2]
    [5.2 3.4 1.4 0.2]
    [4.7 3.2 1.6 0.2]
    [4.8 3.1 1.6 0.2]
    [5.4 3.4 1.5 0.4]
    [5.2 4.1 1.5 0.1]
    [5.5 4.2 1.4 0.2]
    [4.9 3.1 1.5 0.1]
    [5.  3.2 1.2 0.2]
    [5.5 3.5 1.3 0.2]
    [4.9 3.1 1.5 0.1]
    [4.4 3.  1.3 0.2]
    [5.1 3.4 1.5 0.2]
    [5.  3.5 1.3 0.3]
    [4.5 2.3 1.3 0.3]
    [4.4 3.2 1.3 0.2]
    [5.  3.5 1.6 0.6]
    [5.1 3.8 1.9 0.4]
    [4.8 3.  1.4 0.3]
    [5.1 3.8 1.6 0.2]
    [4.6 3.2 1.4 0.2]
    [5.3 3.7 1.5 0.2]
    [5.  3.3 1.4 0.2]
    [7.  3.2 4.7 1.4]
    [6.4 3.2 4.5 1.5]
    [6.9 3.1 4.9 1.5]
    [5.5 2.3 4.  1.3]
    [6.5 2.8 4.6 1.5]
    [5.7 2.8 4.5 1.3]
    [6.3 3.3 4.7 1.6]
    [4.9 2.4 3.3 1. ]
    [6.6 2.9 4.6 1.3]
    [5.2 2.7 3.9 1.4]
    [5.  2.  3.5 1. ]
    [5.9 3.  4.2 1.5]
    [6.  2.2 4.  1. ]
    [6.1 2.9 4.7 1.4]
    [5.6 2.9 3.6 1.3]
    [6.7 3.1 4.4 1.4]
    [5.6 3.  4.5 1.5]
    [5.8 2.7 4.1 1. ]
    [6.2 2.2 4.5 1.5]
    [5.6 2.5 3.9 1.1]
    [5.9 3.2 4.8 1.8]
    [6.1 2.8 4.  1.3]
    [6.3 2.5 4.9 1.5]
    [6.1 2.8 4.7 1.2]
    [6.4 2.9 4.3 1.3]
    [6.6 3.  4.4 1.4]
    [6.8 2.8 4.8 1.4]
    [6.7 3.  5.  1.7]
    [6.  2.9 4.5 1.5]
    [5.7 2.6 3.5 1. ]
    [5.5 2.4 3.8 1.1]
    [5.5 2.4 3.7 1. ]
    [5.8 2.7 3.9 1.2]
    [6.  2.7 5.1 1.6]
    [5.4 3.  4.5 1.5]
    [6.  3.4 4.5 1.6]
    [6.7 3.1 4.7 1.5]
    [6.3 2.3 4.4 1.3]
    [5.6 3.  4.1 1.3]
    [5.5 2.5 4.  1.3]
    [5.5 2.6 4.4 1.2]
    [6.1 3.  4.6 1.4]
    [5.8 2.6 4.  1.2]
    [5.  2.3 3.3 1. ]
    [5.6 2.7 4.2 1.3]
    [5.7 3.  4.2 1.2]
    [5.7 2.9 4.2 1.3]
    [6.2 2.9 4.3 1.3]
    [5.1 2.5 3.  1.1]
    [5.7 2.8 4.1 1.3]
    [6.3 3.3 6.  2.5]
    [5.8 2.7 5.1 1.9]
    [7.1 3.  5.9 2.1]
    [6.3 2.9 5.6 1.8]
    [6.5 3.  5.8 2.2]
    [7.6 3.  6.6 2.1]
    [4.9 2.5 4.5 1.7]
    [7.3 2.9 6.3 1.8]
    [6.7 2.5 5.8 1.8]
    [7.2 3.6 6.1 2.5]
    [6.5 3.2 5.1 2. ]
    [6.4 2.7 5.3 1.9]
    [6.8 3.  5.5 2.1]
    [5.7 2.5 5.  2. ]
    [5.8 2.8 5.1 2.4]
    [6.4 3.2 5.3 2.3]
    [6.5 3.  5.5 1.8]
    [7.7 3.8 6.7 2.2]
    [7.7 2.6 6.9 2.3]
    [6.  2.2 5.  1.5]
    [6.9 3.2 5.7 2.3]
    [5.6 2.8 4.9 2. ]
    [7.7 2.8 6.7 2. ]
    [6.3 2.7 4.9 1.8]
    [6.7 3.3 5.7 2.1]
    [7.2 3.2 6.  1.8]
    [6.2 2.8 4.8 1.8]
    [6.1 3.  4.9 1.8]
    [6.4 2.8 5.6 2.1]
    [7.2 3.  5.8 1.6]
    [7.4 2.8 6.1 1.9]
    [7.9 3.8 6.4 2. ]
    [6.4 2.8 5.6 2.2]
    [6.3 2.8 5.1 1.5]
    [6.1 2.6 5.6 1.4]
    [7.7 3.  6.1 2.3]
    [6.3 3.4 5.6 2.4]
    [6.4 3.1 5.5 1.8]
    [6.  3.  4.8 1.8]
    [6.9 3.1 5.4 2.1]
    [6.7 3.1 5.6 2.4]
    [6.9 3.1 5.1 2.3]
    [5.8 2.7 5.1 1.9]
    [6.8 3.2 5.9 2.3]
    [6.7 3.3 5.7 2.5]
    [6.7 3.  5.2 2.3]
    [6.3 2.5 5.  1.9]
    [6.5 3.  5.2 2. ]
    [6.2 3.4 5.4 2.3]
    [5.9 3.  5.1 1.8]
    

# 16 . Encontre a correlação entre a primeira coluna (sepallength) e a (terceira coluna (petallength) no vetor iris_2d.


```python
import numpy as np

data_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(data_url, delimiter=',', dtype='float', usecols=[0,2])

cor = np.corrcoef(iris[:,0], iris[:,1])

print(cor)
```

    [[1.         0.87175416]
     [0.87175416 1.        ]]
    

# 17 . Verifique se o vetor iris_2d possui qualquer valor faltante, caso possui deverá retornar True caso contrário False.


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

x = np.isnan(iris_2d)

if np.isin(False, iris_2d):
    print('Verdadeiro')
else:
    print('Falso')

```

    Falso
    

# 18 . Substitua todas as ocorrências de nan (Not a Number) por zeros em um vetor NumPy.


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_2d = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])
iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan

for i in range(len(iris_2d)):
    for j in range(4):
        if np.isnan(iris_2d[i,j]).any() == True:
            iris_2d[i,j] = 0

print(iris_2d)
```

    [[5.1 3.5 1.4 0.2]
     [4.9 3.  1.4 0.2]
     [4.7 3.2 1.3 0.2]
     [4.6 3.1 1.5 0.2]
     [5.  3.6 1.4 0.2]
     [5.4 3.9 1.7 0.4]
     [4.6 3.4 1.4 0.3]
     [5.  3.4 0.  0.2]
     [4.4 2.9 1.4 0.2]
     [4.9 3.1 1.5 0.1]
     [5.4 3.7 1.5 0.2]
     [4.8 3.4 1.6 0.2]
     [4.8 3.  1.4 0.1]
     [4.3 3.  1.1 0.1]
     [5.8 4.  1.2 0.2]
     [5.7 4.4 1.5 0. ]
     [5.4 3.9 1.3 0.4]
     [5.1 3.5 1.4 0.3]
     [5.7 3.8 1.7 0.3]
     [5.1 3.8 1.5 0.3]
     [5.4 3.4 1.7 0.2]
     [5.1 3.7 1.5 0.4]
     [4.6 3.6 1.  0.2]
     [5.1 3.3 1.7 0.5]
     [4.8 3.4 0.  0.2]
     [5.  3.  1.6 0.2]
     [5.  3.4 1.6 0.4]
     [5.2 3.5 1.5 0.2]
     [5.2 3.4 1.4 0.2]
     [4.7 3.2 1.6 0.2]
     [4.8 3.1 1.6 0.2]
     [5.4 3.4 1.5 0.4]
     [5.2 4.1 1.5 0.1]
     [5.5 4.2 1.4 0.2]
     [4.9 3.1 1.5 0.1]
     [5.  3.2 1.2 0.2]
     [5.5 3.5 1.3 0.2]
     [4.9 3.1 1.5 0.1]
     [4.4 3.  1.3 0.2]
     [5.1 3.4 1.5 0.2]
     [0.  3.5 1.3 0.3]
     [4.5 2.3 1.3 0.3]
     [4.4 3.2 1.3 0.2]
     [5.  3.5 1.6 0.6]
     [5.1 3.8 1.9 0.4]
     [4.8 3.  1.4 0.3]
     [5.1 3.8 1.6 0.2]
     [4.6 0.  1.4 0.2]
     [5.3 3.7 1.5 0.2]
     [5.  3.3 1.4 0.2]
     [7.  3.2 4.7 1.4]
     [6.4 3.2 4.5 1.5]
     [6.9 3.1 4.9 1.5]
     [5.5 2.3 4.  1.3]
     [6.5 2.8 4.6 1.5]
     [5.7 2.8 4.5 1.3]
     [6.3 3.3 4.7 1.6]
     [4.9 2.4 3.3 0. ]
     [6.6 2.9 4.6 1.3]
     [5.2 2.7 3.9 1.4]
     [5.  2.  0.  1. ]
     [5.9 3.  0.  1.5]
     [6.  2.2 4.  0. ]
     [6.1 2.9 4.7 0. ]
     [5.6 2.9 3.6 1.3]
     [6.7 3.1 4.4 1.4]
     [5.6 3.  4.5 1.5]
     [5.8 2.7 4.1 1. ]
     [6.2 2.2 4.5 1.5]
     [5.6 2.5 3.9 1.1]
     [5.9 3.2 4.8 1.8]
     [6.1 0.  4.  1.3]
     [6.3 2.5 4.9 1.5]
     [6.1 2.8 4.7 1.2]
     [0.  2.9 4.3 1.3]
     [6.6 3.  4.4 1.4]
     [6.8 2.8 4.8 1.4]
     [6.7 3.  5.  1.7]
     [6.  2.9 4.5 1.5]
     [5.7 2.6 3.5 0. ]
     [5.5 2.4 3.8 0. ]
     [5.5 2.4 3.7 1. ]
     [5.8 2.7 3.9 1.2]
     [6.  2.7 5.1 1.6]
     [5.4 3.  4.5 1.5]
     [6.  3.4 4.5 1.6]
     [6.7 3.1 4.7 1.5]
     [6.3 2.3 4.4 1.3]
     [5.6 3.  4.1 1.3]
     [5.5 2.5 4.  1.3]
     [5.5 2.6 4.4 1.2]
     [6.1 3.  4.6 0. ]
     [5.8 2.6 4.  1.2]
     [5.  2.3 3.3 1. ]
     [5.6 2.7 4.2 1.3]
     [5.7 3.  4.2 1.2]
     [5.7 2.9 4.2 1.3]
     [6.2 2.9 4.3 1.3]
     [5.1 2.5 3.  1.1]
     [5.7 2.8 4.1 1.3]
     [6.3 3.3 6.  2.5]
     [5.8 2.7 5.1 0. ]
     [7.1 3.  5.9 2.1]
     [6.3 2.9 5.6 1.8]
     [6.5 3.  5.8 2.2]
     [7.6 3.  0.  2.1]
     [4.9 2.5 4.5 1.7]
     [7.3 2.9 6.3 1.8]
     [6.7 2.5 5.8 1.8]
     [7.2 3.6 6.1 2.5]
     [6.5 3.2 5.1 2. ]
     [6.4 2.7 5.3 1.9]
     [6.8 3.  5.5 2.1]
     [5.7 2.5 5.  2. ]
     [5.8 2.8 5.1 2.4]
     [6.4 3.2 5.3 2.3]
     [6.5 3.  5.5 1.8]
     [7.7 3.8 6.7 2.2]
     [7.7 2.6 6.9 2.3]
     [6.  2.2 5.  1.5]
     [6.9 3.2 5.7 2.3]
     [0.  2.8 4.9 2. ]
     [7.7 2.8 6.7 2. ]
     [6.3 2.7 4.9 1.8]
     [6.7 3.3 5.7 2.1]
     [7.2 3.2 6.  1.8]
     [6.2 2.8 4.8 1.8]
     [6.1 3.  4.9 1.8]
     [6.4 2.8 5.6 2.1]
     [7.2 3.  5.8 1.6]
     [7.4 2.8 6.1 1.9]
     [7.9 3.8 6.4 2. ]
     [6.4 2.8 5.6 2.2]
     [6.3 2.8 5.1 1.5]
     [6.1 2.6 5.6 1.4]
     [7.7 3.  6.1 2.3]
     [6.3 3.4 5.6 2.4]
     [6.4 3.1 5.5 1.8]
     [6.  3.  4.8 1.8]
     [6.9 3.1 5.4 2.1]
     [6.7 0.  5.6 2.4]
     [6.9 3.1 5.1 2.3]
     [5.8 2.7 5.1 1.9]
     [6.8 3.2 5.9 0. ]
     [6.7 3.3 5.7 2.5]
     [6.7 3.  5.2 2.3]
     [6.3 2.5 5.  1.9]
     [6.5 3.  5.2 2. ]
     [6.2 3.4 5.4 2.3]
     [5.9 3.  5.1 1.8]]
    

# 19 . Encontre e conte os valores únicos da coluna species


```python
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(url, delimiter=',', dtype='str', usecols=[4])

array_nome = []

for i in range(len(iris)):
    if array_nome.count(iris[i]) == 0:
        array_nome.append(iris[i])

print(array_nome)
```

    ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    

# 20 . Mapeie terceira coluna (petallength) de tal forma que os valores numéricos sejam substituídos por valores textuais, seguindo de acordo com a tabela abaixo:

| Número               | Classificação                
|:---------------------|:--------------|
| Menor do que 3       | Pequena       |
| entre 3 e 5          | Média         |
| Maior ou Igual a 5   | Grande        |


```python
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2])

iris_nova = []

for i in range(len(iris)):
    if iris[i] < 3:
        iris_nova.append('Pequena')
    elif iris[i] >= 3 and iris[i] < 5:
        iris_nova.append('Média')
    else:
        iris_nova.append('Grande')

print(iris_nova)
        
```

    ['Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Pequena', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Grande', 'Média', 'Média', 'Média', 'Média', 'Média', 'Grande', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Média', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Média', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Média', 'Grande', 'Média', 'Grande', 'Grande', 'Média', 'Média', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Média', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande', 'Grande']
    

# 21 . Crie uma nova coluna chamada *volume*, preenchida com dados resultantes da seguinte equação:
    $$volume=\frac{(~ Pi ~x~ Petallength ~x~ Sepal Length^2 ~)}{3}$$


```python
import numpy as np
import math

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0,1,2,3])

volume = []

for i in range(len(iris[:,0])):
    volume.append((math.pi * iris[i, 2] * iris[i, 0] ** 2) / 3)

print(volume)

```

    [38.13265162927291, 35.200498485922445, 30.0723720777127, 33.238050274980004, 36.65191429188092, 51.911677007917746, 31.022180256648003, 39.269908169872416, 28.38324242763259, 37.714819806345474, 45.80442088933919, 38.60389052731138, 33.77840421139745, 21.298950993787603, 42.273270746704256, 51.03517265756619, 39.69716477076063, 38.13265162927291, 57.83986234524169, 40.85641245993526, 51.911677007917746, 40.85641245993526, 22.158700183320004, 46.303934121259964, 45.84212000118226, 41.88790204786391, 41.88790204786391, 42.47433267653401, 39.6427104980984, 37.01215024949256, 38.60389052731138, 45.80442088933919, 42.47433267653401, 44.34881629317591, 37.714819806345474, 31.415926535897928, 41.18104370080621, 37.714819806345474, 26.355867968515977, 40.85641245993526, 34.033920413889426, 27.567475535250434, 26.355867968515977, 41.88790204786391, 51.75145578258466, 33.77840421139745, 43.58017329059761, 31.022180256648003, 44.123668819668644, 36.65191429188092, 241.16959604057647, 193.01945263655693, 244.29966952110314, 126.71090369478833, 203.52284407505874, 153.10551797269858, 195.34737279286693, 82.97260357396004, 209.8332565185694, 110.43326495898843, 91.62978572970229, 153.102376380045, 150.79644737231007, 183.14123813611934, 118.22441473989107, 206.83827152214724, 147.78051842486386, 144.43367505123953, 181.1442324059875, 128.07644930154868, 174.97414443433715, 155.86488352010159, 203.66002695426553, 183.14123813611934, 184.4408102971544, 200.71007145254472, 232.42759088318724, 235.04349036607638, 169.64600329384882, 119.08206953432112, 120.3753585100489, 117.2075859176792, 137.38812992678882, 192.26547039969532, 137.41326266801755, 169.64600329384882, 220.94088094411177, 182.87839155076904, 134.6444723426537, 126.71090369478833, 139.38199406426716, 179.2446160481168, 140.9109024890142, 86.3937979737193, 137.92848386320625, 142.89848344118533, 142.89848344118533, 173.09337763238804, 81.71282491987051, 139.49613859734757, 249.37962484195774, 179.66140067349306, 311.4564484793409, 232.7543165191606, 256.61575992072625, 399.20846167696214, 113.14445941903642, 351.57249227058014, 272.65044882464855, 331.1489984295929, 225.64489234408688, 227.33402199416705, 266.3232812203187, 170.1172421918873, 179.66140067349306, 227.33402199416705, 243.34253095930936, 415.99189682999014, 428.4095653920794, 188.4955592153876, 284.1853298510792, 160.91656450707399, 415.99189682999014, 203.66002695426553, 267.94957901732704, 325.7203263241898, 193.22051456638667, 190.93448231212446, 240.20198550327083, 314.86298211338345, 349.8016812115067, 418.27583468914986, 240.20198550327083, 211.97268111566407, 218.2108369281422, 378.73889114372236, 232.7543165191606, 235.91266433356955, 180.95573684677208, 269.22820722733815, 263.24870921000553, 254.27108460359707, 179.66140067349306, 285.692247127251, 267.94957901732704, 244.4452299807194, 207.8163540349648, 230.0693019978925, 217.373078887185, 185.9100284614832]
    

# 22 . Usando NumPy e o dataset iris responda: Qual é o segundo maior valor de comprimento de pétala (petallength) da espécie setosa?


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2])

array = np.sort(iris)

print(array[len(array)-2])
```

    6.7
    

# 23 . Ordene o dataset iris com base na coluna sepallength de maneira crescente.


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[0])

array = np.sort(iris)

print(array)
```

    [4.3 4.4 4.4 4.4 4.5 4.6 4.6 4.6 4.6 4.7 4.7 4.8 4.8 4.8 4.8 4.8 4.9 4.9
     4.9 4.9 4.9 4.9 5.  5.  5.  5.  5.  5.  5.  5.  5.  5.  5.1 5.1 5.1 5.1
     5.1 5.1 5.1 5.1 5.1 5.2 5.2 5.2 5.2 5.3 5.4 5.4 5.4 5.4 5.4 5.4 5.5 5.5
     5.5 5.5 5.5 5.5 5.5 5.6 5.6 5.6 5.6 5.6 5.6 5.7 5.7 5.7 5.7 5.7 5.7 5.7
     5.7 5.8 5.8 5.8 5.8 5.8 5.8 5.8 5.9 5.9 5.9 6.  6.  6.  6.  6.  6.  6.1
     6.1 6.1 6.1 6.1 6.1 6.2 6.2 6.2 6.2 6.3 6.3 6.3 6.3 6.3 6.3 6.3 6.3 6.3
     6.4 6.4 6.4 6.4 6.4 6.4 6.4 6.5 6.5 6.5 6.5 6.5 6.6 6.6 6.7 6.7 6.7 6.7
     6.7 6.7 6.7 6.7 6.8 6.8 6.8 6.9 6.9 6.9 6.9 7.  7.1 7.2 7.2 7.2 7.3 7.4
     7.6 7.7 7.7 7.7 7.7 7.9]
    

# 24 . Encontre o valor mais frequente da terceira coluna (petallength) do dataset iris.


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[2])

mais = 0
pos_mais = 0
lista_iris = list(iris)

for i in range(len(iris)):
    if lista_iris.count(lista_iris[i]) > mais:
        mais = lista_iris.count(iris[i])
        pos_mais = i
print(str(lista_iris[pos_mais]) + ' aparece ' + str(mais) + ' vezes.')
```

    1.5 aparece 14 vezes.
    

# 25 . Encontre a posição da primeira ocorrência de um valor maior do que 1.0 da quarta coluna (petalwidth) do dataset iris.


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='float', usecols=[3])

pos = 0
achou = False

for i in range(len(iris)):
    if iris[i] == 1.0 and achou == False:
        pos = i
        achou = True
        
print(pos)
print(iris[pos])
```

    57
    1.0
    

# 26 . Obtenha a posição dos top 5 valores máximos do vetor abaixo:


```python
import numpy as np
import numpy_indexed as npi

np.random.seed(100)
vetor = np.random.uniform(1,50, 20)

array = np.sort(vetor)

print(str(array[len(vetor)-1]) + ' ' + str(array[len(vetor)-2]) + ' ' + str(array[len(vetor)-3]) + ' ' + str(array[len(vetor)-4]) + ' ' + str(array[len(vetor)-5]))
```

    48.95256545066111 44.674775761300936 42.39403048367528 41.46678500014733 40.99501268756616
    

# 27 . Encontre a média de uma coluna numérica agrupada por uma coluna categórica em uma matriz (vetor numpy 2D)


```python
import numpy as np

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = np.genfromtxt(url, delimiter=',', dtype='str')

setosa = []
versicolor = []
virginica = []

for i in range(len(iris[:,4])):
    if iris[i, 4] == 'Iris-setosa':
        setosa.append(float(iris[i,1]))
    elif iris[i, 4] == 'Iris-versicolor':
        versicolor.append(float(iris[i,1]))
    else:
        virginica.append(float(iris[i,1]))

print('Média setosa = ' + str(np.mean(setosa)))
print('Média versicolor = ' + str(np.mean(versicolor)))
print('Média virginica = ' + str(np.mean(virginica)))
```

    Média setosa = 3.418
    Média versicolor = 2.7700000000000005
    Média virginica = 2.974
    
