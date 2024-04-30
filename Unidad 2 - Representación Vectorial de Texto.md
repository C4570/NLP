# Unidad 2 - Representación Vectorial de Texto

En el Procesamiento del Lenguaje Natural, uno de los desafíos más fundamentales es cómo representar el texto de una manera que los modelos puedan entender y procesar. Este capítulo se centra en la codificación y representación vectorial de Texto. Al convertir el texto en vectores numéricos, podremos luego aplicar una amplia gama de algoritmos de aprendizaje automático para tareas de NLP.

## 1. Codificación de Texto en Vectores

**One-hot encoding**

La codificación One-Hot es una técnica de procesamiento de datos que se utiliza para convertir categorías nominales en un formato que se puede proporcionar a los algoritmos de aprendizaje automático para mejorar la precisión de las predicciones. Las categorías nominales son básicamente variables categóricas que pueden ser divididas en múltiples categorías pero no tienen ningún orden ni prioridad. Son el tipo de variables que se utilizan para etiquetar un grupo con un nombre, como los nombres de las ciudades o los estados.

En el contexto del procesamiento del lenguaje natural (NLP), la codificación One-Hot se utiliza para convertir palabras en vectores. En este proceso, cada palabra de la frase se representa como un vector en n-dimensiones, donde n es el tamaño del vocabulario, es decir, el número total de palabras únicas en el texto. Cada palabra se representa con un vector de longitud n, donde la posición correspondiente a la palabra en el vocabulario se establece en 1, y todas las demás posiciones se establecen en 0.

Por ejemplo, si nuestro vocabulario consta de las palabras ['gato', 'perro', 'casa'], la palabra 'gato' se representaría como [1, 0, 0], 'perro' como [0, 1, 0] y 'casa' como [0, 0, 1]:

| gato | perro | casa |
| --- | --- | --- |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

La codificación One-Hot es una forma simple y eficaz de representar datos categóricos para el aprendizaje automático, pero tiene la desventaja de que puede resultar en vectores de alta dimensionalidad si el vocabulario es muy grande. Además, la codificación One-Hot no tiene en cuenta la similitud semántica entre las palabras, es decir, palabras con significados similares no tienen vectores similares.

Para realizar la codificación One-Hot en Python, podemos utilizar la librería **`sklearn`**. Aquí vemos como hacerlo con la frase "*Me gustan las hamburguesas*":

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Nuestra frase de trabajo
frase = "Me gustan las hamburguesas"

# Dividimos la frase en palabras
palabras = frase.split()

# Creamos un codificador One-Hot
onehot_encoder = OneHotEncoder(sparse=False)

# Ajustamos el codificador One-Hot a nuestras palabras
onehot_encoded = onehot_encoder.fit_transform(np.array(palabras).reshape(-1, 1))

# Imprimimos el resultado
print(onehot_encoded)

# Imprimimos cómo se codificó cada palabra
for i, palabra in enumerate(palabras):
    print(f"La palabra '{palabra}' se codificó como: {onehot_encoded[i]}")
```

La salida será una matriz donde cada fila corresponde a una palabra en la frase, y cada columna corresponde a una palabra única en el vocabulario. Un '1' en una posición indica que la palabra de esa fila es la palabra correspondiente a esa columna en el vocabulario.

El resultado de la ejecución será el siguiente:

```python
[[1. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 1.]
 [0. 0. 1. 0.]]
La palabra 'Me' se codificó como: [1. 0. 0. 0.]
La palabra 'gustan' se codificó como: [0. 1. 0. 0.]
La palabra 'las' se codificó como: [0. 0. 0. 1.]
La palabra 'hamburguesas' se codificó como: [0. 0. 1. 0.]
```

También podemos usar Pandas para realizar codificaciones One-Hot. Los dummies son codificaciones basadas en el mismo mecanismo. Veamos un ejemplo:

```python
import pandas as pd

# Nuestra frase de trabajo
frase = "Me gustan las hamburguesas"

# Dividimos la frase en palabras
palabras = frase.split()

# Creamos un DataFrame a partir de nuestras palabras
df = pd.DataFrame(palabras, columns=['Palabras'])

# Creamos una codificación One-Hot usando get_dummies
onehot_encoded = pd.get_dummies(df['Palabras'])

# Imprimimos el resultado
print(onehot_encoded)

# Imprimimos cómo se codificó cada palabra
for i, palabra in enumerate(palabras):
    print(f"La palabra '{palabra}' se codificó como: {onehot_encoded.iloc[i].to_numpy()}")
```

Y el resultado será similar al obtenido con Scikit-Learn.

**Count Vectorizer**

La técnica de Count Vectorizer es una forma de convertir texto en características numéricas. Es una técnica de codificación que es muy útil para el procesamiento del lenguaje natural y la minería de texto.

La idea detrás de Count Vectorizer es bastante simple. Para cada documento en nuestro conjunto de datos, contamos cuántas veces aparece cada palabra. Luego, creamos un vector para cada documento que contiene las cuentas de cada palabra.

Por ejemplo, si nuestro vocabulario consta de las palabras ['gato', 'perro', 'casa'], y tenemos un documento que dice "el gato está en la casa", el vector de características para este documento sería [1, 0, 1] porque la palabra 'gato' aparece una vez, 'perro' no aparece y 'casa' aparece una vez.

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled.png)

Una de las ventajas de Count Vectorizer es que es muy fácil de entender e implementar. Sin embargo, tiene la desventaja de que no tiene en cuenta el orden de las palabras en el documento, lo que puede ser importante en muchos contextos de procesamiento del lenguaje natural.

Además, Count Vectorizer puede dar mucha importancia a las palabras que aparecen con mucha frecuencia, lo que puede no ser siempre deseable. Por ejemplo, palabras como 'el', 'un', 'la', etc., pueden aparecer con mucha frecuencia en los documentos, pero no aportan mucha información útil para tareas como la clasificación de documentos. Para manejar este problema, a menudo se utiliza una técnica llamada TF-IDF (Term Frequency-Inverse Document Frequency), que da más importancia a las palabras que son más raras en el conjunto de datos.

Veamos un ejemplo de cómo usar `**CountVectorizer`** en Python con la librería sklearn:

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Nuestro corpus de texto
corpus = ["El gato está en la casa",
          "El perro está en el jardín",
          "La casa está limpia",
          "El gato juega en el jardín"]

# Creamos una instancia de CountVectorizer
vectorizer = CountVectorizer()

# Ajustamos el vectorizador a nuestro corpus y transformamos nuestro corpus en vectores de conteo
X = vectorizer.fit_transform(corpus)

# Imprimimos los vectores de características
print("Vectores de características:\n", X.toarray())

# Imprimimos las palabras del vocabulario
print("\nPalabras del vocabulario:", vectorizer.get_feature_names_out())

# Convertimos la matriz en un DataFrame de pandas para una mejor visualización
df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Imprimimos el DataFrame
print('\nVectores con palabras como columnas:')
print(df)
```

Este código imprimirá los vectores de características para cada documento en el corpus, así como las palabras del vocabulario. Cada vector de características representa la cantidad de veces que cada palabra del vocabulario aparece en el documento correspondiente:

```python
Vectores de características:
 [[1 1 1 1 1 0 0 1 0 0]
 [0 2 1 1 0 1 0 0 0 1]
 [1 0 0 1 0 0 0 1 1 0]
 [0 2 1 0 1 1 1 0 0 0]]

Palabras del vocabulario: ['casa' 'el' 'en' 'está' 'gato' 'jardín' 'juega' 'la' 'limpia' 'perro']

Vectores con palabras como columnas:
   casa  el  en  está  gato  jardín  juega  la  limpia  perro
0     1   1   1     1     1       0      0   1       0      0
1     0   2   1     1     0       1      0   0       0      1
2     1   0   0     1     0       0      0   1       1      0
3     0   2   1     0     1       1      1   0       0      0
```

**Codificación TF-IDF**

TF-IDF, que significa Frecuencia de Término - Frecuencia Inversa de Documento, es una técnica de codificación de texto que se utiliza comúnmente en el procesamiento del lenguaje natural. Es una forma de representar cómo es de importante una palabra específica para un documento en una colección o corpus. El valor de TF-IDF aumenta proporcionalmente al número de veces que una palabra aparece en el documento, pero se compensa por la frecuencia de la palabra en el corpus, lo que ayuda a ajustar el hecho de que algunas palabras aparecen más frecuentemente en general.

TF-IDF se compone de dos componentes:

- **TF (Frecuencia de Término)**: Es simplemente la frecuencia de una palabra en un documento. Es similar a la codificación de conteo que acabamos de ver, pero en lugar de contar el número de apariciones de cada palabra, calculamos la frecuencia de aparición. Esto se hace dividiendo el número de veces que la palabra aparece en un documento por el número total de palabras en el documento.

$$
\text{TF}(t, d) = \frac{\text{número de veces que el término } t \text{ aparece en el documento } d}{\text{número total de términos en el documento } d}
$$

***t***: Representa un "término" específico o una "palabra".

***d***: Representa un "documento" específico en el corpus de documentos que estamos analizando. Un "documento" podría ser un artículo, un tweet, una publicación de blog, etc.

- **IDF (Frecuencia Inversa de Documento)**: Este es el componente que equilibra la frecuencia de las palabras. Es el logaritmo del número total de documentos en el corpus dividido por el número de documentos en los que aparece la palabra. De esta manera, las palabras que son muy comunes, como "el", "un", "es", etc., que aparecen en muchos documentos, tendrán un valor IDF más bajo, reduciendo su importancia en los cálculos de TF-IDF.

$$
\text{IDF}(t, D) = \log \left( \frac{\text{número total de documentos en el corpus } D}{\text{número de documentos que contienen el término } t} \right)
$$

***D***: representa el conjunto completo de documentos que estamos analizando.

Finalmente, TF-IDF es el producto simple de TF e IDF:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%201.png)

La codificación TF-IDF se utiliza a menudo en la recuperación de información y la minería de texto para representar documentos como vectores, donde cada dimensión es una palabra específica del corpus y el valor en esa dimensión es el TF-IDF de esa palabra en ese documento. Esto es útil para tareas como la clasificación de documentos y la agrupación de documentos, donde necesitamos una forma de representar documentos en un espacio vectorial.

<aside>
💡 Si una palabra aparece muchas veces en un documento, eleva el valor de TF-IDF. Por el contrario, si una palabra aparece muchas veces en el corpus o conjunto de documentos, disminuirá el valor TF-IDF

</aside>

Aquí vemos ejemplo de cómo usar **`TfidfVectorizer`** de **`sklearn`** en Python:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Nuestro corpus de texto
# Aclaración: Para simplificar el ejemplo, ya se han quitado las stopwords.
# La frases originales eran "John has some cats", "Cats eat fish", "I eat a big fish"
corpus = ["John cat", 
          "cat eat fish",
          "eat big fish"]

# Inicializamos el TfidfVectorizer
vectorizer = TfidfVectorizer()

# Ajustamos y transformamos nuestro corpus
X = vectorizer.fit_transform(corpus)

# Mostramos las características (palabras únicas en el corpus)
print("Características: ", vectorizer.get_feature_names_out())

# Mostramos la matriz TF-IDF resultante
print("\nMatriz TF-IDF:")
print(X.toarray())

# Resumen
print("\nVocabulario:")
print(vectorizer.vocabulary_)
print("\nIDF:")
print(vectorizer.idf_)
```

Este código primero inicializa el **`TfidfVectorizer`**, luego ajusta este vectorizador a nuestro corpus y transforma el corpus en una matriz TF-IDF. Luego imprime las características, que son las palabras únicas en el corpus, y la matriz TF-IDF resultante. Cada fila en la matriz corresponde a un documento en el corpus, y cada columna corresponde a una palabra en el corpus. Los valores en la matriz son los valores TF-IDF de cada palabra en cada documento. Como resultado de la ejecución obtendremos:

```python
Características:  ['big' 'cat' 'eat' 'fish' 'john']

Matriz TF-IDF:
[[0.         0.60534851 0.         0.         0.79596054]
 [0.         0.57735027 0.57735027 0.57735027 0.        ]
 [0.68091856 0.         0.51785612 0.51785612 0.        ]]

Vocabulario:
{'john': 4, 'cat': 1, 'eat': 2, 'fish': 3, 'big': 0}

IDF:
[1.69314718 1.28768207 1.28768207 1.28768207 1.69314718]
```

Aquí vemos el desarrollo de cómo se llega al cálculo, según el método de Scikit-Learn, ya que este difiere ligeramente de la definición original:

![[https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d](https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d)](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%202.png)

[https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d](https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d)

Aquí vemos el desarrollo del cálculo de TF-IDF según Scikit-Learn:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%203.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%204.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%205.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%206.png)

**Vectorización de hash**

> ***Hash:*** Para entender este método, primero es necesario comprender que es un [hash](https://es.wikipedia.org/wiki/Funci%C3%B3n_hash). Una función de hash es una función que toma una entrada (o "mensaje") y devuelve una cadena de longitud fija, que generalmente es una secuencia de números y letras. Esta salida, conocida como valor hash, debería ser única (dentro de lo razonable) para cada entrada diferente. Es decir, es muy poco probable que dos entradas diferentes produzcan el mismo valor hash. Esa situación es conocida como **colisión de hash**.
> 

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%207.png)

La [vectorización de hash](https://en.wikipedia.org/wiki/Feature_hashing) es una técnica de vectorización que utiliza una función de hash para convertir las características de texto en representaciones numéricas. A diferencia de las técnicas de vectorización como la codificación one-hot, la vectorización de conteo y la vectorización TF-IDF, la vectorización de hash no requiere que se mantenga un vocabulario, lo que puede ser muy útil en situaciones en las que el vocabulario puede ser muy grande y consumir mucha memoria.

La vectorización de hash se realiza utilizando la clase **`[HashingVectorizer`**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) en **`sklearn`**. Cuando se inicializa un **`[HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)`**, se puede especificar el número de características (es decir, la longitud del vector de características) que se desea para la salida. Cuando se transforma un texto, cada palabra en el texto se convierte en un número entero utilizando una función de hash, y luego se utiliza este número para indexar en el vector de características y aumentar el valor en ese índice.

El proceso de vectorización consiste en tokenizar las palabras, crear los hash de cada palabra, y luego convertir esos hash en una [matriz dispersa](https://en.wikipedia.org/wiki/Sparse_matrix) ("sparse matrix"). Una matriz dispersa es una matriz en la que la mayoría de sus elementos son cero (o, en general, cualquier valor que se considere "predeterminado" o "no significativo"). 

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%208.png)

El tamaño de la matriz resultante, está determinado por la cantidad de frases o documentos, y el número de características que usemos como parámetro en `**HashingVectorizer**`.

Un aspecto importante a tener en cuenta sobre la vectorización de hash es que es una técnica "sin estado", lo que significa que no mantiene ninguna información sobre el estado anterior (como un vocabulario). Esto significa que no puede proporcionar una forma de mapear desde las características a las palabras originales. Esto puede ser un inconveniente si necesitamos interpretar los vectores de características.

```python
from sklearn.feature_extraction.text import HashingVectorizer

# Nuestro texto de trabajo
texto = ['Esta es una introducción a NLP', 'Es probable que sea útil para las personas',
'Machine learning es la nueva electricidad', 'Habrá menos exageración sobre la IA y más acción en adelante',
'¡Python es la mejor herramienta!', 'Python es un buen lenguaje', 'Me gusta este libro', 'Quiero más libros como este']

# Creamos el HashingVectorizer
vectorizer = HashingVectorizer(n_features=10)

# Aplicamos la transformación
vector = vectorizer.transform(texto)

# Imprimimos el resultado
print(vector.toarray())

# summarize the vector
print('\nForma del vector:')
print(vector.shape)
```

En este ejemplo, hemos configurado **`HashingVectorizer`** para producir vectores de características de longitud 10. Luego, transformamos nuestro texto en estos vectores de características utilizando el método **`transform()`**. Finalmente, imprimimos los vectores de características resultantes.

```
[[ 0.          0.          0.          0.37796447  0.75592895  0.
   0.37796447  0.37796447  0.          0.        ]
 [ 0.28867513  0.57735027 -0.28867513  0.57735027  0.28867513  0.
  -0.28867513  0.          0.          0.        ]
 [-0.5         0.          0.5         0.          0.5         0.
   0.          0.          0.          0.5       ]
 [-0.28867513 -0.28867513  0.          0.57735027  0.28867513  0.28867513
   0.          0.57735027  0.          0.        ]
 [ 0.          0.57735027  0.          0.          0.57735027  0.57735027
   0.          0.          0.          0.        ]
 [ 0.         -0.4472136   0.          0.          0.4472136   0.4472136
  -0.4472136  -0.4472136   0.          0.        ]
 [ 0.5        -0.5        -0.5         0.          0.         -0.5
   0.          0.          0.          0.        ]
 [ 0.         -0.4472136   0.          0.4472136   0.         -0.4472136
   0.          0.4472136   0.          0.4472136 ]]

Forma del vector:
(8, 10)
```

Por defecto, la matriz obtenida tiene sus valores normalizados entre -1 y 1. Podríamos indicar que los valores no sean normalizados del siguiente modo:

```python
vectorizer = HashingVectorizer(n_features=10, norm=None)
```

Es importante tener en cuenta que, a diferencia de otros métodos de vectorización, **`HashingVectorizer`** no proporciona una forma de mapear las características de nuevo a las palabras originales, ya que no mantiene un vocabulario. Además, puede haber colisiones de hash, donde diferentes palabras pueden mapearse al mismo índice en el vector de características. Sin embargo, en la práctica, este suele ser un problema menor, especialmente si el número de características es lo suficientemente grande.

**Resumen de métodos vectorización**

Si bien hay más métodos que podríamos explorar, aquí tenemos un resumen de los cuatro métodos vistos:

| Técnica | Descripción | Pros | Contras |
| --- | --- | --- | --- |
| One-hot encoding | Cada palabra en el vocabulario se representa como un vector en el que un elemento es 1 y el resto son 0. | Fácil de entender y de implementar. | Genera vectores de alta dimensionalidad. No tiene en cuenta la frecuencia de las palabras ni su relevancia en el texto. |
| Count Vectorizer | Cada documento se representa como un vector en el que cada elemento es la frecuencia de una palabra en el documento. | Toma en cuenta la frecuencia de las palabras. | No tiene en cuenta la relevancia de las palabras en el texto. Puede dar demasiado peso a las palabras comunes. |
| TF-IDF | Similar a Count Vectorizer, pero da más peso a las palabras que son raras en el corpus y menos peso a las palabras que son comunes. | Toma en cuenta tanto la frecuencia de las palabras como su relevancia en el texto. | Más complejo de entender y de implementar que las técnicas anteriores. |
| Hash Vectorizer | Cada palabra se mapea a un número en un rango predefinido utilizando una función de hash. | Permite trabajar con vectores de tamaño fijo, independientemente del tamaño del vocabulario. Útil cuando el vocabulario es muy grande. | Puede haber colisiones de hash. No se puede mapear las características de vuelta a las palabras originales. |

<aside>
💡 Es importante tener en cuenta que la elección de la técnica de vectorización depende del problema específico que estés tratando de resolver. Algunas técnicas pueden funcionar mejor que otras en diferentes contextos.

</aside>

**Codificación de texto. Conclusión**

Estos métodos de codificación que hemos visto, podrían aplicarse en complementación con modelos como Naive Bayes, Regresión Logística, Máquinas de Vectores de Soporte (SVM), modelos de redes neuronales, entre otros, pero tienen una limitación. Los métodos de codificación de palabras en vectores, se centran en la representación numérica del texto, pero no capturan la semántica o el significado del texto. Estos métodos tratan las palabras como entidades aisladas y no capturan el contexto o la relación entre las palabras.

Por ejemplo, en One-Hot Encoding, cada palabra se representa como un vector en un espacio de alta dimensión, y cada palabra es ortogonal a todas las demás, lo que significa que no hay relación entre las palabras. 

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%209.png)

En el caso de Count Vectorizer y TF-IDF, aunque se tiene en cuenta la frecuencia de las palabras, no se captura la relación semántica entre las palabras. Entonces, independientemente del modelo que usemos, estaremos limitados en cuando a la comprensión del lenguaje natural.

Para capturar la semántica y el contexto de las palabras, se utilizan técnicas como [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf), [GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation) y [FastText](https://fasttext.cc/). Estos métodos generan lo que se conoce como "embeddings" de palabras, que son representaciones vectoriales densas donde las palabras con significados similares se ubican cerca unas de otras en el espacio vectorial, y son capaces de capturar la semántica y las relaciones entre las palabras. Generalmente se entrenan con grandes cantidades de texto y aprenden a predecir palabras en función de su contexto.

## 2. Word embeddings (incrustaciones de palabras)

Los "Word Embeddings" o "Incrustaciones de palabras" son una de las técnicas más populares en el procesamiento del lenguaje natural, especialmente cuando se trata de tareas de aprendizaje automático. Esta técnica se utiliza para representar palabras en un espacio de alta dimensión, donde las palabras con significados similares se agrupan juntas. En otras palabras, los "Word Embeddings" son una forma de representar la semántica de las palabras como vectores, de tal manera que las palabras con contextos similares se encuentren cercanas en el espacio vectorial.

La idea detrás de los "Word Embeddings" es que las palabras que aparecen en contextos similares tienen significados similares. Por ejemplo, las palabras "perro" y "gato" a menudo aparecen en contextos similares (como "mi ___ come mucho") y, por lo tanto, deberían tener vectores cercanos.

Los "Word Embeddings" se generan utilizando algoritmos como Word2Vec, GloVe, entre otros, que utilizan redes neuronales para aprender estas representaciones a partir de grandes corpus de texto. Estos algoritmos pueden capturar sutilezas semánticas y sintácticas, como que "rey" está para "hombre" como "reina" está para "mujer", o que "caminar" es la versión en presente de "caminó".

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2010.png)

Consideremos las siguientes frases similares: "Ten un buen día" y "Ten un gran día". Apenas tienen un significado diferente. Si construimos un vocabulario exhaustivo (llamémoslo V), tendríamos V = {Ten, un, buen, gran, día}.

Ahora, creemos un vector codificado en one-hot para cada una de estas palabras en V. La longitud de nuestro vector codificado en one-hot sería igual al tamaño de V (=5). Tendríamos un vector de ceros excepto para el elemento en el índice que representa la palabra correspondiente en el vocabulario. Ese elemento en particular sería uno. Las codificaciones a continuación explicarían esto mejor.

Ten = [1,0,0,0,0]; un=[0,1,0,0,0]; buen=[0,0,1,0,0]; gran=[0,0,0,1,0]; día=[0,0,0,0,1]

Si intentamos visualizar estas codificaciones, podemos pensar en un espacio de 5 dimensiones, donde cada palabra ocupa una de las dimensiones y no tiene nada que ver con el resto (no hay proyección a lo largo de las otras dimensiones). Esto significa que 'buen' y 'gran' son tan diferentes como 'día' y 'ten', lo cual no es cierto.

Aquí vemos una comparación de la codificación de texto (one-hot) vs. embeddings:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2011.png)

Mientras que las representaciones de palabras obtenidas de la codificación one-hot o hash son escasas, de alta dimensión y codificadas de forma rígida, las incrustaciones de palabras son densas, relativamente de baja dimensión y aprendidas a partir de los datos.

### C**aracterísticas semánticas**

*Fuente: [https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/tutorial.html)*

Consideremos las palabras "man", "woman", "boy", and "girl”. Dos de ellos se refieren a hombres y dos a mujeres. Además, dos de ellos se refieren a adultos y dos a niños. Podemos trazar estos mundos como puntos en un gráfico donde el eje *x representa el género y el eje y* representa la edad:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2012.png)

El género y la edad se denominan *rasgos semánticos*: representan parte del significado de cada palabra. Si asociamos una escala numérica con cada característica, entonces podemos asignar coordenadas a cada palabra:

|  | Gender | Age |
| --- | --- | --- |
| man | 1 | 7 |
| woman | 9 | 7 |
| boy | 1 | 2 |
| girl | 9 | 2 |

Podemos agregar nuevas palabras a la trama en función de sus significados. Por ejemplo, ¿dónde deben ir las palabras "adult" y "child”? ¿Qué tal "infant"? ¿O "grandfather"?

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2013.png)

Ahora consideremos las palabras "king", "queen", "prince" y "princess". Tienen los mismos atributos de género y edad que "man", "woman", "boy" y "girl". Pero no significan lo mismo. Para distinguir "man" de "king", "woman" de "queen", y así sucesivamente, necesitamos introducir una nueva característica semántica en la que se diferencian. Llamémoslo "realeza". Ahora tenemos que trazar los puntos en un espacio tridimensional:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2014.png)

|  | Gender | Age | Royalty |
| --- | --- | --- | --- |
| man | 1 | 7 | 1 |
| woman | 9 | 7 | 1 |
| boy | 1 | 2 | 1 |
| girl | 9 | 2 | 1 |
| king | 1 | 8 | 8 |
| queen | 9 | 7 | 8 |
| prince | 1 | 2 | 8 |
| princess | 9 | 2 | 8 |

Cada palabra tiene tres valores de coordenadas: edad, género y realeza. A estas listas de números las llamamos *vectores* . Dado que representan los valores de las características semánticas, también podemos llamarlos *vectores de características*. Tengamos en cuenta que le hemos asignado a "king" un valor de edad ligeramente mayor (8) que a "reina" (7). Tal vez sea porque hemos leído muchas historias sobre reyes muy antiguos, pero no tantas sobre reinas muy antiguas. Los valores de las características no tienen que ser perfectamente simétricos.

### Similitud **de Coseno**

Nuestro objetivo es que las palabras con un contexto similar ocupen posiciones espaciales cercanas. Matemáticamente, el coseno del ángulo entre tales vectores debería estar cerca de 1, es decir, ángulo cercano a 0. La fórmula es la siguiente:

$$
\cos (\theta ) =   \dfrac {A \cdot B} {\left\| A\right\|\left\| B\right\|} 
$$

Notación:

- *A* y *B* son dos vectores.
- cos(*θ*) es el coseno del ángulo *θ* entre los vectores *A* y *B*
- *A* ⋅ *B* denota el producto punto de los vectores *A* y *B*.
- ∥*A*∥ y ∥*B*∥ son las magnitudes (o normas) de los vectores *A* y *B*, respectivamente.

Veámoslo gráficamente:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2015.png)

Si bien podríamos medir distancia entre palabras usando la distancia euclidiana, en NLP se usa principalmente la **distancia coseno** por estos motivos:

- **Invarianza de la longitud del vector**: En muchos casos en NLP, los vectores de palabras se normalizan para tener una longitud (o magnitud) de 1. Esto significa que solo nos importa la dirección del vector, no su longitud. La similitud del coseno es una medida de la orientación de los vectores y no se ve afectada por la magnitud. Por lo tanto, es una buena opción cuando queremos comparar la similitud de las palabras independientemente de la frecuencia de aparición de las palabras.
- **Alta dimensionalidad**: Los vectores de palabras en NLP suelen ser de alta dimensión (por ejemplo, 300 dimensiones para los vectores de palabras de GloVe). En espacios de alta dimensión, la ["maldición de la dimensionalidad"](https://www.iartificial.net/la-maldicion-de-la-dimension-en-machine-learning/) significa que la distancia euclidiana puede ser menos significativa. La similitud del coseno tiende a ser una métrica más útil en estos casos.
- **Interpretación intuitiva**: La similitud del coseno mide el coseno del ángulo entre dos vectores. Un valor de 1 significa que los vectores son idénticos, un valor de 0 significa que son ortogonales (no relacionados), y un valor de -1 significa que son diametralmente opuestos. Esta es una interpretación intuitiva que a menudo es útil en NLP.
- **Eficiencia computacional**: Calcular la similitud del coseno puede ser más eficiente que calcular la distancia euclidiana, especialmente en espacios de alta dimensión.

La medida que ayuda a estimar el ángulo entre vectores se llama similitud de coseno, y tiene la buena propiedad de ser mayor cuando los dos vectores están más cerca entre sí con un ángulo menor (es decir, más similares) y menor cuando están más distantes con un ángulo mayor (es decir, menos similar). 

Podemos implementar la fórmula vista anteriormente en el gráfico, en una función de Python, usando la librería **`numpy`**:

```python
import numpy as np

def cosine_similarity(A, B):
    """
    Calcula la similitud del coseno entre dos vectores A y B.

    Parámetros:
    - A, B: Vectores de entrada.

    Retorna:
    - Similitud del coseno entre A y B.
    """
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)

    return dot_product / (norm_A * norm_B)

# Ejemplo de uso:
vector_A = np.array([1, 2, 3])
vector_B = np.array([4, 5, 6])

print(cosine_similarity(vector_A, vector_B))
```

La función **`cosine_similarity`** toma dos vectores **`A`** y **`B`** como entrada y devuelve su similitud del coseno. El coseno de un ángulo de 0° es igual a 1, lo que significa máxima cercanía y similitud entre los dos vectores. La siguiente figura muestra un ejemplo:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2016.png)

Incluso, como vemos en el gráfico anterior, podríamos tener similitud negativa. Un valor de -1 significaría vectores opuestos.

Veamos un ejemplo de cómo trabajar con similitud de coseno, usando `**cosine_similarity**`

de Scikit-Learn:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Supongamos que estos son tus vectores (embeddings)
vectors = np.array([
    [0.1, 0.2, 0.4],
    [0.3, 0.4, 0.5],
    [0.1, 0.3, 0.1]
])

# Les asignamos nombres a los vectores
vector_names = ["Banana", "Manzana", "Mandarina"]

# Y este es tu vector de consulta
query_vector = np.array([[0.3, 0.5, 0.5]])

# Calcula la similitud de coseno entre el vector de consulta y todos los otros vectores
similarities = cosine_similarity(query_vector, vectors)

# Imprime las similitudes junto con los nombres de los vectores
for name, similarity in zip(vector_names, similarities[0]):
    print(f"{name}: {similarity:.4f}")
```

Obtendremos:

```python
Banana: 0.9375
Manzana: 0.9942
Mandarina: 0.9028
```

Vemos que al mostrar las distancias, el vector “Manzana” ([0.3, 0.4, 0.5]) es el que más se acerca a [0.3, 0.5, 0.5].

<aside>
💡 Un buen recurso para profundizar en espacios vectoriales es el siguiente: [https://aman.ai/coursera-nlp/vector-spaces/](https://aman.ai/coursera-nlp/vector-spaces/)

</aside>

### **Tipos de modelos de embeddings**

Los modelos de embeddings se pueden clasificar según la capacidad de capturar la relación de las palabras con el contexto:

![Fuente: **[A Comparative Study on Word Embeddings in Deep Learning for Text Classification](https://www.researchgate.net/publication/348946675_A_Comparative_Study_on_Word_Embeddings_in_Deep_Learning_for_Text_Classification?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJfZGlyZWN0In19)**](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2017.png)

Fuente: **[A Comparative Study on Word Embeddings in Deep Learning for Text Classification](https://www.researchgate.net/publication/348946675_A_Comparative_Study_on_Word_Embeddings_in_Deep_Learning_for_Text_Classification?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJfZGlyZWN0In19)**

**1. Contexto-Independiente (Context-independent):**

- Estos métodos son conocidos como "embeddings clásicos". Aprenden representaciones a través de redes neuronales superficiales basadas en modelos de lenguaje o factorización de matrices de co-ocurrencia.
- Las representaciones aprendidas son únicas y distintas para cada palabra sin considerar el contexto de la palabra.
- Estos embeddings suelen ser pre-entrenados en corpus de texto generales y se distribuyen en forma de archivos descargables. Estos archivos pueden aplicarse directamente para inicializar los pesos de embedding para tareas de lenguaje downstream.
- Ejemplos prominentes incluyen: **word2vec**, **GloVe** y **FastText**.

**2. Contexto-Dependiente (Context-dependent):**

- A diferencia de los embeddings de contexto-independiente, los métodos de contexto-dependiente aprenden diferentes embeddings para la misma palabra dependiendo del contexto en el que se utiliza.
- Por ejemplo, la palabra polisémica "banco" tendrá múltiples embeddings dependiendo de si se usa en un contexto relacionado con el deporte o uno relacionado con finanzas.
- Estos embeddings han ganado popularidad recientemente y se dividen en dos categorías principales:
    - **Basados en RNNs (Redes Neuronales Recurrentes)**: Como **CoVe**, **Flair** y **ELMo**.
    - **Basados en Transformer**: Como **BERT** y **ALBERT**.

### **Word2Vec**

[Word2vec](https://arxiv.org/pdf/1301.3781.pdf) fue publicada en 2013 y es una de las técnicas más populares de word embeddings utilizando una red neuronal de dos capas. Su entrada es un corpus de texto y su salida es un conjunto de vectores. 

Hay dos algoritmos de entrenamiento principales para word2vec, uno es la bolsa continua de palabras (CBOW), otro se llama skip-gram. La principal diferencia entre estos dos métodos es que CBOW está utilizando el contexto para predecir una palabra objetivo mientras que skip-gram está utilizando una palabra para predecir un contexto objetivo. 

![La arquitectura CBOW predice la palabra actual en función del contexto, y Skip-gram predice las palabras circundantes dada la palabra actual. [https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2018.png)

La arquitectura CBOW predice la palabra actual en función del contexto, y Skip-gram predice las palabras circundantes dada la palabra actual. [https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)

**Skip-Gram**

Para este modelo, consideremos el problema de predecir un conjunto de palabras de contexto a partir de una única palabra central. En este caso, imaginemos predecir las palabras de contexto "neumático", "carretera", "vehículo", "puerta" a partir de la palabra central "coche". En el enfoque "Skip-Gram", la palabra central se representa como un único vector codificado en one-hot y se presenta a una red neuronal que se optimiza para producir un vector con valores altos en lugar de las palabras de contexto predichas, es decir, valores cercanos a 1 para palabras como "neumático", "vehículo", "puerta", etc.

![El algoritmo Skip-Gram para el entrenamiento de incrustación de palabras utilizando una red neuronal para predecir palabras de contexto a partir de una codificación one-hot de palabras centrales.
[http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2019.png)

El algoritmo Skip-Gram para el entrenamiento de incrustación de palabras utilizando una red neuronal para predecir palabras de contexto a partir de una codificación one-hot de palabras centrales.
[http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

Las capas internas de la red neuronal son pesos lineales, que pueden representarse como una matriz de tamaño (*número de palabras en el vocabulario*) X (*número de neuronas (arbitrario)*). 

Para nuestro ejemplo, vamos a decir que estamos aprendiendo vectores de palabras con 300 características. Entonces, la capa oculta estará representada por una matriz de peso con 10,000 filas (una para cada palabra en nuestro vocabulario) y 300 columnas (una para cada neurona oculta).

300 características es lo que Google usó en su modelo publicado entrenado en el conjunto de datos de noticias de Google ([se puede descargar aquí](https://code.google.com/archive/p/word2vec/)). La cantidad de funciones es un "hiperparámetro" que ajustaremos a nuestra aplicación (es decir, probar diferentes valores y ver qué valor produce los mejores resultados).

Si dos palabras diferentes tienen "contextos" muy similares (es decir, qué palabras es probable que aparezcan a su alrededor), entonces nuestro modelo debe generar resultados muy similares para estas dos palabras. ¿Y qué significa que dos palabras tengan contextos similares? Se podría esperar que las palabras "comer" y "alimento" tuvieran contextos muy similares, o "lluvia" y "clima", probablemente también tengan contextos similares.

Veamos un ejemplo con `[**Gensim`**](https://radimrehurek.com/gensim/) (https://github.com/RaRe-Technologies/gensim) de carga de modelo pre-entrenado. En Colab, instalamos la librería `gensim` y descargamos un modelo word2vec Skip-Gram [entrenado en español](https://crscardellino.ar/SBWCE/) sobre un total de 1000653 tokens:

```
!pip install gensim
!wget https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
```

Una vez que descargamos el modelo, podremos correr el siguiente ejemplo:

```python
from gensim.models import KeyedVectors

# Carga un modelo Word2Vec preentrenado (asegúrate de tener el archivo en tu directorio)
model = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin.gz', binary=True)

# Información del modelo
print(model)

# Similitud entre dos palabras específicas
print(f"Similitud entre 2 palabras: {model.similarity('perro', 'conejo')}")

# Palabra de consulta
query_word = "gato"

# Encuentra las palabras más similares a la palabra de consulta
most_similar_words = model.most_similar(positive=[query_word], topn=10)

# Imprime las palabras más similares y sus similitudes de coseno
print(f'Palabras cercanas a {query_word}:')
for word, similarity in most_similar_words:
    print(f"Palabra: {word}, Similitud: {similarity}")
```

Y los resultados serán:

```python
KeyedVectors<vector_size=300, 1000653 keys>
Similitud entre 2 palabras: 0.6112384796142578
Palabras cercanas a gato:
Palabra: perro, Similitud: 0.7445881366729736
Palabra: zorro, Similitud: 0.7061581611633301
Palabra: conejo, Similitud: 0.7018613815307617
Palabra: montés, Similitud: 0.6875571012496948
Palabra: mapache, Similitud: 0.6867462396621704
Palabra: maúlla, Similitud: 0.6719425916671753
Palabra: tigre, Similitud: 0.6647046804428101
Palabra: lybica, Similitud: 0.6631762385368347
Palabra: huiña, Similitud: 0.6606248617172241
Palabra: gatito, Similitud: 0.6600393056869507
```

Una herramienta interesante para explorar las relaciones semánticas, es usar `**negative`** cuando usamos el método `**most_similar**`: 

```python
result = model.most_similar(positive=['mujer', 'rey'], negative=['hombre'], topn=1)
print(result)

# Obtenemos:
# [('reina', 0.7493031620979309)]
```

Un ejemplo clásico de analogía usando word embeddings es "hombre" - "mujer" + "rey" = "reina”. El argumento **`negative`** permite especificar una lista de palabras cuyos vectores deben ser restados del vector resultante antes de buscar las palabras más similares.

### **CBOW**

CBOW, que significa "Continuous Bag of Words", también forma parte de la herramienta Word2Vec. En este caso, CBOW intenta predecir una palabra objetivo (la "palabra central") basándose en las palabras de su entorno (las "palabras de contexto"). Por ejemplo, si tuviéramos la frase "El gato persigue al ratón", y eligiéramos "persigue" como la palabra objetivo, las palabras de contexto podrían ser "El", "gato", "al", "ratón".

El modelo CBOW toma todas las palabras de contexto, las codifica como vectores (a través de "one-hot encoding"), y luego las alimenta a una red neuronal. La red tiene una capa oculta de tamaño N, donde N es una cantidad arbitraria que determina la "dimensión" de los vectores de palabras finales. La red se entrena para predecir la palabra objetivo basándose en las palabras de contexto, ajustando los pesos de la red para minimizar el error de predicción.

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2020.png)

La entrada o la palabra de contexto es un vector codificado en one-hot de tamaño V. La capa oculta contiene N neuronas y la salida es nuevamente un vector de longitud V con los elementos siendo los valores softmax.

Aclaremos los términos en la imagen:

- Wvn es la matriz de pesos que mapea la entrada x a la capa oculta (matriz de dimensiones V*N)
- W’nv es la matriz de pesos que mapea las salidas de la capa oculta a la capa de salida final (matriz de dimensiones N*V)

El modelo anterior toma C palabras de contexto. Cuando se usa Wvn para calcular las entradas de la capa oculta, tomamos un promedio de todas estas C entradas de palabras de contexto.

Después de entrenar el modelo en un gran corpus de texto, los pesos de la capa oculta de la red se utilizan como los vectores de palabras. 

Veamos un ejemplo de como entrenar un modelo CBOW en Python:

```python
from gensim.models import Word2Vec
import numpy as np 

# Supongamos que este es tu corpus tokenizado
sentences = [
    ["el", "gato", "come", "pescado"],
    ["los", "perros", "ladran", "todo", "el", "tiempo"],
    ["el", "gato", "maúlla", "por", "la", "noche"],
    ["los", "perros", "corren", "sin", "parar"]
]

# Entrenar un modelo CBOW
model_cbow = Word2Vec(sentences, vector_size=100, window=4, min_count=1, workers=4, sg=0, epochs=5000)

context_words = ["el", "gato", "come"]

# Buscar las palabras más similares al contexto suministrado
similar_words = model_cbow.wv.most_similar(positive=context_words, topn=5)

# Filtrar las palabras similares para excluir las palabras en context_words
filtered_words = [word for word, similarity in similar_words if word not in context_words]

# Palabra con la mayor similitud que no esté en context_words
most_probable_word = filtered_words[0] if filtered_words else None

if most_probable_word:
    print(f"La palabra más probable para el contexto ({context_words}) es: {most_probable_word}")
else:
    print("No se encontró una palabra probable fuera del contexto dado.")

# Resultado:
# La palabra más probable para el contexto (['el', 'gato', 'come']) es: pescado
```

En este código, después de calcular las palabras similares, filtramos la lista para excluir las palabras en **`context_words`**. Luego, seleccionamos la primera palabra de la lista filtrada.

### **GloVe**

El algoritmo de vectores globales para representación de palabras, o [GloVe](https://nlp.stanford.edu/projects/glove/), es una extensión del método word2vec para el aprendizaje eficiente de vectores de palabras, desarrollado por Pennington, et al. en Stanford. GloVe es un algoritmo de aprendizaje no supervisado para embeddings de palabras. El entrenamiento se realiza en estadísticas globales agregadas de coocurrencia palabra-palabra de un corpus, y las representaciones resultantes muestran subestructuras lineales interesantes del espacio vectorial de palabras.

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2021.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2022.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2023.png)

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2024.png)

El método GloVe se basa en una idea importante: “Se pueden derivar relaciones semánticas entre palabras a partir de una matriz de co-ocurrencia.”

![La matriz de co-ocurrencia para la oración “the cat sat on the mat” (el gato se sentó en la alfombra).](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2025.png)

La matriz de co-ocurrencia para la oración “the cat sat on the mat” (el gato se sentó en la alfombra).

El ejemplo es conceptual, sobre una sola frase. Aquí hay algunas probabilidades precisas extraídas de un corpus de 6 mil millones de palabras:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2026.png)

Consideremos:

P_ik/P_jk donde P_ik = X_ik/X_i

Aquí, P_ik denota la probabilidad de ver las palabras i y k juntas, que se calcula dividiendo el número de veces que i y k aparecieron juntos (X_ik) por el número total de veces que apareció la palabra i en el corpus (X_i).

Se puede ver que dadas dos palabras, es decir, hielo y vapor, si la tercera palabra k (también llamada "palabra de prueba"):

- es muy similar al hielo pero irrelevante para el vapor (por ejemplo, k=sólido), P_ik/P_jk será muy alto (>1)
- es muy similar al vapor pero irrelevante para el hielo (por ejemplo, k=gas), P_ik/P_jk será muy pequeño (<1)
- está relacionado o no con cualquiera de las palabras, entonces P_ik/P_jk estará cerca de 1
    
    ![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2027.png)
    

O dicho de otro modo: "hielo" co-ocurre más frecuentemente con "sólido" que con "gas", mientras que "vapor" co-ocurre más frecuentemente con "gas" que con "sólido". Ambas palabras co-ocurren con su propiedad compartida "agua" con frecuencia, y ambas co-ocurren con la palabra no relacionada "moda" (fashion) con poca frecuencia. Solo en la proporción de probabilidades se cancela el ruido de palabras no discriminativas como "agua" y "moda", de modo que los valores grandes (mucho mayores que 1) se correlacionan bien con las propiedades específicas del hielo, y los valores pequeños (mucho menores que 1) se correlacionan bien con las propiedades específicas del vapor. 

Tanto GloVe como Word2Vec son dos algoritmos populares de embeddings de palabras. Aquí vemos una comparación de ambos:

- **Enfoque de aprendizaje**: Word2Vec es un método de aprendizaje predictivo: intenta predecir una palabra dada su contexto (en el caso del modelo Skip-gram) o predecir el contexto dada una palabra (en el caso del modelo CBOW). Por otro lado, GloVe es un método de aprendizaje basado en conteo: se basa en las estadísticas de co-ocurrencia de palabras en todo el corpus.
- **Uso del contexto**: Word2Vec toma en cuenta el contexto local de cada palabra, es decir, las palabras que aparecen en una ventana de tamaño específico alrededor de la palabra objetivo. GloVe, en cambio, considera el contexto global, es decir, cuántas veces cada par de palabras co-ocurre en todo el corpus.
- **Relaciones de palabras**: Ambos métodos son capaces de capturar relaciones semánticas y sintácticas entre palabras (por ejemplo, "rey" es a "hombre" como "reina" es a "mujer"). Sin embargo, GloVe, al ser un modelo global, tiende a ser mejor en capturar relaciones de co-ocurrencia a nivel de corpus y puede ser más eficaz para capturar significados más abstractos o difusos.

> Fuentes de información: 
[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
[https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010](https://towardsdatascience.com/light-on-math-ml-intuitive-guide-to-understanding-glove-embeddings-b13b4f19c010)
> 

Desafortunadamente, no es fácil utilizar [Glove](https://nlp.stanford.edu/projects/glove/) por su escaso soporte de modelos pre-entrenados en lenguaje español. De todos modos, un ejemplo para utilizarlo en inglés usando [**`TorchText`**](https://pytorch.org/text/stable/index.html) es el siguiente:

```python
import torch
from torchtext.vocab import GloVe

# Cargar embeddings de GloVe
embedding = GloVe(name='6B', dim=100)

# Obtener el embedding para una palabra específica
word = "king"
vector = embedding[word]
print(f"Embedding para '{word}':\n{vector}\n")

# Calcular la similitud de coseno entre dos palabras
word1, word2 = "king", "queen"
vec1 = embedding[word1]
vec2 = embedding[word2]

cosine_similarity = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
print(f"Similitud de coseno entre '{word1}' y '{word2}': {cosine_similarity.item()}")

# Encontrar las palabras más similares a una palabra dada (esto es más complicado en torchtext en comparación con otras bibliotecas)
word = "computer"
by_similarity = sorted(embedding.stoi.keys(), key=lambda w: torch.nn.functional.cosine_similarity(embedding[word].unsqueeze(0), embedding[w].unsqueeze(0)), reverse=True)
print(f"Palabras más similares a '{word}': {by_similarity[:5]}")

# Similitud de coseno entre 'king' y 'queen': 0.7507690787315369
# Palabras más similares a 'computer': ['computer', 'computers', 'software', 'technology', 'pc']
```

### **BERT**

BERT, que significa "Bidirectional Encoder Representations from Transformers", es un modelo de procesamiento de lenguaje natural (NLP) desarrollado por Google. Es uno de los modelos más populares y efectivos en tareas de NLP y ha revolucionado el campo desde su introducción en 2018. Aquí hay una descripción general de BERT:

1. **Bidireccionalidad**: A diferencia de los modelos anteriores que leían el texto de izquierda a derecha o de derecha a izquierda, BERT lee el texto en ambas direcciones, lo que le permite capturar el contexto de cada palabra de manera más efectiva.
2. **Transformers**: BERT se basa en la arquitectura de "transformers", que utiliza mecanismos de atención para capturar el contexto en diferentes partes de un texto. Esta arquitectura ha demostrado ser muy efectiva para tareas de NLP.
3. **Pre-entrenamiento y afinación**: BERT se pre-entrena en grandes cantidades de texto (como Wikipedia) en dos tareas principales: predicción de palabras enmascaradas y predicción de la siguiente oración. Después del pre-entrenamiento, BERT puede ser afinado en un conjunto de datos específico para tareas particulares, como clasificación de texto, respuesta a preguntas, etc.
4. **Representaciones contextuales**: A diferencia de los embeddings de palabras tradicionales como Word2Vec o GloVe, que generan un único vector de embedding para cada palabra, BERT produce embeddings contextuales. Esto significa que la representación vectorial de una palabra puede cambiar según el contexto en el que se encuentra.
5. **Modelos de diferentes tamaños**: BERT está disponible en diferentes tamaños (BERT-Base, BERT-Large, etc.) para adaptarse a diferentes requisitos de capacidad y velocidad.
6. **Multilingüe**: Además de los modelos entrenados en inglés, Google también ha proporcionado versiones multilingües de BERT que pueden trabajar con texto en varios idiomas.

Desde la introducción de BERT, se han desarrollado muchas variantes y extensiones, como RoBERTa, DistilBERT, ALBERT y más, que buscan mejorar o adaptar el modelo original de BERT a diferentes requisitos y escenarios.

BERT se utiliza para crear embeddings, pero hay algunas diferencias clave entre cómo BERT genera embeddings y cómo lo hacen modelos como Word2Vec o GloVe:

1. **Embeddings Contextuales**: Una de las características más distintivas de BERT es que produce embeddings contextuales. Esto significa que la representación vectorial de una palabra depende del contexto en el que aparece. Por ejemplo, la palabra "banco" en "Saqué dinero del banco" y "Me senté en el banco" tendría diferentes embeddings con BERT debido a los diferentes contextos. En contraste, modelos como Word2Vec y GloVe generan un único vector de embedding para cada palabra, independientemente del contexto.
2. **Profundidad y Complejidad**: BERT es un modelo mucho más profundo y complejo que Word2Vec o GloVe. Utiliza la arquitectura de transformers y mecanismos de atención para capturar relaciones complejas y contextos en el texto.
3. **Proceso de Entrenamiento**: BERT se pre-entrena utilizando tareas de predicción de palabras enmascaradas y predicción de la siguiente oración. En la tarea de predicción de palabras enmascaradas, algunas palabras del texto se ocultan (enmascaran) y el modelo intenta predecirlas. Word2Vec, por otro lado, utiliza técnicas como Skip-Gram o CBOW, donde el modelo predice palabras vecinas o el contexto dado una palabra. GloVe se entrena en matrices de co-ocurrencia de palabras.
4. **Uso en Tareas Downstream**: Aunque BERT puede ser utilizado para extraer embeddings de palabras o frases, su verdadero poder radica en su capacidad para ser afinado en tareas específicas. Después de preentrenar BERT, se puede afinar en un conjunto de datos específico para tareas como clasificación de texto, respuesta a preguntas, etc., utilizando los embeddings y las capas superiores del modelo.
5. **Tamaño y Recursos**: BERT, especialmente sus variantes más grandes, es significativamente más grande en términos de parámetros y requiere más recursos computacionales en comparación con Word2Vec o GloVe.

**Tokenización en BERT**

BERT utiliza una técnica de tokenización conocida como "[WordPiece tokenization](https://huggingface.co/learn/nlp-course/chapter6/6?fw=pt#:~:text=Tokenization%20differs%20in%20WordPiece%20and,vocabulary%2C%20then%20splits%20on%20it.)" y se asemeja a [BPE](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt). Esta técnica de tokenización es especialmente útil para manejar un vocabulario grande y diverso, y para abordar el problema de las palabras fuera del vocabulario (OOV).

Aquí hay algunas características clave de la tokenización WordPiece:

- **Subpalabras y caracteres**: En lugar de tokenizar solo a nivel de palabra completa, WordPiece descompone palabras en subpalabras más pequeñas o incluso en caracteres individuales. Por ejemplo, la palabra "untokenizable" podría descomponerse en subpalabras como ["un", "token", "##iz", "##able"].
- **Prefijos '##'**: BERT utiliza dos almohadillas ('##') para denotar subpalabras que son fragmentos de palabras más grandes y no aparecen al principio de la palabra original. En el ejemplo anterior, "##iz" y "##able" son subpalabras que no están al principio de la palabra "untokenizable".
- **Manejo de palabras OOV**: Dado que WordPiece puede descomponer palabras en subpalabras y caracteres, es capaz de manejar palabras que no están en el vocabulario original. Si una palabra no está en el vocabulario, se descompone en subpalabras más pequeñas hasta llegar a unidades que sí están en el vocabulario.

Veamos un ejemplo en Python usando la librería `**[transformers](https://github.com/huggingface/transformers)**`:

```python
import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity

# Cargar el tokenizador y el modelo BERT multilingüe
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

# Texto de entrada en español
text = "¡Hola, embeddings de BERT en idioma español son interesantes!"

# Tokenizar el texto y obtener los IDs de los tokens
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Convertir los IDs de los tokens a tensores y obtener los embeddings
token_ids = torch.tensor([token_ids])
with torch.no_grad():
    outputs = model(token_ids)
    embeddings = outputs.last_hidden_state

# Calcular la similitud del coseno entre dos palabras (por ejemplo, "bert" y "interesantes")
word1_idx = tokens.index("idioma")
word2_idx = tokens.index("español")
similarity = cosine_similarity(embeddings[0][word1_idx].unsqueeze(0), embeddings[0][word2_idx].unsqueeze(0))

print(f"Tokens: {tokens}")
print(f"Similitud entre 'idioma' y 'español': {similarity.item()}")

# Tokens: ['¡', 'Ho', '##la', ',', 'em', '##bed', '##ding', '##s', 'de', 'BE', '##RT', 'en', 'idioma', 'español', 'son', 'interesante', '##s', '!']
# Similitud entre 'idioma' y 'español': 0.6664961576461792
```

### **FastText**

[FastText](https://fasttext.cc/) es un modelo de procesamiento de lenguaje natural desarrollado por Facebook's AI Research (FAIR) lab ([https://arxiv.org/pdf/1607.04606v2.pdf](https://arxiv.org/pdf/1607.04606v2.pdf)). Es una extensión del modelo Word2Vec y también está diseñado para generar representaciones vectoriales (embeddings) de palabras. Aquí hay algunas características y diferencias clave de FastText en comparación con otros modelos como Word2Vec:

1. **Subpalabras**: Una de las características más distintivas de FastText es que representa palabras como bolsas de subpalabras. Esto significa que, en lugar de aprender un único vector para cada palabra, FastText aprende vectores para subpalabras (n-gramas de caracteres) dentro de cada palabra. Por ejemplo, la palabra "chat" podría descomponerse en los n-gramas: "cha", "hat", y otros, dependiendo del rango de n-gramas elegido. Estos n-gramas se utilizan luego para construir el vector de la palabra completa.
2. **Manejo de Palabras Fuera de Vocabulario (OOV)**: Gracias a su enfoque basado en subpalabras, FastText puede generar embeddings para palabras que no estaban en el vocabulario de entrenamiento. Al descomponer palabras desconocidas en n-gramas y sumar los embeddings de estos n-gramas, FastText puede producir representaciones razonables para palabras OOV.
3. **Eficiencia en el Entrenamiento**: FastText es conocido por ser eficiente en términos de tiempo de entrenamiento y memoria, especialmente cuando se compara con modelos más profundos como BERT.
4. **Clasificación de Texto**: Además de generar embeddings de palabras, FastText también se puede utilizar para tareas de clasificación de texto. De hecho, uno de los usos principales de FastText es la clasificación de texto a gran escala, y es conocido por ser rápido y eficiente en esta tarea.
5. **Multilingüe**: Hay versiones preentrenadas de FastText disponibles para múltiples idiomas, lo que facilita su uso en aplicaciones multilingües.
6. **Simplicidad y Accesibilidad**: FastText es relativamente simple en comparación con modelos más recientes y complejos como BERT o transformers. Además, FAIR ha hecho disponible FastText como una biblioteca de código abierto, lo que facilita su uso y adaptación.

Veamos un ejemplo en Colab. Para eso deberemos instalar las librerías `**fasttext` (**https://github.com/facebookresearch/fastText/) ****y `**huggingface_hub`** para descargar el modelo español.

```python
!pip install fasttext
!pip install huggingface_hub

import fasttext
from huggingface_hub import hf_hub_download

# Descargamos el modelo FastText para español (Aprox. 7Gb)
# https://huggingface.co/facebook/fasttext-es-vectors
model_path = hf_hub_download(repo_id="facebook/fasttext-es-vectors", filename="model.bin")
model = fasttext.load_model(model_path)
```

Una vez que tenemos preparado nuestro entorno, podremos usar el modelo:

```python
import numpy as np

# Función para calcular la similitud del coseno
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

  
neighbors = model.get_nearest_neighbors("perro", k=5)
print("neighbors:", neighbors)

# Calcular la similitud del coseno 
similarity = cosine_similarity(model.get_word_vector("rey"), model.get_word_vector("reina"))
print(f"\nSimilitud del coseno entre 'rey' y 'reina': {similarity}")

# Calcular la similitud del coseno 
similarity = cosine_similarity(model.get_word_vector("derecha"), model.get_word_vector("izquierda"))
print(f"\nSimilitud del coseno entre 'derecha' y 'izquierda': {similarity}")

# Calcular la similitud del coseno 
similarity = cosine_similarity(model.get_word_vector("pentagrama"), model.get_word_vector("casualidad"))
print(f"\nSimilitud del coseno entre 'pentagrama' y 'casualidad': {similarity}")
```

Cuyo salida al correr será:

```
neighbors: [(0.8194511532783508, 'gato'), (0.8161919116973877, 'cachorro'), (0.8057317137718201, 'perrito'), (0.7528718709945679, 'perros'), (0.7441142201423645, 'gatito')]

Similitud del coseno entre 'rey' y 'reina': 0.6124091744422913

Similitud del coseno entre 'derecha' y 'izquierda': 0.9428764581680298

Similitud del coseno entre 'pentagrama' y 'casualidad': 0.08354239910840988
```

<aside>
💡 FastText mejora word2vec, ya que descompone cada palabra en n-gramas de caracteres. Por ejemplo, el trigram (n = 3) del término "donde" sería: **`<do, don, onde, nde>`**. Esta descomposición permite que FastText capture información subpalabra, lo que es especialmente útil para adaptarse a diferentes lenguajes y palabras fuera del vocabulario (OOV)

</aside>

### **ELMo (Embeddings from Language Model)**

[ELMo](https://arxiv.org/abs/1802.05365) es un método de embedding de palabras que representa una secuencia de palabras como una secuencia correspondiente de vectores. A diferencia de otros métodos de embedding como Word2Vec y GloVe, que producen representaciones fijas para cada palabra, ELMo genera embeddings que son sensibles al contexto. Esto significa que ELMo produce diferentes representaciones para palabras que se escriben igual pero tienen diferentes significados (homónimos). Por ejemplo, la palabra "banco" tendrá múltiples embeddings dependiendo de si se usa en un contexto relacionado con el deporte o uno relacionado con finanzas.

Las características clave de ELMo son:

1. **Tokens a Nivel de Carácter:** ELMo toma tokens a nivel de carácter como entradas.
2. **Uso de LSTM Bidireccional:** Estos tokens se procesan mediante una LSTM bidireccional para producir embeddings a nivel de palabra.
3. **Sensibilidad al Contexto:** A diferencia de los embeddings producidos por enfoques como "Bag of Words" y métodos vectoriales anteriores como Word2Vec y GloVe, los embeddings de ELMo son sensibles al contexto.

ELMo interpreta el contexto considerando la oración de entrada completa hacia adelante y hacia atrás usando 2 capas de modelos de lenguaje bidireccional (biLM). En la Figura se muestra la arquitectura de ELMo:

![Fuente: [https://techblog.ezra.com/different-embedding-models-7874197dc410](https://techblog.ezra.com/different-embedding-models-7874197dc410)](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2028.png)

Fuente: [https://techblog.ezra.com/different-embedding-models-7874197dc410](https://techblog.ezra.com/different-embedding-models-7874197dc410)

Las palabras de entrada se convierten en una secuencia de id’s y se integran mediante una CNN. En cada paso de tiempo, la representación vectorial de una palabra dada se pasa a la capa LSTM hacia adelante y hacia atrás con conexiones residuales. Las salidas de estos dos LSTM se pasan a otra capa de LSTM hacia adelante y hacia atrás. Finalmente, hay una capa softmax que se usa para predecir la siguiente palabra de la red de paso hacia adelante y la palabra anterior de la red hacia atrás. La contribución de cada capa intermedia en la predicción final se pondera y normaliza durante el entrenamiento.

Veamos un ejemplo de como utilizar ELMo en Colab. Primero instalamos librerías necesarias y descargamos el modelo español (https://github.com/HIT-SCIR/ELMoForManyLangs):

```python
!pip install elmoformanylangs
# Descarga modelo en español
!wget http://vectors.nlpl.eu/repository/11/145.zip
!unzip 145.zip -d elmo_es
# Fix para evitar error en Colab (https://github.com/HIT-SCIR/ELMoForManyLangs/issues/100)
!pip uninstall overrides 
!pip install overrides==3.1.0
```

Cargamos el modelo desde la carpeta donde fue descomprimido:

```python
from elmoformanylangs import Embedder

# Ruta al directorio del modelo descargado
model_dir = '/content/elmo_es'

# Inicializar el modelo
e = Embedder(model_dir)
```

Y aquí probaremos como ELMo nos permite obtener diferentes embeddings según el contexto:

```python
# Lista de frases que contienen la palabra "banco" en diferentes contextos
sents = [
    ['Fui', 'al', 'banco', 'a', 'retirar', 'dinero'],  # Contexto financiero
    ['Me', 'senté', 'en', 'el', 'banco', 'del', 'parque'],  # Contexto de asiento
    ['El', 'jugador', 'estuvo', 'en', 'el', 'banco', 'de', 'suplentes']  # Contexto deportivo
]

# Le ponemos nombres a los contextos, para luego identificarlos
contexts = ['financiero', 'asiento', 'deportivo']

# Obtener embeddings
embeddings = e.sents2elmo(sents)

# Imprimir los embeddings para la palabra "banco" en cada contexto
for idx, (sentence, embedding) in enumerate(zip(sents, embeddings)):
    # El embedding de la palabra "banco" se encuentra en la posición donde aparece en cada frase. 
    position = sentence.index('banco')
    
    # Obtener el embedding de la palabra "banco" en esa posición
    banco_embedding = embedding[position]

    # Determinar el contexto basado en la frase
    context = contexts[idx]

    print(f"Embedding para 'banco' en contexto '{context}':\n{banco_embedding}\n")
```

El resultado será:

```
Embedding para 'banco' en contexto 'financiero':
[ 0.01047617 -2.0515497   0.52021605 ...  0.3223724  -0.08630773
  0.78804964]

Embedding para 'banco' en contexto 'asiento':
[ 0.14607799 -1.7709602   0.26095876 ... -0.837271    0.01180764
  0.59168345]

Embedding para 'banco' en contexto 'deportivo':
[-0.28225622 -1.9305519   0.10399798 ... -0.46536005 -0.01572029
  0.46354493]
```

> Más información:
[Video de DotCSV](https://www.youtube.com/watch?v=RkYuH_K7Fx4)
> 

[https://www.youtube.com/watch?v=RkYuH_K7Fx4](https://www.youtube.com/watch?v=RkYuH_K7Fx4)

### Resumen

Hasta aquí hemos visto diversos métodos que nos permiten llevar palabras u oraciones a vectores. Veamos una tabla comparativa de las características principales:

| Modelo/Método | Enfoque | Captura Similitud Semántica |
| --- | --- | --- |
| One-Hot Encoding | Palabra |  |
| Count Vectorizer | Oración |  |
| Hash Vectorizer | Oración |  |
| TF-IDF | Oración |  |
| Word2Vec | Palabra |                      ✓ |
| GloVe | Palabra |                      ✓ |
| BERT Embeddings | Palabra/Oración |                      ✓ |
| FastText | Palabra (y subpalabra) |                      ✓ |
| ELMo | Palabra (Contextual) |                      ✓ |

## 3. Visualización de embeddings

Visualizar word embeddings es una tarea importante en el procesamiento de lenguaje natural (NLP) para entender cómo las palabras están representadas en un espacio vectorial. Dado que los embeddings suelen tener muchas dimensiones (por ejemplo, 100, 200, 300 o más), se utilizan técnicas de reducción de dimensionalidad para visualizarlos en un espacio bidimensional o tridimensional. Aquí mencionamos algunos de los métodos más comunes de reducción de dimensionalidad:

- **[t-SNE (t-Distributed Stochastic Neighbor Embedding)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)**:
t-SNE es una técnica popular para visualizar datos de alta dimensión. Reduce la dimensionalidad mientras intenta mantener las relaciones de vecindad entre los puntos. Es especialmente útil para visualizar cómo los embeddings de palabras están agrupados en el espacio.
- **[PCA (Principal Component Analysis)](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_componentes_principales)**:
PCA es un método estadístico que transforma los datos a un nuevo sistema de coordenadas en el que las primeras coordenadas capturan la mayor cantidad de variación en los datos. Al tomar las dos o tres primeras componentes principales, se puede visualizar la estructura de los embeddings en un espacio bidimensional o tridimensional.

Vemos un ejemplo de visualización de embeddings usando el modelo word2vec y reducción a 2 dimensiones usando T-SNE. Primero preparamos el entorno en Colab:

```python
!pip install gensim
!wget https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5.bin.gz
```

Luego podremos correr el siguiente ejemplo:

```python
from gensim.models import KeyedVectors

# Carga un modelo Word2Vec preentrenado (asegúrate de tener el archivo en tu directorio)
model = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin.gz', binary=True)

# Lista de palabras para visualizar
words = ["rey", "reina", "hombre", "mujer", "niño", "niña", "príncipe", "princesa"]

# Obtener embeddings para las palabras
embeddings = np.array([model[word] for word in words])

# Usar t-SNE para reducir la dimensionalidad
tsne = TSNE(n_components=2, random_state=0, perplexity=6)
embeddings_2d = tsne.fit_transform(embeddings)

# Visualizar los embeddings en 2D
plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], marker='o', color='red')
    plt.text(embeddings_2d[i, 0]+0.1, embeddings_2d[i, 1], word, fontsize=12)
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.title('Visualization of Word2Vec Embeddings using t-SNE')
plt.grid(True)
plt.show()
```

Este código cargará un modelo pre-entrenado de **`word2vec`**, obtendrá embeddings para una lista de palabras y luego visualizará estos embeddings en un espacio 2D usando t-SNE.

Y el resultado será:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2029.png)

Como variante, también podemos usar [PCA de Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) y `**plotly`** para graficar en 3D y navegación interactiva:

```python
from gensim.models import KeyedVectors
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

# Carga un modelo Word2Vec preentrenado
model = KeyedVectors.load_word2vec_format('SBW-vectors-300-min5.bin.gz', binary=True)

# Lista de palabras para visualizar
words = ["rey", "reina", "hombre", "mujer", "niño", "niña", "príncipe", "princesa"]

# Obtener embeddings para las palabras
embeddings = np.array([model[word] for word in words])

# Usar PCA para reducir la dimensionalidad a 3D
pca = PCA(n_components=3)
embeddings_3d = pca.fit_transform(embeddings)

# Convertir los embeddings a un DataFrame para visualizar con plotly
import pandas as pd
df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
df['word'] = words

# Visualizar los embeddings en 3D usando plotly
fig = px.scatter_3d(df, x='x', y='y', z='z', text='word', color='word', size_max=18, opacity=0.7)
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers+text'))
fig.show()
```

Y el resultado será:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2030.png)

> Más información:
[https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html)
[https://projector.tensorflow.org/](https://projector.tensorflow.org/)
[](https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/index.html)[https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5C](https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5)
> 

## 4. Embeddings en conjuntos de palabras

Los "sentence embeddings" (o incrustaciones de oraciones) se refieren a la representación vectorial de oraciones completas, párrafos o incluso documentos más largos. Mientras que las incrustaciones de palabras (word embeddings) representan palabras individuales en un espacio vectorial, las incrustaciones de oraciones buscan capturar el **significado semántico y la estructura de oraciones completas en un vector.**

Las siguientes son algunas características y detalles sobre las incrustaciones de oraciones:

1. **Objetivo**: El objetivo principal de las incrustaciones de oraciones es capturar el significado semántico de una oración completa en un vector de dimensiones fijas.
2. **Aplicaciones**: Las incrustaciones de oraciones se utilizan en diversas tareas de NLP, como la clasificación de texto, la búsqueda por similitud semántica entre oraciones, la respuesta automática a preguntas, la traducción automática, entre otras.
3. **Métodos**:
    - **Modelos Preentrenados**: Existen modelos como BERT, RoBERTa, y Universal Sentence Encoder que pueden generar incrustaciones de oraciones directamente o mediante la agregación de incrustaciones de palabras.
    - **Promedio de Word Embeddings**: Una técnica simple pero limitada, es tomar el promedio (o suma ponderada) de las incrustaciones de palabras en una oración para obtener una incrustación de oración.
    - **Modelos Específicos**: Modelos como InferSent han sido entrenados específicamente para generar incrustaciones de oraciones.
4. **Ventajas sobre Word Embeddings**:
    - **Captura de Contexto**: Mientras que las incrustaciones de palabras representan el significado de una palabra en general, las incrustaciones de oraciones pueden capturar el contexto en el que se utilizan las palabras, lo que puede ser crucial para entender el significado de una oración.
    - **Representación Unificada**: Proporcionan una representación unificada para oraciones de diferentes longitudes.
5. **Desafíos**:
    - **Variedad de Información**: Una oración puede contener una variedad de información, desde hechos y opiniones hasta emociones y relaciones entre conceptos. Capturar toda esta información en un vector de tamaño fijo es un desafío.
    - **Longitud Variable**: Las oraciones pueden tener longitudes variables, lo que puede dificultar la obtención de una representación coherente.
6. **Uso en Modelos de Aprendizaje Profundo**: Las incrustaciones de oraciones se pueden utilizar como entrada para modelos de aprendizaje profundo, como redes neuronales recurrentes (RNN) o redes neuronales de atención, para tareas más avanzadas.

Una técnica “ingenua” para hacer embeddings de oraciones es promediar los embeddings de palabras en una oración y usar el promedio como representación de la oración completa. Este enfoque tiene algunos desafíos.

![Promediar vectores de palabras para incrustar oraciones](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2031.png)

Promediar vectores de palabras para incrustar oraciones

Entendamos estos desafíos con algunos ejemplos de código usando la biblioteca spacy. Primero instalamos `**spacy`** y creamos un objeto nlp para cargar la versión mediana de su modelo.

```python
# pip install spacy
# python -m spacy download en_core_web_md

import en_core_web_md
nlp = en_core_web_md.load()
```

**Pérdida de información**
Si calculamos la similitud del coseno de los documentos que se muestran a continuación utilizando vectores de palabras promediados, la similitud es bastante alta incluso si la segunda oración tiene una sola palabra “It” y no tiene el mismo significado que la primera oración.

```python
nlp('It is cool').similarity(nlp('It'))

0.8963861908844291
```

**El orden no afecta**
En este ejemplo, intercambiamos el orden de las palabras en una oración, lo que da como resultado una oración con un significado diferente. Sin embargo, la similitud obtenida a partir de vectores de palabras promediados es del 100%:

```python
nlp('this is cool').similarity(nlp('is this cool'))

1.0
```

Podríamos solucionar algunos de estos desafíos con ingeniería manual de funciones, como omitir las “stop-words”, ponderar las palabras según sus puntuaciones TF-IDF, agregar n-gramas para respetar el orden al promediar, concatenar embeddings, etc. .

Una línea de pensamiento diferente es entrenar un modelo “end-to-end” para obtener incrustaciones de oraciones

### Universal Sentence Encoder

El [Universal Sentence Encoder (USE)](https://arxiv.org/pdf/1803.11175.pdf) es un modelo desarrollado por Google que tiene como objetivo convertir oraciones completas en representaciones vectoriales de longitud fija. Estas representaciones, conocidas como "embeddings", pueden capturar en cierto modo, la esencia semántica de la oración.

A un nivel general, la idea es diseñar un codificador que resuma cualquier oración dada en una representación de oración de 512 dimensiones. Utilizamos esta misma representación para resolver múltiples tareas y, en función de los errores que comete en ellas, actualizamos la representación de la oración. Dado que la misma representación debe funcionar en múltiples tareas genéricas, solo capturará las características más informativas y descartará el ruido. La intuición es que esto resultará en una representación genérica que se transferirá de manera universal a una amplia variedad de tareas de NLP, como la relación, agrupación, detección de paráfrasis y clasificación de texto.

![Fuente: [https://amitness.com/2020/06/universal-sentence-encoder/](https://amitness.com/2020/06/universal-sentence-encoder/)](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2032.png)

Fuente: [https://amitness.com/2020/06/universal-sentence-encoder/](https://amitness.com/2020/06/universal-sentence-encoder/)

Si aplicamos un modelo pre-entrenado, para calcular la similitud entre diferentes oraciones, podríamos obtener un heatmap como el siguiente:

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2033.png)

Algo muy interesante del modelo, es que incluso podemos utilizar un modelo multilingüe, lo que permite comparar similitud semántica entre oraciones en diferentes idiomas. 

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2034.png)

Veamos un ejemplo:

```python
# Primero preparamos el entorno en Colab
%%capture
!pip install "tensorflow-text==2.12.*"
!pip install bokeh==2.4.3
!pip install simpleneighbors[annoy]
!pip install tqdm
```

Luego ejecutamos el siguiente ejemplo:

```python
import bokeh
import bokeh.models
import bokeh.plotting
import numpy as np
import os
import tensorflow.compat.v2 as tf
import tensorflow_hub as hub

def visualize_similarity(embeddings_1, embeddings_2, labels_1, labels_2,
                         plot_title,
                         plot_width=1200, plot_height=600,
                         xaxis_font_size='12pt', yaxis_font_size='12pt'):

  assert len(embeddings_1) == len(labels_1)
  assert len(embeddings_2) == len(labels_2)

  # arccos based text similarity (Yang et al. 2019; Cer et al. 2019)
  sim = 1 - np.arccos(
      sklearn.metrics.pairwise.cosine_similarity(embeddings_1,
                                                 embeddings_2))/np.pi

  embeddings_1_col, embeddings_2_col, sim_col = [], [], []
  for i in range(len(embeddings_1)):
    for j in range(len(embeddings_2)):
      embeddings_1_col.append(labels_1[i])
      embeddings_2_col.append(labels_2[j])
      sim_col.append(sim[i][j])
  df = pd.DataFrame(zip(embeddings_1_col, embeddings_2_col, sim_col),
                    columns=['embeddings_1', 'embeddings_2', 'sim'])

  mapper = bokeh.models.LinearColorMapper(
      palette=[*reversed(bokeh.palettes.YlOrRd[9])], low=df.sim.min(),
      high=df.sim.max())

  p = bokeh.plotting.figure(title=plot_title, x_range=labels_1,
                            x_axis_location="above",
                            y_range=[*reversed(labels_2)],
                            width=plot_width, height=plot_height,
                            tools="save",toolbar_location='below', tooltips=[
                                ('pair', '@embeddings_1 ||| @embeddings_2'),
                                ('sim', '@sim')])
  p.rect(x="embeddings_1", y="embeddings_2", width=1, height=1, source=df,
         fill_color={'field': 'sim', 'transform': mapper}, line_color=None)

  p.title.text_font_size = '12pt'
  p.axis.axis_line_color = None
  p.axis.major_tick_line_color = None
  p.axis.major_label_standoff = 16
  p.xaxis.major_label_text_font_size = xaxis_font_size
  p.xaxis.major_label_orientation = 0.25 * np.pi
  p.yaxis.major_label_text_font_size = yaxis_font_size
  p.min_border_right = 300

  bokeh.io.output_notebook()
  bokeh.io.show(p)

# El modelo que usaremos es uno multilingüe (de 16 idiomas)
module_url = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
model = hub.load(module_url)

def embed_text(input):
  return model(input)

# Algunos textos de diferente longitud en diferentes idiomas.
arabic_sentences = ['كلب', 'الجراء لطيفة.', 'أستمتع بالمشي لمسافات طويلة على طول الشاطئ مع كلبي.']
chinese_sentences = ['狗', '小狗很好。', '我喜欢和我的狗一起沿着海滩散步。']
english_sentences = ['dog', 'Puppies are nice.', 'I enjoy taking long walks along the beach with my dog.']
french_sentences = ['chien', 'Les chiots sont gentils.', 'J\'aime faire de longues promenades sur la plage avec mon chien.']
german_sentences = ['Hund', 'Welpen sind nett.', 'Ich genieße lange Spaziergänge am Strand entlang mit meinem Hund.']
italian_sentences = ['cane', 'I cuccioli sono carini.', 'Mi piace fare lunghe passeggiate lungo la spiaggia con il mio cane.']
japanese_sentences = ['犬', '子犬はいいです', '私は犬と一緒にビーチを散歩するのが好きです']
korean_sentences = ['개', '강아지가 좋다.', '나는 나의 개와 해변을 따라 길게 산책하는 것을 즐긴다.']
russian_sentences = ['собака', 'Милые щенки.', 'Мне нравится подолгу гулять по пляжу со своей собакой.']
spanish_sentences = ['perro', 'Los cachorros son agradables.', 'Disfruto de dar largos paseos por la playa con mi perro.']

# Calculamos los embeddings
ar_result = embed_text(arabic_sentences)
en_result = embed_text(english_sentences)
es_result = embed_text(spanish_sentences)
de_result = embed_text(german_sentences)
fr_result = embed_text(french_sentences)
it_result = embed_text(italian_sentences)
ja_result = embed_text(japanese_sentences)
ko_result = embed_text(korean_sentences)
ru_result = embed_text(russian_sentences)
zh_result = embed_text(chinese_sentences)

visualize_similarity(en_result, es_result, english_sentences, spanish_sentences, 'Similaridad Español-Inglés')
```

Y obtendremos un gráfico como el siguiente: 

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2035.png)

### Doc2Vec

Doc2Vec es una técnica de modelado que permite representar documentos completos, ya sean frases, párrafos o artículos, como vectores en un espacio multidimensional. Es una extensión del método Word2Vec, que se utiliza para representar palabras individuales en un espacio vectorial. Fue introducido por [Le y Mikolov en 2014](https://arxiv.org/pdf/1405.4053.pdf).

Características principales:

1. **Concepto Básico:** Al igual que Word2Vec representa palabras en un espacio vectorial de manera que palabras con significados similares estén cerca unas de otras, Doc2Vec hace lo mismo pero con documentos completos. Esto significa que, por ejemplo, dos artículos sobre temas similares tendrán representaciones vectoriales cercanas en el espacio.
2. **Cómo Funciona:** Al igual que Word2Vec, que representa palabras en un espacio vectorial de manera que palabras con contextos similares estén cerca entre sí, Doc2Vec busca representar documentos completos como vectores en un espacio similar.
3. **Modelo de Entrenamiento**: Hay dos enfoques principales para Doc2Vec:
    - **Distributed Memory (DM)**: En este modelo, se utiliza un vector de párrafo junto con los vectores de palabras para predecir la siguiente palabra en una ventana de palabras. El vector del párrafo actúa como una memoria que recuerda qué se ha dicho en el párrafo.
    
    ![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2036.png)
    
    - **Distributed Bag of Words (DBOW)**: En este enfoque, se ignora el contexto de las palabras y se utiliza el vector del documento para predecir palabras que aparecen en el documento.
    
    ![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2037.png)
    
4. **Aplicaciones:** Una vez que los documentos están representados como vectores, se pueden realizar diversas tareas, como clasificación de documentos, agrupación (clustering), recomendación de contenido y búsqueda semántica. Por ejemplo, si tenemos un vector para un artículo sobre "inteligencia artificial", podemos encontrar rápidamente otros artículos relacionados buscando los vectores más cercanos en el espacio.
5. **Relación con Word2Vec:** Mientras que Word2Vec se centra en aprender representaciones vectoriales de palabras basadas en su contexto en frases y párrafos, Doc2Vec amplía este concepto para aprender representaciones de documentos completos. Para hacer esto, Doc2Vec introduce un identificador único para cada documento y lo entrena junto con los vectores de palabras.

Veamos un ejemplo de como aplica Doc2Vec:

```python
# Instalar la librería
#!pip install gensim

# Importar las librerías necesarias
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Datos de ejemplo en español
data = ["Soy un estudiante de inteligencia artificial.",
        "Me gusta estudiar matemáticas y física.",
        "El fútbol es un deporte popular en Argentina.",
        "El asado era un plato típico en Argentina."]

# Tokenizar los datos y etiquetarlos
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

# Configurar parámetros para el modelo
model = Doc2Vec(vector_size=20, window=2, min_count=1, workers=4, epochs=1000)

# Construir el vocabulario
model.build_vocab(tagged_data)

# Entrenar el modelo
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Obtener el vector de un documento
vector = model.infer_vector(word_tokenize("Me gustaría saber cuál es el plato típico en Argentina.".lower()))

print(vector)
```

Obtendremos como resultado:

```python
[-0.31152847  0.45640787  0.5139149  -0.4611784   0.8048704  -1.2181339
  0.3892808   0.08421853 -0.01324374 -1.5394584  -0.36275834  0.37329495
 -1.6075045  -1.8461896  -0.66440237  0.6069402  -0.42048925 -0.8328456
 -0.65525806  0.63359654]
```

Este código tokeniza y etiqueta los datos en español, configura y entrena un modelo **`Doc2Vec`**, y finalmente infiere un vector para una nueva frase.

Es importante tener en cuenta que este es un ejemplo muy básico con un pequeño conjunto de datos. En aplicaciones reales, necesitaríamos un conjunto de datos mucho más grande y tal vez ajustar los parámetros para obtener representaciones significativas. Además, es recomendable realizar más preprocesamiento en el texto, como eliminar signos de puntuación, números y stopwords.

La configuración del modelo de define con los siguiente parámetros:

- **`vector_size`**: Dimensión del vector de salida (20 en este caso).
- **`window`**: Máxima distancia entre la palabra actual y la predicha dentro de una oración.
- **`min_count`**: Ignora todas las palabras con una frecuencia total menor a esta.
- **`workers`**: Número de núcleos de la CPU para usar durante el entrenamiento.
- **`epochs`**: Número de iteraciones sobre el corpus durante el entrenamiento.

Una vez que generamos un vector con una nueva frase, podemos realizar una búsqueda por similitud con las frases que usamos en el entrenamiento:

```python
# Encontrar el documento más similar al vector inferido
similares = model.docvecs.most_similar([vector], topn=1)  # topn=1 para obtener el más similar solamente

# La función most_similar devuelve una lista de tuplas (etiqueta, similitud)
etiqueta_similar, similitud = similares[0]

print(f"La frase más cercana es: '{data[int(etiqueta_similar)]}' con una similitud de {similitud}.")
```

Lo que hace este código es:

1. Utiliza el método **`most_similar`** de **`docvecs`** del modelo para encontrar los documentos más similares al vector inferido.
2. Selecciona el documento más similar (dado que **`topn=1`**).
3. Imprime el documento y su similitud con el vector inferido.

La salida al correr el script, será:

```
La frase más cercana es: 'El asado era un plato típico en Argentina.' con una similitud de 0.9720115661621094.
```

![Untitled](Unidad%202%20-%20Representacio%CC%81n%20Vectorial%20de%20Texto%206ad0dcf3b18a447e8b2989b325b57d3f/Untitled%2038.png)