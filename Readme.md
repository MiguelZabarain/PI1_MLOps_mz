
# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h2 align=center> Miguel Zabarain DataScience_FT_18 </h2>
# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

¡Hola, soy Miguel Zabaraín, y presento a continuación el primer proyecto individual de la etapa de labs. En esta ocasión, se realizó un trabajo propio del rol de un ***MLOps Engineer***.  

<hr>  

## Contexto y Objetivos

Se tiene acceso a tres archivos con datos de la plataforma de videojuesgos STEAM. El primero de ellos contiene información de las calificaciones / reseñas hechas por los usuarios del sitio WEB. El segundo es un catálogo con información de cada juego ofrecido por STEAM en su plataforma. Finalmente, el tercer archivo contiene estadísticas de la cantidad de minutos jugados por cada usuario en cada juego desde el momento de su lanzamiento, y los minutos de juego acumulados en las últimas dos semanas.

Se pide desarrollar cinco funciones y usar FastAPI para crear muy agilmente una aplicación que permita a usuarios interesados, acceder a la información que esas funciones suministran. Los usuarios deberían ser capaces de acceder a las funciones desde cualquier dispositivo conectado a INTERNET, por lo que se usa Render para suplir esta necesidad.

Adicional a lo anterior, se pide poner a disposición del usuario -desde la solución creada con FastAPI y disponibilizada con Render- un servicio de recomendaciones, por medio del cual, el usuario interesado reciba una lista con cinco videojuegos que guarden similitud con uno suministrado.

## Hoja de ruta

<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

La solución propuesta tiene tres grandes componentes; una corresponde a la **Ingeniería de Datos**; otra a un proceso de **Aprendizaje Automático o Machine Learning (ML)**; y finalmente un componente **DevOps**.

## Componente Ingeniería de Datos
### Proceso ETL
Los datos originalmente recibidos provienen del siguiente enlace: https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj

Allí se encontraron los tres archivos referidos en la descripción del **Contexto y Objetivos**.
Aunque los archivos se suponían JSON por la extensión de los mismos (.json), en realidad resultaron ser NDJSON, lo que supuso un problema para la exploración inicial de tales archivos.

Todo el trabajo de ETL (Extraction, Tansformation & Load) está documentado en [ETL_on_json_files.ipynb](Code/ETL_on_json_files.ipynb).

El proceso de ETL aplicado, toma los archivos originales de la ruta 'Datasets/Original_Data_STEAM/' y los convierte en verdaderos archivos JSON con elementos JSON válidos, almacenandolos en la ruta 'Datasets/Fixed_Data_STEAM/'. Finalmente, los archivos pueden ser leidos desde python y se finaliza el proceso de ETL convirtiendo los archivos 'Fixed' a formaro **parquet** para evitar las demoras en la carga del tercero de ellos. Como resultado de este proceso el archivo que pesaba originalmente alrededor de 500 MB y tardaba cerca de 60 segundos para cargar en un DataFrame, pasó a pesar cerca de 50 MB y se cargaba en menos de 2 segundos. La ruta a los archivos **parquet** es 'Datasets/Clean_Parquet_Data_Steam/'.

***IMPORTANTE:***
1. En relación a las Transformaciones en el proceso de ETL, se me pidio expresamente lo siguiente, cito textualmente:
"Para este MVP no se te pide transformaciones de datos(` aunque encuentres una motivo para hacerlo `) pero trabajaremos en leer el dataset con el formato correcto. Puedes eliminar las columnas que no necesitan para responder las consultas o preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo."

2. En relación con los archivos contenidos en 'Datasets', debido al tamaño de los mismos, solo se dejó en el repositorio los que estan dentro de 'Datasets/Clean_Parquet_Data_Steam/', que son los requeridos por las Funciones con las que se creó la solución con FastAPI y Render -incluida la que hace la Recomendación de Juegos.
A través del siguiente enlace, se tiene acceso al 'Datasets' completo, en caso de tener interés en ejecutar el código relativo al ETL_on_json_files.ipynb.

Datasets: https://drive.google.com/file/d/1C3cPov6e-AtVzh7rP3FVe4uGyfmr7l7J/view?usp=sharing

### Feature Engineering
En [Feature_Engineering.ipynb](Code/Feature_Engineering.ipynb) se encuentra documentado el detalle de este proceso. Se refiere a que el DataFrame construido con el archivo que incluye reseñas de juegos hechos por distintos usuarios, tiene la columna 'review', la cual es un texto con la respectiva reseña hecha por el usuario. Se me pide eliminar la columna luego de crear la columna ***'sentiment_analysis'*** aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna, que reemplaza la columna 'review' busca facilitar el trabajo del modelo de machine learning y el análisis de datos. De no ser posible este análisis por estar ausente la reseña escrita, ***'sentiment_analysis'*** debe tomar el valor de '1'.

## Componente Aprendizaje Automático o Machine Learning (ML)
Se requirió construir un modelo de recomendación de juegos utilizando similitud del coseno. Entre diferentes alternativas, se optó por un Sistema de Recomendación item-item, en el que un usuario interesado suministra el 'id' de un videojuego, y recibe una lista con cinco videojuegos similares.

### EDA
La decisión de realizar un Sistema de Recomendación item-item, facilitó el Exploratory Data Analysis (EDA), por cuanto un solo archivo de datos -el que contiene el Catálogo de juegos- es requerido para el entrenamiento del modelo. En [EDA.ipynb](Code/EDA.ipynb) se encuentra documentado el detalle de este proceso. 

### Sistema de Recomendación item-item
Todo el detalle de cómo se desarrolló el Modelo del Sistema de Recomendación item-item está documentado en [ML_Model.ipynb](Code/ML_Model.ipynb).

## Componente DevOps
Se requirió que la Solución con FastAPI estuviera conformada por cinco funciones que responden a sendas consultas para ser resueltas con los datos ya tratados.

Las funciones requeridas se detallan en en [Endpoints_FastAPI.ipynb](Code/Endpoints_FastAPI.ipynb)

El servicio de recomendación desarrollado se presta con la función 'recomendacion_juego(id)' que está en [ML_Model.ipynb](Code/ML_Model.ipynb).

Todas estas funciones se consolidan en el archivo [main.py](main.py) para ser levantadas con FastAPI y disponibilizadas a través de ***Render***.

#### **Como crear soluciones con FastAPI en un ambiente virtual habilitando el control de versiones con GIT.**    

Desde la consola Gitbash o la terminal de VSC...    

0) Crea un repositorio GIT en la carpeta local que contendrá el proyecto:    
git init

1) Crea el entorno virtual (my_venv):    
python -m venv my_venv

2) Activa el entorno virtual:     
source my_venv/Scripts/activate

3) Crea el archivo .gitignore    
touch .gitignore

4) Edita .gitignore para agregar el entorno virtual y así quede excluido del repositorio GIT:    
my_venv    
/my_venv    
my_venv/    
/my_venv/    

5) Con el entorno virtual activo, instala las librerías requeridas por el código en main.py:    
pip install fastapi uvicorn pandas scikit-learn pyarrow fastparquet    

6) Asegurate que el archivo 'main.py' que contiene la API desarrollada, se encuentra al mismo nivel de las carpetas 'my_venv' y .git.    
La estructura del contenido de main.py debe ser:    

    from fastapi import FastAPI    
    app = FastAPI()    
    
    A continuación el endpoint...    
    @app.get("/PlayTimeGenre/{genero}")    
    def PlayTimeGenre(genero: str):    
    ...     

7) Ejecuta y prueba tu solución software creada con FastAPI. Para ello, levanta el servidor uvicorn que aloja la API, usando alguna de las siguientes líneas desde la ventana Git Bash:    
uvicorn main:app --reload          
uvicorn main:app --reload > output.txt        
uvicorn main:app --port 10000 --reload         

8) Con el servidor uvicorn arriba, puedes dejar momentaneamente la ventana Git Bash para probar tu API en un web browser usando la siguiente URL:    
localhost:8000/docs.    

    Ahora puedes probar cada uno de los endpoints que creaste en tu API.    

9) Cuando todo funcione OK, es tiempo de volver a la ventana Git Bash para crear un archivo 'requirements.txt' al mismo nivel de las carpetas 'my_venv' y .git:    
pip freeze > requirements.txt    

10) Si lo estimas conveniente, sube el repositorio local al repo público en GitHub.

El siguiente enlace da acceso a un video donde se explica en detalle como implementar soluciones con FastAPI:   
https://www.youtube.com/watch?v=12NIB_RjxMo

## Video
El siguiente enlace da acceso al video corto -menos de 7 minutos- que presenta el proyecto funcionando: 

https://drive.google.com/file/d/1EI3eZHm4opUYNF1iUbKJSSe5HAppFzNv/view?usp=sharing
  
<br/>
