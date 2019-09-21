# TCUE - Sentiment

<div>
  <img src="https://github.com/alvarob96/tcue/blob/master/resources/tcue_es.png"><br>
</div>

## Introducción

A continuación se presentará la plataforma de sorporte del TCUE, la cual se encargará de ingerir tweets en tiempo real 
(_streaming_) con [twipper](https://github.com/alvarob96/twipper), los cuales son almacenados en **MongoDB**. De este modo,
la plataforma muestra los tweets almacenados, y da la posibilidad al usuario de realizar una clasificación supervisada
de los mismos en sentimientos: positivo, negativo o neutral. Dichos resultados serán almacenados en otra colección de 
MongoDB, los cuales serán extraidos en la fase de análisis de cara a crear un modelo de análisis de sentimientos en 
español.

## Instalación/Configuración

Para la instalación y configuración se requiere la versión 3.x de Python y la instalación de los requisitos especificados 
en el fichero [requirements.txt](https://github.com/alvarob96/tcue/blob/master/tcue_sentiment/requirements.txt), lo cual 
podrá realizarse mediante la orden: ``python3 -m pip install -r requirements``. 

Dado que la ingesta no se realiza directamente desde [twipper](https://github.com/alvarob96/twipper) sino que el 
resultado de esta se vuelca en **MongoDB**, será necesario introducir las credenciales de acceso a Mongo en el fichero: 
[tcue_sentiment/tcue_classifier/views.py](https://github.com/alvarob96/tcue/blob/master/tcue_sentiment/tcue_classifier/views.py), 
por lo que será necesario modificar los valores de los campos de acceso al inicio del fichero, de la forma: 

````python
_USER = 'MONGO USERNAME'
_PASS = 'MONGO PASSWORD'
_IP = 'MONGO IP'
_PORT = 'MONGO PORT'

mongodb = MongoClient(f'mongodb://{_USER}:{_PASS}@{_IP}:{_PORT}')
````

Además, será necesario incluir en el fichero 
[settings.py](https://github.com/alvarob96/tcue/blob/master/tcue_sentiment/tcue_sentiment/settings.py) y añadir en la línea
``ALLOWED_HOSTS = []`` la IP o nombre del dominio del servidor dónde se va a lanzar el Django.

![tcue_sentiment_setup](https://github.com/alvarob96/tcue/blob/master/resources/tcue_sentiment_setup.gif)

Tras haber modificado el fichero de configuración previamente, añadiendo la IP o IPs válidas, se lanzará el servidor con
la orden: ``python3 manage.py runserver 0.0.0.0:8000``, siendo 0.0.0.0 la IP pública y 8000 el puerto habilitado para 
desplegar el servidor. Antes de lanzar el servidor Django, conviene ejecutar la orden: ``python3 manage.py migrate``; todo
esto estando situado en el directorio [tcue_sentiment/](https://github.com/alvarob96/tcue/tree/master/tcue_sentiment).

## Uso

Se presentará un caso de uso de cara a mostrar todas las funcionalidades del sistema, recuperando tweets ingeridos en
**streaming** a través de [twipper](https://github.com/alvarob96/twipper), donde dichos tweets serán insertados en 
una base de datos _noSQL_, _MongoDB_. A lo largo de la siguiente demo, se podrán observar las principales 
funcionalidades de la plataforma a la hora de clasificar tweets en sentimientos (**positivo**, **negativo** o **neutral**).

![tcue_sentiment_demo](https://github.com/alvarob96/tcue/blob/master/resources/tcue_sentiment_demo.gif)

## Autor

**Álvaro Bartolomé del Canto**