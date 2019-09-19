# TCUE - Platform

<div>
  <img src="https://github.com/alvarob96/tcue/blob/master/resources/tcue_es.png"><br>
</div>

## Introducción

A continuación se presentará la plataforma del TCUE, sobre la cual se realizarán las peticiones al servidor desde un 
cliente web. La plataforma web en Django consiste en la visualización tanto de los datos recuperados por 
[investpy](https://github.com/alvarob96/investpy) como de los resultados y análisis producto de aplicar las técnicas 
de análisis técnico e Inteligencia Artificial.

## Instalación/Configuración

Para la instalación y configuración se requiere la versión 3.x de Python y la instalación de los requisitos especificados 
en el fichero [requirements.txt](https://github.com/alvarob96/tcue/blob/master/tcue_platform/requirements.txt), lo cual 
podrá realizarse mediante la orden: ``python3 -m pip install -r requirements``. Además, será necesario incluir en el fichero 
[settings.py](https://github.com/alvarob96/tcue/blob/master/tcue_platform/tcue_platform/settings.py) y añadir en la línea
``ALLOWED_HOSTS = []`` la IP o nombre del dominio del servidor dónde se va a lanzar el Django.

![tcue_platform_setup](https://github.com/alvarob96/tcue/blob/master/resources/tcue_platform_setup.gif)

Tras haber modificado el fichero de configuración previamente, añadiendo la IP o IPs válidas, se lanzará el servidor con
la orden: ``python3 manage.py runserver 0.0.0.0:8000``, siendo 0.0.0.0 la IP pública y 8000 el puerto habilitado para 
desplegar el servidor. Antes de lanzar el servidor Django, conviene ejecutar la orden: ``python3 manage.py migrate``; todo
esto estando situado en el directorio [tcue_platform/](https://github.com/alvarob96/tcue/tree/master/tcue_platform).

## Uso

Se presentará un caso de uso de cara a mostrar todas las funcionalidades del sistema, así como la gestión de errores.
A lo largo de la siguiente demo, se podrán observar las principales funcionalidades de la plataforma, así como una serie
de resultados propuestos a modo de ejemplo.

![tcue_platform_demo](https://github.com/alvarob96/tcue/blob/master/resources/tcue_platform_demo.gif)

## Problemas Frecuentes

Uno de los problemas más frecuentes es instalar el paquete [TA-Lib](https://mrjbq7.github.io/ta-lib/func.html) para el 
cálculo de los factores técnicos que serán utilizados en combinación con los algoritmos de _Machine Learning_ a la hora 
de realizar la predicción del comportamiento futuro del mercado. Para solventar dicho error será necesario seguir los 
pasos explicados a continuación:

![tcue_platform_talib](https://github.com/alvarob96/tcue/blob/master/resources/tcue_platform_talib.gif)