# TCUE

<div>
  <img src="https://github.com/alvarob96/tcue/blob/master/resources/tcue_es.png"><br>
</div>

## Descripción General del Proyecto

El proyecto consiste en la creación y desarrollo de una plataforma para la recomendación de inversión bursátil en 
acciones de empresas del mercado continuo español a través del uso de técnicas de análisis técnico financiero, 
algoritmos de **Inteligencia Artificial** (IA) y técnicas de **Procesamiento del Lenguaje Natural** (NLP) de Twitter. 
De este modo, se busca determinar tanto la correlación de las técnicas de análisis de mercado tradicionales con 
el uso de la IA como el impacto de la opinión volcada sobre las redes sociales, en este caso en Twitter, y su 
impacto o no sobre el comportamiento futuro del mercado. De este modo, el proyecto a partir del nombre de una acción 
del mercado continuo español, recuperará los datos en tiempo real del mismo y los mostrará, junto con el análisis 
realizado del mismo, la aplicación de los distintos algoritmos de Machine Learning (ML), las tendencias identificadas 
a lo largo de los datos históricos recuperado, y, en función del tipo de tendencia, el análisis de sentimiento de 
Twitter de cada una de dichas tendencias para estudiar la correlación entre dicho sentimiento y la tendencia que 
siguió el mercado.

## Objetivos

Los principales objetivos del proyecto se refieren a **determinar la correlación y/o posible combinación del uso de 
técnicas tradicionales de estudio y predicción del mercado bursátil junto con técnicas más modernas de IA que 
abarcan desde el uso de algoritmos de regresión para la predicción, algoritmos para la identificación de tendencias 
en el mercado y técnicas y modelos de NLP de la opinión extraída de Twitter**. 

## Resultados Obtenidos (Aplicabilidad)

La plataforma desarrollada se ha hecho modular, de forma que se han creado una serie de módulos, paquetes de Python en 
este caso, [investpy](https://github.com/alvarob96/investpy) para la extracción de los datos históricos de acciones, 
fondos y ETFs de Investing.com, [twipper](https://github.com/alvarob96/twipper) para el uso de la API de Twitter con 
Python y [trendet](https://github.com/alvarob96/trendet) para la identificación de tendencias de mercado en series 
temporales. Además, también se han realizado los estudios tanto de mercado a nivel técnico financiero como la 
aplicación de algoritmos de IA  y el uso de NLP dando lugar así a la creación de un sistema de asignación de los 
hiper-parámetros más óptimos para cada una de las acciones del mercado continuo español dando así lugar a la 
predicción más precisa posible; y la creación de un modelo de clasificación de sentimiento no supervisado en 
español junto con la creación de un modelo de lenguaje natural en español.

## Estructura

A continuación se desglosará la estructura del repositorio:

* ``docs/``: en este directorio se encuentra tanto la memoria como la presentación del proyecto en pptx. Adicionalmente,
se pueden encontrar en este directorio los PDF generados a partir de los Jupyter Notebooks, es decir, los informes
detallados sobre cada uno de los paquetes y de las funcionalidades implementadas en el proyecto. Estos informes recogen la
funcionalidad principal de [investpy](https://github.com/alvarob96/investpy), [trendet](https://github.com/alvarob96/trendet) 
y [twipper](https://github.com/alvarob96/twipper); y, en consecuencia, la integración de los mismos para con la 
plataforma a modo de estudio.

* ``notebooks/``: en este directorio se recogen los Jupyter Notebooks creados tanto para justificar el uso y creación de
los distintos paquetes de Python, como informes sobre su uso e integración entre ellos.

* ``resources/``: en este directorio se encuentran los distintos recursos que se utilizan a lo largo de la justificación
y documentación del proyecto, en su mayoría, los GIFs para demostrar tanto el uso de la plataforma, como su configuración
e instalación desde cero.

* ``tcue_platform/``: este proyecto es la plataforma Django creada para el TCUE con el fin de visualizar tanto los datos 
extraidos con [investpy](https://github.com/alvarob96/investpy) como el análisis y la predicción realizada sobre ellos.
Por tanto, esta es la plataforma central del TCUE que integra el resto de módulos creados.

* ``tcue_sentiment/``: este proyecto es la plataforma en Django creada para clasificar los tweets ingeridos en tiempo real 
a través de [twipper](https://github.com/alvarob96/twipper) de una forma más mecánica en sentimientos 
(**positivo, neutral y negativo**), con el fin de utilizarlos más adelante de cara a crear un modelo de clasificación 
para el análisis de sentimientos.

## Autor

**Álvaro Bartolomé del Canto**