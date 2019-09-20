import datetime
import json

import investpy
import pandas as pd
import pkg_resources
import unidecode
import talib as ta

from django.shortcuts import render, render_to_response

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


def index(request):
    with open('log.txt', 'a') as f:
        f.write(str(get_client_ip(request)) + ' ' + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + '\n')
    f.close()

    return render_to_response('index.html')


def overview(request):
    mesg = request.GET['equity'].strip()

    today = str(datetime.datetime.today().strftime('%d/%m/%Y'))
    five_years_ago = str((datetime.datetime.today() - datetime.timedelta(days=5 * 365)).strftime('%d/%m/%Y'))

    try:
        json_ = investpy.get_historical_data(equity=mesg,
                                             country='spain',
                                             from_date=five_years_ago,
                                             to_date=today,
                                             as_json=True)
    except:
        return render_to_response('error.html')

    parsed_json = json.loads(json_)

    try:
        obj = investpy.get_equity_company_profile(equity=mesg,
                                                  country='spain',
                                                  language='spanish')
    except:
        return render_to_response('error.html')

    try:
        equities = investpy.get_equities(country='spain')

        equity = unidecode.unidecode(mesg.lower())
        equity = equity.strip()

        name = equities.loc[(equities['name'].str.lower() == equity).idxmax(), 'name']
        full_name = equities.loc[(equities['name'].str.lower() == equity).idxmax(), 'full_name']
    except:
        return render_to_response('error.html')

    context = {'full_name': full_name,
               'name': name,
               'data_json': json_,
               'decoded_data': parsed_json,
               'desc': obj['desc'],
               'source': obj['url']}

    return render(request, 'overview.html', context=context)


def recommendation(request):
    mesg = request.GET['equity'].strip()

    today = str(datetime.datetime.today().strftime('%d/%m/%Y'))
    five_years_ago = str((datetime.datetime.today() - datetime.timedelta(days=5 * 365)).strftime('%d/%m/%Y'))

    try:
        json_ = investpy.get_historical_data(equity=mesg,
                                             country='spain',
                                             from_date=five_years_ago,
                                             to_date=today,
                                             as_json=True)
    except:
        return render_to_response('error.html')

    parsed_json = json.loads(json_)

    try:
        obj = investpy.get_equity_company_profile(equity=mesg,
                                                  country='spain',
                                                  language='spanish')
    except:
        return render_to_response('error.html')

    try:
        equities = investpy.get_equities(country='spain')

        equity = unidecode.unidecode(mesg.lower())
        equity = equity.strip()

        name = equities.loc[(equities['name'].str.lower() == equity).idxmax(), 'name']
        full_name = equities.loc[(equities['name'].str.lower() == equity).idxmax(), 'full_name']
    except:
        return render_to_response('error.html')

    try:
        df_ = investpy.get_historical_data(equity=mesg,
                                           country='spain',
                                           from_date=five_years_ago,
                                           to_date=today,
                                           as_json=False)
    except:
        return render_to_response('error.html')

    X = [[value] for value in df_['Open'].values]
    y = df_['Close'].values.tolist()

    algorithms = ['MLPRegressor', 'SVM-LinearSVR', 'KNeighborsRegressor',
                  'GradientBoostingRegressor', 'RandomForestRegressor']
    algorithms_ = json.dumps(algorithms)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=False)

    values_ = dict()
    predictions_ = dict()

    resource_package = __name__
    resource_path = '/'.join(('static', 'resources', 'results.json'))

    with open(pkg_resources.resource_filename(resource_package, resource_path), 'r') as f:
        grid_search_results = json.load(f)

    if mesg.lower() in grid_search_results.keys():
        mlp_params = grid_search_results[mesg.lower()]['MLPRegressor']['best_params']

        mlp = MLPRegressor(activation=mlp_params['activation'],
                           hidden_layer_sizes=mlp_params['hidden_layer_sizes'],
                           learning_rate=mlp_params['learning_rate'],
                           shuffle=mlp_params['shuffle'],
                           solver=mlp_params['solver'])

        mlp.fit(X_train, y_train)

        mlp_prediction = mlp.predict(X_test)

        json_prediction = json.dumps(mlp_prediction.tolist())

        values_.update({'MLPRegressor': mlp.score(X_test, y_test)})
        predictions_.update({'MLPRegressor': mlp_prediction.tolist()})

        linear_params = grid_search_results[mesg.lower()]['SVM-LinearSVR']['best_params']

        linear = LinearSVR(loss=linear_params['loss'],
                           max_iter=linear_params['max_iter'])

        linear.fit(X_train, y_train)

        linear_prediction = linear.predict(X_test)

        values_.update({'SVM-LinearSVR': linear.score(X_test, y_test)})
        predictions_.update({'SVM-LinearSVR': linear_prediction.tolist()})

        neighbor_params = grid_search_results[mesg.lower()]['KNeighborsRegressor']['best_params']

        neighbor = KNeighborsRegressor(algorithm=neighbor_params['algorithm'],
                                       n_neighbors=neighbor_params['n_neighbors'],
                                       weights=neighbor_params['weights'])

        neighbor.fit(X_train, y_train)

        neighbor_prediction = neighbor.predict(X_test)

        values_.update({'KNeighborsRegressor': neighbor.score(X_test, y_test)})
        predictions_.update({'KNeighborsRegressor': neighbor_prediction.tolist()})

        gradient_params = grid_search_results[mesg.lower()]['GradientBoostingRegressor']['best_params']

        gradient = GradientBoostingRegressor(criterion=gradient_params['criterion'],
                                             loss=gradient_params['loss'],
                                             n_estimators=gradient_params['n_estimators'])

        gradient.fit(X_train, y_train)

        gradient_prediction = gradient.predict(X_test)

        values_.update({'GradientBoostingRegressor': gradient.score(X_test, y_test)})
        predictions_.update({'GradientBoostingRegressor': gradient_prediction.tolist()})

        regressor_params = grid_search_results[mesg.lower()]['RandomForestRegressor']['best_params']

        regressor = RandomForestRegressor(bootstrap=regressor_params['bootstrap'],
                                          criterion=regressor_params['criterion'],
                                          n_estimators=regressor_params['n_estimators'])

        regressor.fit(X_train, y_train)

        regressor_prediction = regressor.predict(X_test)

        values_.update({'RandomForestRegressor': regressor.score(X_test, y_test)})
        predictions_.update({'RandomForestRegressor': regressor_prediction.tolist()})

        best_params_start = "<center><button type=\"button\" class=\"btn btn-primary\" data-toggle=\"modal\" " \
                            "data-target=\".bd-example-modal-lg\">Mapas de Calor de Selección de Híper-Parámetros</button>" \
                            "</center><div class=\"modal fade bd-example-modal-lg\" tabindex=\"-1\" role=\"dialog\" " \
                            "aria-labelledby=\"myLargeModalLabel\" aria-hidden=\"true\"><div " \
                            "class=\"modal-dialog modal-lg\"><div class=\"modal-content\"><div class=\"modal-header\">"\
                            "<h5 class=\"modal-title\">Mapas de Calor de Selección de Híper-Parámetros</h5>" \
                             "<button type=\"button\" class=\"close\" data-dismiss=\"modal\" aria-label=\"Close\">" \
                            "<span aria-hidden=\"true\">&times;</span></button></div><div class=\"modal-body\">"

        regression_algorithms = {
            'MLPRegressor': {
                'desc': 'Un perceptrón multicapa (MLP) es una clase de red neuronal artificial de retroalimentación. '
                        'Una MLP consta de al menos tres capas de nodos: una capa de entrada, una capa oculta y una '
                        'capa de salida. A excepción de los nodos de entrada, cada nodo es una neurona que utiliza una '
                        'función de activación no lineal. MLP utiliza una técnica de aprendizaje supervisado llamada '
                        'backpropagation para la formación. Sus múltiples capas y su activación no lineal distinguen '
                        'al MLP de un perceptrón lineal. Puede distinguir datos que no son linealmente separables.',
                'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html'
            },
            'LinearSVR': {
                'desc': 'En el aprendizaje automático, las máquinas vectoriales de apoyo (SVM, también llamadas redes '
                        'vectoriales de apoyo) son modelos de aprendizaje supervisados con algoritmos de aprendizaje '
                        'asociados que analizan los datos utilizados para la clasificación y el análisis de regresión. '
                        'Dado un conjunto de ejemplos de entrenamiento, cada uno marcado como perteneciente a una '
                        'u otra de las dos categorías, un algoritmo de entrenamiento SVM construye un modelo que asigna '
                        'nuevos ejemplos a una u otra categoría, convirtiéndolo en un clasificador lineal binario no '
                        'probabilístico tcue_classifier (aunque existen métodos como el escalado Platt para usar SVM '
                        'en un entorno de clasificación probabilístico). En SVR intentamos ajustar el error dentro de '
                        'un cierto umbral. Similar a SVR con el parámetro kernel=linear en este caso.',
                'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html'
            },
            'KNeighborsRegressor': {
                'desc': 'En el reconocimiento de patrones, el algoritmo k-nearest neighbors (k-NN) es un método no '
                        'paramétrico utilizado para la clasificación y regresión. En ambos casos, la entrada consiste '
                        'en los ejemplos de entrenamiento más cercanos a k en el espacio de características. La salida '
                        'depende de si k-NN se utiliza para la clasificación o la regresión, en la regresión k-NN, la '
                        'salida es el valor de la propiedad del objeto. Este valor es el promedio de los valores de k '
                        'de los vecinos más cercanos. El objetivo se predice mediante la interpolación local de los '
                        'objetivos asociados a los vecinos más cercanos en el conjunto de entrenamiento.',
                'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html'
            },
            'GradientBoostingRegressor': {
                'desc': 'El aumento de gradiente es una técnica de aprendizaje automático para problemas de regresión y'
                        ' clasificación, que produce un modelo de predicción en forma de un conjunto de modelos de '
                        'predicción débiles, normalmente árboles de decisión. Construye el modelo de forma escénica, '
                        'como lo hacen otros métodos de refuerzo, y los generaliza permitiendo la optimización de una '
                        'función de pérdida arbitraria y diferenciable. El aumento de gradiente construye un modelo '
                        'aditivo en una etapa avanzada; permite la optimización de funciones de pérdida diferenciables '
                        'arbitrarias. En cada etapa se coloca un árbol de regresión en el gradiente negativo de la '
                        'función de pérdida dada.',
                'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html'
            },
            'RandomForestRegressor': {
                'desc': 'Los bosques aleatorios o bosques de decisión aleatoria son un método de aprendizaje conjunto '
                        'para la clasificación, regresión y otras tareas que opera construyendo una multitud de árboles '
                        'de decisión en el momento del entrenamiento y produciendo la clase que es el modo de las '
                        'clases (clasificación) o la predicción media (regresión) de los árboles individuales. Un '
                        'bosque aleatorio es un metaestimulador que se ajusta a una serie de árboles de decisión de '
                        'clasificación en varias submuestras del conjunto de datos y utiliza el promediado para '
                        'mejorar la precisión predictiva y el ajuste excesivo de control.',
                'url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'
            },
        }

        for algorithm in regression_algorithms:
            best_params_start += "<a href=\"" + regression_algorithms[algorithm]['url'] + \
                                 "\" target=\"_blank\"><h4 style=\"margin-left: 40px;\">" + str(algorithm) + \
                                 "</h4></a><p style=\"margin-left: 40px;\">" \
                                 + str(regression_algorithms[algorithm]['desc']) + \
                                 "</p><img width=\"100%\" src=\"../static/resources/equities/" + mesg.lower() \
                                 + "/" + algorithm.lower() + "_heatmap.png\" alt=\"\"/></br>"

        best_params_end = "</div><div class=\"modal-footer\"><button type=\"button\" class=\"btn btn-secondary\" " \
                          "data-dismiss=\"modal\">Cerrar</button></div></div></div></div>"

        best_params_modal_ = best_params_start + best_params_end
    else:
        mlp = MLPRegressor()

        mlp.fit(X_train, y_train)

        mlp_prediction = mlp.predict(X_test)

        json_prediction = json.dumps(mlp_prediction.tolist())

        values_.update({'MLPRegressor': mlp.score(X_test, y_test)})
        predictions_.update({'MLPRegressor': mlp_prediction.tolist()})

        linear = LinearSVR()

        linear.fit(X_train, y_train)

        linear_prediction = linear.predict(X_test)

        values_.update({'SVM-LinearSVR': linear.score(X_test, y_test)})
        predictions_.update({'SVM-LinearSVR': linear_prediction.tolist()})

        neighbor = KNeighborsRegressor()

        neighbor.fit(X_train, y_train)

        neighbor_prediction = neighbor.predict(X_test)

        values_.update({'KNeighborsRegressor': neighbor.score(X_test, y_test)})
        predictions_.update({'KNeighborsRegressor': neighbor_prediction.tolist()})

        gradient = GradientBoostingRegressor()

        gradient.fit(X_train, y_train)

        gradient_prediction = gradient.predict(X_test)

        values_.update({'GradientBoostingRegressor': gradient.score(X_test, y_test)})
        predictions_.update({'GradientBoostingRegressor': gradient_prediction.tolist()})

        regressor = RandomForestRegressor()

        regressor.fit(X_train, y_train)

        regressor_prediction = regressor.predict(X_test)

        values_.update({'RandomForestRegressor': regressor.score(X_test, y_test)})
        predictions_.update({'RandomForestRegressor': regressor_prediction.tolist()})

        best_params_modal_ = "<p class=\"text-center\">Dado que los datos históricos no son suficientes, se han" \
                             "utilizando los <b>híper-parámetros por defecto</b> de los algoritmos de regresión " \
                             "utilizandos</p>"

    best = max(values_, key=lambda key: values_[key])
    best_ = json.dumps(best)

    scores = [mlp.score(X_test, y_test), linear.score(X_test, y_test), neighbor.score(X_test, y_test),
              gradient.score(X_test, y_test), regressor.score(X_test, y_test)]
    scores_ = json.dumps(scores)

    actual_value = y_test[-1]
    predicted_value = mlp_prediction[-1]
    open_value = X_test[-1][0]

    if predicted_value > open_value:
        trend = 'ALCISTA'
        if predicted_value > actual_value:
            signal = 'COMPRAR'
        elif predicted_value == actual_value:
            signal = 'MANTENER'
        else:
            signal = 'VENDER'
    else:
        trend = 'BAJISTA'
        if predicted_value > actual_value:
            signal = 'COMPRAR'
        elif predicted_value == actual_value:
            signal = 'MANTENER'
        else:
            signal = 'VENDER'

    five_ma = df_.rolling(window=5).mean()['Close'].tolist()[4:]
    ten_ma = df_.rolling(window=10).mean()['Close'].tolist()[9:]
    twenty_ma = df_.rolling(window=20).mean()['Close'].tolist()[19:]
    fifty_ma = df_.rolling(window=50).mean()['Close'].tolist()[49:]
    one_hundred_ma = df_.rolling(window=100).mean()['Close'].tolist()[99:]
    two_hundred_ma = df_.rolling(window=200).mean()['Close'].tolist()[199:]

    moving_averages_ = {
        '5 days': {
            'result': five_ma,
            'last_value': round(five_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if five_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '10 days': {
            'result': ten_ma,
            'last_value': round(ten_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if ten_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '20 days': {
            'result': twenty_ma,
            'last_value': round(twenty_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if twenty_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '50 days': {
            'result': fifty_ma,
            'last_value': round(fifty_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if fifty_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '100 days': {
            'result': one_hundred_ma,
            'last_value': round(one_hundred_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if one_hundred_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '200 days': {
            'result': two_hundred_ma,
            'last_value': round(two_hundred_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if two_hundred_ma[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        }
    }

    five_ema = df_.ewm(span=5, adjust=False).mean()['Close'].tolist()[4:]
    ten_ema = df_.ewm(span=10, adjust=False).mean()['Close'].tolist()[9:]
    twenty_ema = df_.ewm(span=20, adjust=False).mean()['Close'].tolist()[19:]
    fifty_ema = df_.ewm(span=50, adjust=False).mean()['Close'].tolist()[49:]
    one_hundred_ema = df_.ewm(span=100, adjust=False).mean()['Close'].tolist()[99:]
    two_hundred_ema = df_.ewm(span=200, adjust=False).mean()['Close'].tolist()[199:]

    exponential_moving_averages_ = {
        '5 days': {
            'result': five_ema,
            'last_value': round(five_ma[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if five_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '10 days': {
            'result': ten_ema,
            'last_value': round(ten_ema[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if ten_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '20 days': {
            'result': twenty_ema,
            'last_value': round(twenty_ema[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if twenty_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '50 days': {
            'result': fifty_ema,
            'last_value': round(fifty_ema[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if fifty_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '100 days': {
            'result': one_hundred_ema,
            'last_value': round(one_hundred_ema[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if one_hundred_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        },
        '200 days': {
            'result': two_hundred_ema,
            'last_value': round(two_hundred_ema[-1], 5),
            'signal': '<font color="green">COMPRAR</font>' if two_hundred_ema[-1] < df_['Close'][-1] else '<font color="red">VENDER</font>'
        }
    }

    labels = ['5 days', '10 days', '20 days', '50 days', '100 days', '200 days']
    aux = pd.DataFrame(columns=['SMA', 'EMA', 'SMA Signal', 'EMA Signal'], index=labels)

    aux['SMA']['5 days'] = round(five_ma[-1], 5)
    aux['SMA']['10 days'] = round(ten_ma[-1], 5)
    aux['SMA']['20 days'] = round(twenty_ma[-1], 5)
    aux['SMA']['50 days'] = round(fifty_ma[-1], 5)
    aux['SMA']['100 days'] = round(one_hundred_ma[-1], 5)
    aux['SMA']['200 days'] = round(two_hundred_ma[-1], 5)

    aux['EMA']['5 days'] = round(five_ema[-1], 5)
    aux['EMA']['10 days'] = round(ten_ema[-1], 5)
    aux['EMA']['20 days'] = round(twenty_ema[-1], 5)
    aux['EMA']['50 days'] = round(fifty_ema[-1], 5)
    aux['EMA']['100 days'] = round(one_hundred_ema[-1], 5)
    aux['EMA']['200 days'] = round(two_hundred_ema[-1], 5)

    last_value = df_['Close'][-1]

    for index, row in aux.iterrows():
        if row['SMA'] < last_value:
            row['SMA Signal'] = 'COMPRAR'
        elif row['SMA'] > last_value:
            row['SMA Signal'] = 'VENDER'

        if row['EMA'] < last_value:
            row['EMA Signal'] = 'COMPRAR'
        elif row['EMA'] > last_value:
            row['EMA Signal'] = 'VENDER'

    sma_data = aux.groupby('SMA Signal').size()

    if 'COMPRAR' in sma_data:
        ma_buy_signals_ = sma_data['COMPRAR']
    else:
        ma_buy_signals_ = 0

    if 'VENDER' in sma_data:
        ma_sell_signals_ = sma_data['VENDER']
    else:
        ma_sell_signals_ = 0

    ma_conclusion_ = 'COMPRA FUERTE' if ma_buy_signals_ > 4 else 'COMPRAR' if ma_buy_signals_ > ma_sell_signals_ else 'MANTENER' if ma_buy_signals_ == ma_sell_signals_ else 'VENTA FUERTE' if ma_sell_signals_ > 4 else 'VENDER'

    ema_data = aux.groupby('EMA Signal').size()

    if 'COMPRAR' in ema_data:
        ema_buy_signals_ = ema_data['COMPRAR']
    else:
        ema_buy_signals_ = 0

    if 'VENDER' in ema_data:
        ema_sell_signals_ = ema_data['VENDER']
    else:
        ema_sell_signals_ = 0

    ema_conclusion_ = 'COMPRA FUERTE' if ema_buy_signals_ > 4 else 'COMPRAR' if ema_buy_signals_ > ema_sell_signals_ else 'MANTENER' if ema_buy_signals_ == ema_sell_signals_ else 'VENTA FUERTE' if ema_sell_signals_ > 4 else 'VENDER'

    rsi_result = ta.RSI(df_['Close'], timeperiod=14)[-1]
    stoch_result = ta.STOCH(df_['High'], df_['Low'], df_['Close'], fastk_period=9, slowk_period=6)[0][-1]
    ultosc_result = ta.ULTOSC(df_['High'], df_['Low'], df_['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)[-1]
    willr_result = ta.WILLR(df_['High'], df_['Low'], df_['Close'], timeperiod=14)[-1]

    tech_factors_ = {
        'RSI - 14 días': {
            'result': round(rsi_result, 5),
            'signal': '<font color="grey">MANTENER</font>' if 45 < rsi_result < 55 else '<font color="green">COMPRA FUERTE</font>' if rsi_result > 75 else '<font color="green">COMPRAR</font>' if rsi_result > 50 else '<font color="red">VENTA FUERTE</font>' if rsi_result < 25 else '<font color="red">VENDER</font>'
        },
        'STOCH - 9,6 días': {
            'result': round(stoch_result, 5),
            'signal': '<font color="grey">MANTENER</font>' if 45 < stoch_result < 55 else '<font color="green">COMPRA FUERTE</font>' if stoch_result > 75 else '<font color="green">COMPRAR</font>' if stoch_result > 50 else '<font color="red">VENTA FUERTE</font>' if stoch_result < 25 else '<font color="red">VENDER</font>'
        },
        'ULTOSC - 7,14,28 días': {
            'result': round(ultosc_result, 5),
            'signal': '<font color="green">COMPRA FUERTE</font>' if ultosc_result > 75 else '<font color="green">COMPRAR</font>' if ultosc_result > 50 else '<font color="red">VENTA FUERTE</font>' if ultosc_result < 25 else '<font color="red">VENDER</font>'
        },
        'WILLR - 14 días': {
            'result': round(willr_result, 5),
            'signal': '<font color="grey">MANTENER</font>' if -45 > willr_result > -55 else '<font color="green">SOBRECOMPRA</font>' if willr_result > -20 else '<font color="red">SOBREVENTA</font>' if willr_result < -80 else '<font color="green">COMRPAR</font>' if -20 > willr_result > -45 else '<font color="red">VENDER</font>'
        }
    }

    context = {
        'full_name': full_name,
        'name': name,
        'data_json': json_,
        'decoded_data': parsed_json,
        'last_open': X_test[-1][0],
        'predicted_close': round(mlp_prediction[-1], 5),
        'current_value': actual_value,
        'all_predictions': predictions_,
        'prediction': json_prediction,
        'algorithms': algorithms_,
        'scores': scores_,
        'best_algorithm': best_,
        'trend': trend,
        'hold_prediction': signal,
        'moving_averages': moving_averages_,
        'ma_buy_signals': str(ma_buy_signals_),
        'ma_sell_signals': str(ma_sell_signals_),
        'ma_conclusion': ma_conclusion_,
        'exponential_moving_averages': exponential_moving_averages_,
        'ema_buy_signals': str(ema_buy_signals_),
        'ema_sell_signals': str(ema_sell_signals_),
        'ema_conclusion': ema_conclusion_,
        'tech_factors': tech_factors_,
        'desc': obj['desc'],
        'source': obj['url'],
        'best_params_modal': best_params_modal_
    }

    return render(request, 'recommendation.html', context=context)


def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')

    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')

    return ip
