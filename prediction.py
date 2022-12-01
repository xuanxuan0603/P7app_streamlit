import requests

def get_proba_for_client(client_id:str):
    #url = f'https://debugapi-p7.herokuapp.com/predict?id_client={client_id}'
    url = 'http://127.0.0.1:8000/predict?SK_ID_CURR={}'.format(client_id)
    response = requests.get(url).json()
    return response['prediction'][1]

def force_plot(client_id:str):
    url = 'http://127.0.0.1:8000/shapplot?SK_ID_CURR={}'.format(client_id)
    response = requests.get(url).json()
    return response['force_plot_html']