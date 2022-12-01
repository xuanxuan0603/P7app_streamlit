import requests

def get_proba_for_client(client_id:str):
    #url = f'https://debugapi-p7.herokuapp.com/predict?id_client={client_id}'
    url = 'https://fastapi-p7-zhao.herokuapp.com/predict?SK_ID_CURR={}'.format(client_id)
    response = requests.get(url).json()
    return response['prediction'][1]

def force_plot(client_id:str):
    url = 'https://fastapi-p7-zhao.herokuapp.com/shapplot?SK_ID_CURR={}'.format(client_id)
    response = requests.get(url).json()
    return response['force_plot_html']