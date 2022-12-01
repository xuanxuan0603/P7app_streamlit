import requests
import joblib
import shap
shap.initjs()

classifier = joblib.load('data/classifier.joblib')
data = joblib.load('data/data.joblib')


def get_proba_for_client(client_id:str):
    url = 'https://fastapi-p7-zhao.herokuapp.com/predict?SK_ID_CURR={}'.format(client_id)
    response = requests.get(url).json()
    return response['prediction'][1]


def get_plot(SK_ID_CURR):
    # SK_ID_CURR to X
    dt = data.loc[data['SK_ID_CURR'] == int(SK_ID_CURR),:]
    X = dt.drop(columns = ['TARGET','SK_ID_CURR'])
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], X.values, feature_names = X.columns, matplotlib=False)
    
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    return {'force_plot_html': shap_html}