import streamlit as st
import joblib
from prediction import get_proba_for_client, get_plot
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
shap.initjs()
import streamlit.components.v1 as components
import plotly.graph_objects as go

st.set_page_config(layout="wide")

image = Image.open('assets/feature_importance_plot.png')
image_home_page = Image.open('assets/image_homepage.png')

classifier = joblib.load('data/classifier.joblib')
data = joblib.load('data/data.joblib')
threshold = joblib.load('data/threshold.joblib')
orignal_data = joblib.load('data/app_train_api.joblib')


select_box_options = ['Home page', 'Customer Information', 'Customer Credit Risk', 'Customer analysis']

selectbox = st.sidebar.selectbox("Chose one of the options below", select_box_options, index=0)


if selectbox == select_box_options[0]:
    st.title('Welcome to **Home Credit Risk** :sunglasses:')
    st.image(image_home_page)

    #original_title = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Hello everyone, welcome to APP Home Credit Default Risk!:sunglasses:</p>'
    st.write(' ')
    
    tab1, tab2 = st.tabs(["General informations", "Feature important"])
    with tab1:
        st.header("Hello everybody :panda_face: ")
        st.markdown('''In this application, you will see all personal information and loans probabilitis about our credit customers.
                    This application use Machine Learning model to predict the probability of credit risk. If you want to understand which feature is the most important for this model, please see more information in the right tab.
                    In addition, this application also provide somes figures allowing to understand better customer statistic informations.
                    ''')
        st.info('To start, please choose a customer that you want using form in the page customer information. After get the ID, you can analyse the risk and see more details', icon="ℹ️")
        # nombre de client
        # plot: nombre de homme et femme
        # 
    with tab2:
        st.header("Feature important")
        st.image(image, caption='Feature important plot')
        expander = st.expander("See explanation")
        expander.write('''The SHAP value plot can show the positive and negative relationships of the predictors with the target variable.
                          This plot is made of all the dots in the data. It delivers the following information: 
                          1. Feature importance: Variables are ranked in descending order.
                          2. Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
                          3. Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.
                          4. Correlation: A high level of the “alcohol” content has a high and positive impact on the quality rating. The “high” comes from the red color, and the “positive” impact is shown on the X-axis. 
                        ''')


elif selectbox == select_box_options[1]: 

    with st.form(key='client_info_form'):
        c1, c2, c3 = st.columns([3, 3, 3])
        with c1:
            sex = st.selectbox('Gender', ('F', 'M'))
        with c2:
            work = st.selectbox('Job type', ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Unemployed', 'Student'])
        with c3:
            family_status = st.selectbox('Family status', ('Married', 'Single / not married', 'Separated', 'Widow','Civil marriage', 'Unknown'))
        
        c1, c2, c3 = st.columns([3, 3, 3])
        with c1:
            child = st.slider('Number of childrens', 0, 14,(0, 14))
        with c2:
            age = st.slider('Age', 0, 100, (0, 100))
        with c3:
            income = st.slider('Total income', 26000, 3825000, (26000, 3825000))
        
        c1, c2, c3 = st.columns([3, 3, 3])
        with c1:
            amt_credit = st.slider('Credit amount of the loan', 45000, 4050000, (45000, 4050000))
        with c2:
            amt_annuity = st.slider('Loan annuity', 2052, 225000,(2052, 225000))
        with c3:
            goods_price = st.slider('The price of the goods', 45000, 4050000, (45000, 4050000))

        submit_button = st.form_submit_button(label='Submit')

        if not submit_button:
            st.stop()

    with st.container():
        filtered_df = orignal_data.loc[(orignal_data['CODE_GENDER'] == sex) & 
                                       (orignal_data['NAME_INCOME_TYPE'] == work) &
                                       (orignal_data['NAME_FAMILY_STATUS'] == family_status) &
                                       (orignal_data['AGE'] <= age[1]) &
                                       (orignal_data['AGE'] >= age[0])&
                                       (orignal_data['CNT_CHILDREN'] <= child[1]) &
                                       (orignal_data['CNT_CHILDREN'] >= child[0])&
                                       (orignal_data['AMT_INCOME_TOTAL'] <= income[1]) &
                                       (orignal_data['AMT_INCOME_TOTAL'] >= income[0])&
                                       (orignal_data['AMT_CREDIT'] <= amt_credit[1]) &
                                       (orignal_data['AMT_CREDIT'] >= amt_credit[0])&
                                       (orignal_data['AMT_ANNUITY'] <= amt_annuity[1]) &
                                       (orignal_data['AMT_ANNUITY'] >= amt_annuity[0])&
                                       (orignal_data['AMT_GOODS_PRICE'] <= goods_price[1])&
                                       (orignal_data['AMT_GOODS_PRICE'] >= goods_price[0]),:]
        
        st.header('Rearch results : {} customers found'.format(len(filtered_df)))
        st.table(filtered_df['SK_ID_CURR'])

elif selectbox == select_box_options[2]:
    client_id = st.sidebar.text_input('SK_ID_CURR')
            
    tab1, tab2, tab3  = st.tabs(["Personal information", " Credit risk", "Local Interpretability"])
    with tab1:
        st.header("")

    with tab1:
        if client_id != "":
            info_client = orignal_data.loc[orignal_data['SK_ID_CURR'] == int(client_id),:]
            info_client_dt  = info_client[['CODE_GENDER','AGE','NAME_FAMILY_STATUS','CNT_CHILDREN','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']]
            info_client_dt = info_client_dt.T
            info_client_dt.columns = ['Customer information']
            st.write(info_client_dt)

    with tab2:
        if client_id != "":
            risk = get_proba_for_client(client_id=str(client_id))
            risk_decimals = round(risk, 2)
            st.title("Credit risk is {} %".format(risk_decimals))   
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_decimals/100,
                title = {'text': "Credit Risk Evaluation", 'font': {'size': 25}},
                delta = {'reference': 0.522, 'increasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [None, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "darkblue",
                    'steps': [
                        {'range': [0, threshold], 'color': 'white'},
                        {'range': [threshold, 1], 'color': 'red'}]}))
            st.write(fig)
            st.warning('Warning! If customer risk is higer than risk threshold, this loan should be refused', icon="⚠️")
    
    with tab3:
        if client_id != "":
            client_id
            st.text("This interpretability plot is about understanding a single customer of a model")
            plot_html = get_plot(client_id)
            components.html(html=plot_html['force_plot_html'])
            st.info('The explain ability for any individual customer is the most critical step to convince you to adopt your model. Note that red means this feature push the prediction higher (to the right) and those pushing the prediction lower are in blue.', icon="ℹ️")


else: 
    st.subheader('In this page, we are going to compare a customer with others customers in the groupe. Please input customer ID')
    client_id = st.text_input('SK_ID_CURR')

    if client_id != "":
        customer_row = orignal_data.loc[orignal_data['SK_ID_CURR'] == int(client_id),:]
        info_client_dt  = customer_row[['CODE_GENDER','AGE','NAME_FAMILY_STATUS','CNT_CHILDREN','NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']]
        st.write(info_client_dt)

        customer_age = customer_row['AGE'].values
        customer_income  = customer_row['AMT_INCOME_TOTAL'].values
        customer_anuu  = customer_row['AMT_ANNUITY'].values
        customer_price = customer_row['AMT_GOODS_PRICE'].values
        customer_credit = customer_row['AMT_CREDIT'].values


        tab1, tab2, tab3, tab4, tab5, tab6= st.tabs(["Gender", "Age", 'Family situation','Total income','Credit','Credit annuity'])
        with tab1:
            fig = plt.figure(figsize=(6, 4))
            sns.histplot(data=orignal_data, x="CODE_GENDER", hue="TARGET", multiple="dodge",shrink=.8)
            st.pyplot(fig)
        with tab2:
            fig = plt.figure(figsize=(6, 4))
            plt.axvline(x=customer_age, ymin=0, ymax=1, color='r')
            sns.kdeplot(data=orignal_data, x="AGE", hue="TARGET")
            st.pyplot(fig)

        with tab3:
            fig = plt.figure(figsize=(10, 4))
            plt.xticks(rotation='vertical')
            sns.barplot(data=orignal_data, x= "NAME_FAMILY_STATUS", y = 'CNT_CHILDREN', hue="TARGET")
            st.pyplot(fig)
            # tableau de ce clients

        with tab4:
            fig = plt.figure(figsize=(6, 4))
            plt.axvline(x=customer_income, ymin=0, ymax=1, color='r')
            sns.kdeplot(data=orignal_data, x="AMT_INCOME_TOTAL", hue="TARGET", cumulative=True, common_norm=False, common_grid=True)
            st.pyplot(fig)

        with tab5:
            fig = plt.figure(figsize=(10, 10))
            plt.axvline(x=customer_credit , ymin = 0, ymax = customer_price, color='r')
            plt.axhline(y=customer_price, xmin = 0, xmax = customer_credit, color='r')

            sns.scatterplot(data=orignal_data, x= "AMT_CREDIT", y = 'AMT_GOODS_PRICE', hue="TARGET")
            st.pyplot(fig)

        with tab6:
            fig = plt.figure(figsize=(10, 4))
            plt.axvline(x=customer_anuu, ymin=0, ymax=1, color='r')
            sns.kdeplot(data=orignal_data, x= "AMT_ANNUITY", hue="TARGET", multiple="fill")
            st.pyplot(fig)
