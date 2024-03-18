
import streamlit as st 
from streamlit_option_menu import option_menu
import pickle
import pandas as pd



with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Predection Page", "Dataset", "Models & Accuracy"]
    )


if selected == 'Predection Page':




    workclass = [' State-gov', ' Self-emp-not-inc', 
                ' Private', ' Federal-gov',
                ' Local-gov', ' Self-emp-inc', ' Without-pay']

    education = [' Bachelors', ' HS-grad', 'Lower', 
                ' Masters', ' Some-college',
                ' Assoc-acdm', ' Doctorate', 
                ' Assoc-voc', ' Prof-school']

    marital_status = ['not-married', 'married']

    occupation = [' Adm-clerical', ' Exec-managerial',
                ' Handlers-cleaners',' Prof-specialty',
                ' Other-service', ' Sales', ' Transport-moving',
                ' Farming-fishing', ' Machine-op-inspct', 
                ' Tech-support',' Craft-repair', ' Protective-serv',
                ' Armed-Forces',' Priv-house-serv']

    sex = [' Male', ' Female']

    hours_per_week = ['between 31,40', 'between 10,20', 'between 41,50',
                    'between 71,80','between 21,30', 'between 51,60',
                    'between 61,70', '<10','between 91,100', 'between 81,90']





    random_f = pickle.load(open('random_f.pkl','rb'))



    st.title("Income Prediction Classifier")




    col1, col2 = st.columns(2)

    with col1:
        select_workclass = st.selectbox('Select Workclass',sorted(workclass))
        
    with col2:
        select_education = st.selectbox('Select Education',sorted(education))
        
        
        
        
    select_occupation = st.selectbox('Select Occupation',sorted(occupation))
        
        
        
        
    col5, col6 = st.columns(2)
        
    with col5:
        select_sex = st.selectbox('Select Sex',sorted(sex))
        
    with col6:
        select_hours_per_week = st.selectbox('Select Hours Per Week',sorted(hours_per_week))
        

    col7, col8 = st.columns(2)

    with col7:
        select_age = st.slider('',min_value = 20,max_value = 100,step = 1,value = 40)
        
    with col8:
        select_marital_status = st.selectbox('Select Marital Status',sorted(marital_status))
        
        
        
        
        
    if st.button('Predict'):
        input_df = pd.DataFrame({'age':[select_age],'workclass':[select_workclass],'education':[select_education],
                                'marital_status':[select_marital_status],'occupation':[select_occupation],
                                'sex':[select_sex],'hours_per_week':[select_hours_per_week]})

        #st.table(input_df)
        result = random_f.predict(input_df)
        result_percentage = random_f.predict_proba(input_df)
        
        st.title('Predicted Outcome...')
        
        if result[0] == 'Income less than 50K':
            st.info('Income less than 50K')
        else:
            st.info('Income more than 50K')
        
        
        
        st.title("To be more precise....") 
        
        f = result_percentage[0][0]
        s = result_percentage[0][1]
        
        st.success("Chances of Income more than 50K : "+str(round(f*100,2))+"%")
        st.error("Chances of Income less than 50K : "+str(round(s*100,2))+"%")
    





if selected == 'Dataset':
    
    data = pd.read_csv('adult.csv')
    show = data.head()
    show2 = data.tail()
    show3 = data.columns
    show4 = data.describe().T
    
    
    
    st.title('First 5 Values of the Dataset...')
    
    st.table(show)
    
    
    
    st.title('Last 5 Values of the Dataset...')
    
    st.table(show2)
    
    
    
    
       
    st.title('Columns of the Dataset...')
    
    st.table(show3)
        
        
           
    st.title('Description of the Dataset...')
    
    st.table(show4)
 
 
 
 
 
 
 
if selected == 'Models & Accuracy':
    
    st.title('Logistic Regression')
    
    st.info('Gives us 78.21%, accuracy')
    
    
    
    st.title('KNeighborsClassifier')
    
    st.info('Gives us 79.91%, accuracy')
    
    
    
    st.title('Naive Bayes')
    
    st.info('Gives us 76.41%, accuracy')
    
    
    
    st.title('Decision Tree Classifier')
    
    st.info('Gives us 77.19%, accuracy')
    
    
    
    st.title('Random Forest Classifier')
    
    st.info('Gives us 81.68%, accuracy')
    
    
    
    
    
    
    
    
    
    
    
