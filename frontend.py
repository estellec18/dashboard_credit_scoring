import streamlit as st
import json
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#https://credit-app-backend-730c22127ecd.herokuapp.com/

def local_comps(data, df_feat):
    list_feat = list(df_feat.index)
    fig, axs = plt.subplots(3, 3, figsize=(15,15))
    #fig.suptitle("Positionnement du demandeur de crédit sur les 9 variables explicatives principales\n(les valeurs pour le demandeur de crédit sont indiquées en orange)\n\n", fontsize=18)
    i = 0
    j = 0
    info=[]
    for feat in list_feat:
        if len(data[feat].value_counts().index)>5:
            data[feat].plot(kind='hist', ax=axs[i,j])
            axs[i,j].set_title(feat)
            axs[i,j].axvline(x=df_feat.loc[feat,"Value"], color='orange', linewidth=2)
            axs[i,j].tick_params(axis='x', labelsize=8)
            info.append(f"Variable : {feat}")
            info.append(f"\tValeur pour le client : {df_feat.loc[feat,'Value']}")
            info.append(f"\tMin : {data[feat].min():.2f} -- Max : {data[feat].max():.2f} -- Moyenne : {data[feat].mean():.2f}\n")
            info.append(f"------")
        else:
            list_val = data[feat].value_counts(normalize=True).tolist()
            x = df_feat.loc[feat,"Value"]
            possible_cat = data[feat].value_counts().index.to_list()
            info.append(f"Variable : {feat}")
            info.append(f"\tValeur pour le client : {x} (comme {data[feat].value_counts(normalize=True)[x]*100}% de la population étudiée)\n")
            info.append(f"------")
            colors = []
            explode = []
            for cat in possible_cat:
                if cat == x:
                    colors.append("orange")
                    explode.append(0.1)
                else:
                    colors.append("tab:blue")
                    explode.append(0)
            idx_client = colors.index("orange")
            collec = axs[i,j].pie(data[feat].value_counts().values, labels=data[feat].value_counts().index, autopct='%1.0f%%', colors=colors, explode=explode, wedgeprops={"edgecolor":'white'})
            for idx in range(len(data[feat].value_counts().index)):
                if idx == idx_client:
                    collec[1][idx_client].set_fontweight("bold")
                    collec[1][idx_client].set_fontsize(13)
                    collec[2][idx_client].set_fontweight("bold")
                    collec[2][idx_client].set_fontsize(13)
                else:
                    if list_val[idx] > 0.04 :
                        collec[1][idx].set_fontsize(10)
                        collec[2][idx].set_fontsize(10)
                        collec[2][idx].set_color("white")
                        collec[2][idx].set_fontweight("bold")
                    else:
                        collec[1][idx].set_fontsize(10)
                        collec[2][idx].remove()
            axs[i,j].set_title(feat)
        j+=1
        if j>2:
            i+=1
            j=0
    fig.tight_layout()
    return(fig, info)

def custom_comps(data, num_client, feat):
    x = data.loc[str(num_client), feat]
    fig, ax = plt.subplots(figsize=(3,3))
    info = []
    if len(data[feat].value_counts().index)>5:
        data[feat].plot(kind='hist')
        ax.set_title(feat)
        ax.axvline(x, color='orange', linewidth=2)
        ax.tick_params(axis='x', labelsize=8)
        info.append(f"\tValeur pour le client : {x}")
        info.append(f"\tMin : {data[feat].min():.2f} -- Max : {data[feat].max():.2f} -- Moyenne : {data[feat].mean():.2f}\n")
    else:
        list_val = data[feat].value_counts(normalize=True).tolist()
        possible_cat = data[feat].value_counts().index.to_list()
        info.append(f"\tValeur pour le client : {x}")
        info.append(f"(comme {data[feat].value_counts(normalize=True)[x]*100}% de la population étudiée)\n")
        colors = []
        explode = []
        for cat in possible_cat:
            if cat == x:
                colors.append("orange")
                explode.append(0.1)
            else:
                colors.append("tab:blue")
                explode.append(0)
        idx_client = colors.index("orange")
        collec = plt.pie(data[feat].value_counts().values, labels=data[feat].value_counts().index, autopct='%1.0f%%', colors=colors, explode=explode, wedgeprops={"edgecolor":'white'})
        for i in range(len(data[feat].value_counts().index)):
            if i == idx_client:
                collec[1][idx_client].set_fontweight("bold")
                collec[1][idx_client].set_fontsize(13)
                collec[2][idx_client].set_fontweight("bold")
                collec[2][idx_client].set_fontsize(13)
            else:
                if list_val[i] > 0.04 :
                    collec[1][i].set_fontsize(6)
                    collec[2][i].set_fontsize(6)
                    collec[2][i].set_color("white")
                    collec[2][i].set_fontweight("bold")
                else:
                    collec[1][i].set_fontsize(6)
                    collec[2][i].remove()
        ax.set_title(feat)
    #print(features[features["Row"]==feat]["Description"].values[0])
    return(fig, info)

def scatter(data, feat1, feat2):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.scatter(data[feat1], data[feat2])
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f"Corrélation entre {feat1} et {feat2}")
    return(fig)

st.set_page_config(page_title="Prédiction de la capacité de remboursement d'un demandeur de prêt",
                   page_icon="🏦",
                   layout="wide",
                   initial_sidebar_state="expanded")

# set up de differente taille de font
st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown("""<style>.big-font {font-size:30px !important;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>.less-font {font-size:20px !important;}</style>""", unsafe_allow_html=True)
st.markdown("""<style>.sml-font {font-size:18px !important;}</style>""", unsafe_allow_html=True)

with st.container():
    st.title("Prédiction de la capacité de remboursement d'un demandeur de prêt")
    st.markdown("❗*Cet outil permet d'assister à la prise de décision et doit être utilisé conjointement avec une analyse approfondie réalisée par un professionel*❗")
    st.markdown('##')


# set up de la sidebar
req_i = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/id_client")
resultat_i = req_i.json()

# permet de changer la taille de la sidebar (en l'occurence, augmentation par rapport à la taille standard pour pouvoir voir la jauge)
st.markdown('''
    <style>
        section[data-testid="stSidebar"]{width: 400px !important;}
    </style>
''',unsafe_allow_html=True)
st.sidebar.markdown('<p class="big-font">Selection du client</p>', unsafe_allow_html=True)
option = st.sidebar.selectbox("ID du demandeur de crédit",(resultat_i["list_id"]))

schema = {"num_client": option, "feat":"string"}

# texte d'explication
st.markdown('<p class="less-font">En cochant la case *Analyse* ci-dessous vous obtiendrez les résultats pour le client spécifié à partir de la liste déroulante située dans la barre latérale gauche.</p>', unsafe_allow_html=True)
st.markdown('<p class="sml-font">Veuillez décocher la case avant de sélectionner un nouveau client.</p>', unsafe_allow_html=True)
mycheckb = st.checkbox("Analyse", key='one')
st.markdown("##")

list_tab = ["Top 10 des variables explicatives\ndu modèle","Corrélation entre les variables \ndu modèle","Top 9 des variables explicatives \ndu résultat", "Positionnement du demandeur de crédit \npar rapport aux autres demandes", "Positionnement du demandeur de crédit \nsur une variable en particulier"]
whitespace = 30
tab1, tab2, tab3, tab4, tab5 = st.tabs([s.center(whitespace, "\u2001") for s in list_tab])
# mise en page des tabs
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 2px;
    }
    
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size:14px;}

	.stTabs [data-baseweb="tab"] {
		height: 80px;
        white-space: pre-wrap;
		background-color: #F0F2F6;
        margin: 0px 2px 0px 2px;
        gap: 1px;
        border-radius: 4px 4px 0px 0px;
        padding-right: 5px;
        padding-left: 5px;
    }
                
    .stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
        color: #000000;
	}

</style>""", unsafe_allow_html=True)


req0 = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/get_data", json=schema)
resultat0 = req0.json()
data = pd.DataFrame(resultat0["data"])
data2 = data.copy()

with tab1:
    st.markdown("<h1 style='text-align: center; font-size:20px;'>Les 10 variables explicatives principales du modèle</h1>", unsafe_allow_html=True)
    st.markdown("##")
    st.image("feat_g.png")
    st.markdown("##")
    st.markdown(f"<p class='less-font'>Nombre le plus courant de devise emprunté auprès d'institutions financières tierces : {data['NB_CURRENCY_OFI'].mode()[0]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Nombre de jour moyen depuis le dernier remboursement de crédit auprès d'institutions financières tierces : {-data['DAYS_SINCE_LAST_REIMB_OFI'].mean():.2f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Note moyenne EXT_SOURCE 3 (entre 0 et 1) : {data['EXT_SOURCE_3'].mean():.2f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Note moyenne EXT_SOURCE 2 (entre 0 et 1) : {data['EXT_SOURCE_2'].mean():.2f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Montant moyen de la dette en cours auprès d'institutions financières tierces : {data['TOTAL_CURR_DEBT_OFI'].mean():,.0f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Montant du down payment moyen : {-data['DOWN_PAYMENT'].mean():,.0f}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Taux de paiement moyen (annuité/credit) : {data['PAYMENT_RATE'].mean()*100:.2f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Nombre le plus courant de crédit en cours auprès d'institutions financières tierces : {data['NB_ACTIVE_CR_OFI'].mode()[0]}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='less-font'>Part moyenne des emprunts précédents dont les mensualités ont été payées dans les délais prévus : {data['PREV_PERC_INST_PAID_ON_TIME'].mean()*100:.2f}%</p>", unsafe_allow_html=True)
    if data['CODE_GENDER'].mode()[0]==0:
        gender = "femmes"
    else : 
        gender = "hommes"
    st.markdown(f"<p class='less-font'>La majorité des demandeurs de crédit sont des {gender}</p>", unsafe_allow_html=True)

with tab2:
    var_num=[]
    for feat in data.columns:
        if len(data[feat].value_counts().index) > 5:
            var_num.append(feat)
    st.markdown("<h1 style='text-align: center; font-size:15px;'>Sélectionnez les deux variables que vous souhaitez analyser</h1>", unsafe_allow_html=True)
    cola, colb, colc = st.columns([0.20, 0.6, 0.20], gap='small')
    with colb:
        feat_1 = st.selectbox("Variable 1",var_num)
        feat_2 = st.selectbox("Variable 2",var_num)
        st.markdown("##")
        fig0 = scatter(data, feat_1, feat_2)
        st.pyplot(fig0)

with tab3:
    st.markdown("##")
    placeholder3 = st.empty()
    placeholder3.text("Veuillez sélectionner un numéro de client dans la liste déroulante située dans la barre latérale gauche, \npuis cochez la checkbox 'Analyse' pour obtenir les résultats.")
with tab4:
    st.markdown("##")
    placeholder4 = st.empty()
    placeholder4.text("Veuillez sélectionner un numéro de client dans la liste déroulante située dans la barre latérale gauche, \npuis cochez la checkbox 'Analyse' pour obtenir les résultats.")
with tab5:
    st.markdown("##")
    placeholder5 = st.empty()
    if mycheckb:
        placeholder5 = st.empty()
    else:
        placeholder5.text("Veuillez sélectionner un numéro de client dans la liste déroulante située dans la barre latérale gauche, \npuis cochez la checkbox 'Analyse' pour obtenir les résultats.")

# st.button doesn’t keep its state when the app gets rerun
# when you change the selectbox, the app gets rerun, and the button is no longer “clicked”
# one way to work around this is to switch out the button for a checkbox
if mycheckb:
    
    # sidebar
    # infos générales
    req = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/perso_info", json=schema)
    resultat = req.json()
    if resultat["gender"] == 0:
        st.sidebar.markdown(f'Genre:  \tFemale')
    else:
        st.sidebar.markdown(f'Genre:  \tMale')
    st.sidebar.markdown(f'Situation familiale:  \t{resultat["family"]}')
    st.sidebar.markdown(f"Nombre d'enfants:  \t{resultat['nb_child']}")
    st.sidebar.markdown(f"Montant du crédit demandé:  \t{round(resultat['credit']):,}")
    st.sidebar.markdown(f"Revenu:  \t{round(resultat['income_amount']):,}")
    st.sidebar.markdown(f"Source du revenu:  \t{resultat['income_type']}")
    # résultat
    st.sidebar.markdown('##')
    req1 = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/predict", json=schema)
    resultat1 = req1.json()
    st.sidebar.markdown(f"<h1 style=font-size:18px;'>{resultat1['verdict']}</h1>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<h1 style=font-size:18px;'>{resultat1['proba']}</h1>", unsafe_allow_html=True)
    # jauge
    req2 = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/gauge", json=schema)
    resultat2 = req2.json()
    with st.sidebar:
        st.components.v1.html(resultat2['fig'], height=280)


    with tab3: 
        placeholder3.empty()
        req3 = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/explanation", json=schema)
        resultat3 = req3.json()
        df = pd.DataFrame(resultat3["df_feat"])
        df["Shap_value"] = pd.to_numeric(df["Shap_value"])
        df_viz = df.style.background_gradient(cmap='seismic', subset=["Shap_value"], vmin=-1, vmax=1)
        st.markdown("<h1 style='text-align: center; font-size:20px;'>Détail des 9 variables principales à l'origine du résultat obtenu</h1>", unsafe_allow_html=True)
        try:
            st.dataframe(df_viz)
        except:
            st.markdown("L'image qui devrait apparaitre à cet emplacement est un tableau indiquant les principales variables explicatives du résultat (accord ou refus du crédit) pour le client étudié")
        try:
            st.components.v1.html(resultat3["fig"])
        except:
            st.markdown("L'image qui devrait apparaitre à cet emplacement correspond au forceplot pour visiualiser les principales variables explicatives du résultat (accord ou refus du crédit) pour le client étudié")
    
    with tab4:
        placeholder4.empty()
        fig1, info1 = local_comps(data2, df)
        st.markdown("<h1 style='text-align: center; font-size:20px;'>Positionnement du demandeur de crédit sur les 9 variables explicatives principales</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size:18px;'>(les valeurs pour le demandeur de crédit sont indiquées en orange 🔶)</h1>", unsafe_allow_html=True)
        st.markdown('##')
        try:
            st.pyplot(fig1)
            st.markdown("<h1 style=font-size:15px;'>Informations détaillées</h1>", unsafe_allow_html=True)
            for el in info1:
                st.text(el)
        except:
            st.markdown("L'image qui devrait apparaitre à cet emplacement correspond à une série de graphiques indiquant le positionnement du demandeur de crédit par rapport à ses pairs sur les variables explicatives principales")


    with tab5:
        st.markdown("<h1 style='text-align: center; font-size:20px;'>Positionnement du demandeur de crédit sur la variable explicative de votre choix</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size:18px;'>(la valeur pour le demandeur de crédit est indiquée en orange 🔶)</h1>", unsafe_allow_html=True)
        st.markdown('##')
        col1, col2, col3 = st.columns([0.25, 0.5, 0.25], gap='small')
        with col2:
            with st.spinner("Veuillez patienter..."):
                option2 = st.selectbox("Choix de la variable à analyser",(data.columns.tolist()))
                time.sleep(3)
                schema2 = {"num_client": 0, "feat":option2}
                req_last = requests.post("https://credit-app-backend-730c22127ecd.herokuapp.com/description", json=schema2)
                resultat_last = req_last.json()
                st.markdown("Description de la variable:")
                st.text(resultat_last['description'])
                fig2, info2 = custom_comps(data, option, option2)
                st.pyplot(fig2)
                for el in info2:
                    st.text(el)
        
