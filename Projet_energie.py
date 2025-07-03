import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import statsmodels.api as sm
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from PIL import Image
import joblib

df=pd.read_csv("eco2mix-regional-cons-def.csv",sep=";")

st.sidebar.title("Projet énergie")

pages=["Présentation","Exploration et pré-processing","Enrichissement de la base","Visualisations","Machine learning ","Conclusion"]

page=st.sidebar.radio("Menu",pages)

if page == pages[0] :
    st.write("### Objectif")
    st.write("L'objectif du projet est de constater le phasage entre la consommation et la production énergétique au niveau national et au niveau régional (risque de black out notamment).")
    st.write("### Dataset")
    image=Image.open("Carte_ODRE.jpg")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True)
    st.write("Le jeu de données est issu de la source de données de l'ODRE (Open Data Réseaux Énergies) avec un accès à toutes les informations de consommation et de production d'électricité par filière par jour (toutes les 1/2 heure) et par région depuis 2013.")
    st.write("Les données proviennent d’une application éCO2mix et sont mises à disposition sous data.gouv.fr ou sur odre.opendatasoft.com .")
    st.write('Extrait du data set : 10 premières lignes')
    st.dataframe(df.head(10))
    st.write("Contenu du jeu de données :")
    st.write("●  la consommation d'électricité")
    st.write("●  la production d'électricité par filière")
    st.write("●  la consommation des pompes dans les Stations de Transfert d'Energie par Pompage (STEP)")
    st.write("●  le solde des échanges avec les régions limitrophes")
    st.write("●  TCO : le Taux de COuverture (TCO) de chaque filière de production, soit la part d'une filière dans la production totale")
    st.write("●  TCH : le Taux de CHarge (TCH) ou facteur de charge (FC) d'une filière représente son volume de production par rapport à la capacité de production installée et en service de cette filière.")
    
if page == pages[4] :
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯  Valeur cible", "🧹 Preprocessing","📊 Modélisation", "⚙️ Optimisation","📈 Visualisations","⏳ Prophet"])

    with tab1:
          st.header("Choix de la valeur cible")
          st.write("L’un des objectifs de cette modélisation est de prédire le risque de black-out électrique en France, c’est-à-dire une situation où la demande en électricité ne pourrait plus être satisfaite par l’offre, provoquant des coupures généralisées.")
          st.write("Dans cette optique, nous avions choisi initialement comme valeur cible le solde électrique, défini comme : ")
          st.write("Solde électrique = Consommation d’électricité – Production totale (toutes filières confondues)")
          st.write("Limites :")
          st.write("- Données incomplètes sur les capacités de production maximales")
          st.write("- Résultats non concluants des modèles prédictifs testés ")
          st.write("Valeur cible retenue :")
          st.write("- Prédiction de la consommation électrique par jour")

    with tab2 :
        st.write("### Principales étapes de préparation du dataset")
        st.write("1. Encodage des variables non numériques :")
        st.write("- Transformation des valeurs temporelles avec encodage cyclique pour le mois et le jour ")
        st.write("- Transformation des colonnes en valeur booléenne (période de chauffe)")
        st.write("2. Séparation des données : La variable cible a été séparée des variables explicatives.")
        st.write("3. Séparation Train/Test : Le dataset a été divisé en un ensemble d’entraînement (80 %) et un ensemble de test (20 %).")
        st.write("4. Gestion des valeurs manquantes : Les valeurs manquantes ont été imputées avec la moyenne des colonnes, à l’aide de SimpleImputer.")
        st.write("5. Standardisation des variables : Les valeurs numériques ont été standardisées pour garantir une échelle comparable entre les variables avec StandardScaler.")
    
    with tab3:
          st.write("### Choix des modèles entraînés :")
          st.write("Pour la modélisation, nous avons retenu 4 modèles différents :")
          st.write("- la régression linéaire : approche simple, interprétable et rapide à mettre en oeuvre")
          st.write("- un arbre de décision : facilement interprétable mais présente un risque de surapprentissage")
          st.write("- une forêt aléatoire : approche optimisée de l'arbre de décision")
          st.write("- le modèle XG boost : modèle qui permet de réduire le risque d'apprentissage")
          
          df_ML=pd.read_csv("df_conso_ML.csv",sep=",")
          df_ML_rest=df_ML.drop(['mois','jour','Région','region'],axis=1)
          x=df_ML_rest.drop('consommation',axis=1)
          y=df_ML_rest['consommation']
          X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
          imputer=SimpleImputer (strategy='mean')
          col=['TMin','TMax','TMoy']
          X_train[col]=imputer.fit_transform(X_train[col])
          X_test[col]=imputer.transform(X_test[col])
          X_train['periode_de_chauffe']=X_train['periode_de_chauffe'].astype('int')
          X_test['periode_de_chauffe']=X_test['periode_de_chauffe'].astype('int')
          datetime_col_train = X_train[['date']]
          datetime_col_test = X_test[['date']]
          X_train_numeric = X_train.drop(columns=['date'])
          X_test_numeric = X_test.drop(columns=['date'])
          scaler=StandardScaler()
          scaler.fit(X_train_numeric)
          X_train_scaled=scaler.transform (X_train_numeric)
          X_test_scaled=scaler.transform(X_test_numeric)
          
          
          st.write("### Extrait du dataset utilisé pour la modélisation")
          if st.checkbox("Afficher le dataset utilisé pour le ML") :
                   st.dataframe(df_ML_rest.head())
        
          st.write("### Résultats de chaque modèle :")
          lr=joblib.load("model_lr.pkl")
          dtr=joblib.load("model_dtr.pkl")
          rfr=joblib.load("model_rfr.pkl")
          xgb=joblib.load("model_xgb.pkl")

          y_pred_lr=lr.predict(X_test_scaled)
          y_pred_dtr=dtr.predict(X_test_scaled)
          y_pred_rfr=rfr.predict(X_test_scaled)
          y_pred_xgb=xgb.predict(X_test_scaled)
          
          modele_choisi=st.selectbox(label='Choix du modèle', options=['Linear Regression','Decision Tree','Random Forest','XG Boost'])
          st.write("Résultats des métriques (jeu de test) : ")
          
          def train_model_r2(modele_choisi) :
            if modele_choisi == 'Linear Regression' :
                y_pred=y_pred_lr
            elif modele_choisi == 'Decision Tree' :
                y_pred=y_pred_dtr
            elif modele_choisi == 'Random Forest' :
                y_pred=y_pred_rfr
            elif modele_choisi == 'XG Boost' :
                y_pred=y_pred_xgb
            r2=r2_score(y_test, y_pred)
            mae=mean_absolute_error(y_test, y_pred)
            return r2
          st.write("Score : ",train_model_r2(modele_choisi))
    
          def train_model_mae(modele_choisi) :
            if modele_choisi == 'Linear Regression' :
                y_pred=y_pred_lr
            elif modele_choisi == 'Decision Tree' :
                y_pred=y_pred_dtr
            elif modele_choisi == 'Random Forest' :
                y_pred=y_pred_rfr
            elif modele_choisi == 'XG Boost' :
                y_pred=y_pred_xgb
            mae=mean_absolute_error(y_test, y_pred)
            return mae
          st.write("MAE : ",train_model_mae(modele_choisi))

          def train_model_mse(modele_choisi) :
            if modele_choisi == 'Linear Regression' :
                y_pred=y_pred_lr
            elif modele_choisi == 'Decision Tree' :
                y_pred=y_pred_dtr
            elif modele_choisi == 'Random Forest' :
                y_pred=y_pred_rfr
            elif modele_choisi == 'XG Boost' :
                y_pred=y_pred_xgb
            mse=mean_squared_error(y_test, y_pred)
            return mse
          st.write("MSE : ",train_model_mse(modele_choisi))

          def train_model_rmse(modele_choisi) :
            if modele_choisi == 'Linear Regression' :
                y_pred=y_pred_lr
            elif modele_choisi == 'Decision Tree' :
                y_pred=y_pred_dtr
            elif modele_choisi == 'Random Forest' :
                y_pred=y_pred_rfr
            elif modele_choisi == 'XG Boost' :
                y_pred=y_pred_xgb
            rmse= mean_squared_error(y_test, y_pred)**0.5
            return rmse
          st.write("RMSE : ",train_model_rmse(modele_choisi))
          
          def comment_model(modele_choisi) :
            if modele_choisi == 'Linear Regression' :
                comment=st.markdown(" Le modèle **Linear regression** montre des performances correctes, mais limitées. Il capture les tendances globales mais a du mal à modéliser les variations fines de la consommation.")
            elif modele_choisi == 'Decision Tree' :
                comment=st.markdown("Le modèle **Decision Tree** présente un surapprentissage évident. L’erreur est nulle sur le jeu d'entraînement (score train = 1) mais nettement plus élevée sur le jeu de test, ce qui révèle une mauvaise généralisation.")
            elif modele_choisi == 'Random Forest' :
                comment=st.markdown("Le modèle **Random Forest** améliore nettement la performance par rapport à l’arbre unique, avec une bonne capacité de généralisation. Il réduit considérablement les erreurs de test tout en gardant un excellent score R².")
            elif modele_choisi == 'XG Boost' :
                comment=st.markdown("Le modèle **XG Boost** surpasse l’ensemble des autres modèles sur toutes les métriques du jeu de test. Il présente les meilleures performances en termes de précision et de généralisation.")
            return comment
          comment_model(modele_choisi)
          
          
          

    with tab4:
        st.markdown("*Les modèles RandomForest et XGBoost affichent déjà d’excellentes performances (plus de 97 %). Cela soulève la question de l’utilité d’une optimisation supplémentaire et du risque possible de surapprentissage. Dans cette perspective, nous avons considéré que l’optimisation des hyperparamètres ne viserait pas uniquement à améliorer la précision brute du modèle, mais également à renforcer sa stabilité et à optimiser le risque de surapprentissage.*")
        st.write("### Choix des modèles à optimiser")
        st.write("Parmi les modèles évalués, XGBoost et Random Forest ont d’excellentes performances. Les performances de ces 2 modèles sont très proches. Sur les visualisations ci-dessous qui comparent la prédictions au réel, on constate que le modèle XG Boost est légèrement plus centré sur la droite rouge qui représente l’égalité parfaite entre les valeurs prédites et les observations réelles :")
        st.image("graph_pred_vs_reel.jpg")
        st.write("### Optimisation des modèles avec Grid Search")
        st.write("##### - Random Forest :")
        st.write("L’ajustement du nombre d’arbres (n_estimators) et la profondeur (max_depth) permet d’éviter le sur-apprentissage. La sélection automatique des features (max_features) améliore la diversité des arbres, donc la précision.")
        if st.checkbox("Afficher le code",key="code_rfr") :
            code_1="""
            param_grid_rfr = {
            'n_estimators': [100, 200],  # Nombre d'arbres
            'max_depth': [None, 10, 20], # Profondeur des arbres
            'min_samples_split': [2, 5], # Nombre minimal d'échantillons
            'max_features': ['sqrt', 'log2', None]  # nombre de variables utilisées
            }
            
            grid_search_rfr = GridSearchCV(rfr, param_grid_rfr, cv=5, 
            scoring='neg_root_mean_squared_error', n_jobs=-1) # Initialiser GridSearchCV

            grid_search_rfr.fit(X_train_numeric, y_train) # Entraîner GridSearch

            print("Meilleurs paramètres :", grid_search_rfr.best_params_)

            """
            st.code(code_1, language="python")
        st.markdown("**Les meilleurs paramètres pour le modèle Random Forest sont les suivants :**")
        st.markdown("max_depth: **None**<br>max_features: **sqrt**<br>min_samples_split: **2**<br>n_estimators : **200**", unsafe_allow_html=True)
        
        st.write("##### - XG Boost :")
        st.write("A l’image de Random Forest, l’ajustement du nombre d’arbres (n_estimators) et la profondeur (max_depth) a pour but d’éviter le sur-apprentissage. De plus, subsample et colsample_bytree réduisent la variance pour éviter le sur-apprentissage. Enfin, le learning rate a quant à lui un impact sur la durée de l'entraînement des données. ")
        if st.checkbox("Afficher le code",key="code_xgb") :
            code_2="""
            param_grid_xgb = {
                'n_estimators': [100, 200, 500],   # Nombre d'arbres
                'learning_rate': [0.01, 0.05, 0.1], # Taux d'apprentissage
                'max_depth': [3, 6, 9],          # Profondeur des arbres
                'subsample': [0.6,0.8, 1],        # Pourcentage de données utilisées
                'colsample_bytree': [0.6,0.8, 1]  # Pourcentage de variables utilisées
            }

            grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5,
            scoring='neg_root_mean_squared_error', n_jobs=-1) # Initialiser GridSearchCV

            grid_search_xgb.fit(X_train_numeric, y_train) # Entraîner GridSearch

            print("Meilleurs paramètres :", grid_search_xgb.best_params_)
            """
            st.code(code_2, language="python")
        
        st.markdown("**Les meilleurs paramètres pour le modèle XG Boost sont les suivants :**")
        st.markdown("colsample_bytree : **0.8**<br>learning_rate : **0.1**<br>max_depth :**9**<br>n_estimators : **500**<br>subsample : **1**", unsafe_allow_html=True)

        st.write("### Comparaison des résultats")
        st.image("tab_recap_metr.jpg")
        st.write("On constate que l’optimisation des hyperparamètres a permis de réduire les métriques MAE, MSE et RMSE, ainsi qu’une légère augmentation du score R².")
        st.write("L’amélioration des métriques est plus marquée pour XGBoost que Random Forest.")

    with tab6:
        st.markdown("*La prédiction de la consommation d’électricité en France s’inscrit dans un cadre typique de modélisation de séries temporelles. Dans ce contexte, nous avons choisi d’expérimenter Prophet, un modèle développé par Facebook spécifiquement conçu pour la prévision de séries chronologiques.*")
        st.write("##### Modélisation")
        st.write("Prophet présente l’avantage d’une mise en œuvre rapide et simple, ne nécessitant que deux variables principales : la date (ds) et la valeur cible (y).")
        st.write("Nous avons paramétré le modèle avec comme valeur cible la consommation et une séparation du jeu d'entraînement et de test de 80%/20%.")
        
        if st.checkbox("Afficher le code",key="code_prophet") :
            code_3="""
            from prophet import Prophet

            # DF prophet
            df_prophet=df_conso_tot.groupby('date').agg(
                {'consommation':'sum'}).reset_index()
            df_prophet=df_prophet.rename(columns={'date': 'ds','consommation':'y'})

            # Séparation jeu d'enrainement et de test
            split_index = int(len(df_prophet) * 0.8)
            train_prophet = df_prophet.iloc[:split_index]
            test_prophet = df_prophet.iloc[split_index:]

            # Entrainement
            model = Prophet()
            model.fit(train_prophet)

            # Prédictions
            future = model.make_future_dataframe(periods=len(test_prophet), freq='D')
            forecast = model.predict(future)

            # Modélisation
            fig = model.plot(forecast)
            plt.show()

            # Composantes de la série temporelle
            model.plot_components(forecast);
            """
            st.code(code_3, language="python")

        st.write("Il permet de modéliser automatiquement les tendances, les saisonnalités multiples et les effets de jours fériés. Voici les tendances identifiées sur notre série : ")
     
        trend=st.selectbox(label='Choix de la tendance', options=["par année","sur la semaine","sur l'année"])
        def trend_show(trend) :
            if trend== "par année" :
                img=st.image("prophet_trend.jpg")
            elif trend== "sur la semaine" :
                img=st.image("prophet_weekly.jpg")
            elif trend == "sur l'année" :
                img=st.image("prophet_yearly.jpg")
            return img
        trend_show(trend)
    




             


