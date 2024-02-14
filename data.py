import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,recall_score,precision_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC







####### creation une fonction pour la modularité
def main():
    st.title("application machine learning de detection card fraud")
    st.subheader("zakaria Sarhan")
    ######### importation des données
    @st.cache_data(persist=True)
    def load_data():
        data=pd.read_csv("C:/Users/hp/Desktop/creditcard.csv")
        return data
    ####### affichage notre data
    df=load_data()
    df_sample=df.sample(100)
    if st.sidebar.checkbox("afficher le jeu de données d'un echantillion 100 ",False):
        st.subheader("jeu de données de carte credit bancaire")
        st.write(df_sample)
        ######## diviser notre jeu de données
    seed=123

    @st.cache(persist=True)
    def split(df):
        x = df.drop("Class", axis=1)
        y = df["Class"]
        x_train,x_test,y_train,y_test=train_test_split(x,y,
                                test_size=0.2,
                                random_state=seed,
                                stratify=y
        )
        return x_train,x_test,y_train,y_test

    x_train,x_test,y_train,y_test= split(df)


    ###### creer un classificateur

    classifier=st.sidebar.selectbox("classificateur",
                                    ("random forest", "regression logistique","SVM")


    )
    graphique_confusion = st.sidebar.checkbox(
        "affichage  graphique de matrix de confusion", False

    )



    if classifier=="random forest":
        st.sidebar.subheader("hyper paramétres de model")
        n_arbre= st.sidebar.number_input(
            "choisir le nombre d'arbre decesion",100,1000,step=10
        )
        profondeur= st.sidebar.number_input(
            "profondeur maximale de l'arbe",1,20,step=1


        )
        boostrape= st.sidebar.radio(
            "echantillion lors de creation de l'arbre",
             ("True","False")
        )


        ####### affichage entrainnement de model

        if st.sidebar.button("excution",key="classify"):
            st.subheader("random forest resultat")
            ###### initiation notre model
            model=RandomForestClassifier(n_estimators=n_arbre,
                                         max_depth=profondeur,
                                         bootstrap=(boostrape==True)

            )


            model.fit(x_train,y_train)
            y_pred=model.predict(x_test)
            ###### calculer les métrics
            score=np.round(accuracy_score(y_test,y_pred),2)
            recall= np.round(recall_score(y_test,y_pred),2)
            precesion_score= np.round(precision_score(y_test,y_pred),2)
            f_score=np.round(f1_score(y_test,y_pred),2)


            st.write("score:",score)
            st.write("recall:",recall)
            st.write("precesion_score:",precesion_score)
            st.write("f_score:", f_score)



            cm = confusion_matrix(y_test, y_pred)
            if graphique_confusion:
                plt.figure(figsize=(6, 4))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=["transaction authentique", "transaction frauduleuse"],
                            yticklabels=["transaction authentique", "transaction frauduleuse"])
                plt.xlabel("Prédictions")
                plt.ylabel("Vraies valeurs")
                plt.title("Matrice de Confusion")
                st.pyplot()
                st.write("confusion matrix des class ")
                st.write(cm)


        ######## regression logistique

    if classifier == "regression logistique":
            st.sidebar.subheader("hyper paramétres de model")
            hyp_c = st.sidebar.number_input(
                "choisir la valeur de régularisation", 0.01, 10.0
            )
            n_max_itere = st.sidebar.number_input(
                "nombre d'itération ", 100, 1000, step=10

            )



            if st.sidebar.button("exécution",key="classify"):
                st.subheader("regresiion logistique resultat")

                model = LogisticRegression(C=hyp_c,
                                            max_iter=n_max_itere,
                                           random_state=seed


                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                ###### calculer les métrics
                accuracy = np.round(accuracy_score(y_test, y_pred), 2)
                recall = np.round(recall_score(y_test, y_pred), 2)
                precesion_score = np.round(precision_score(y_test, y_pred), 2)
                f_score = np.round(f1_score(y_test, y_pred), 2)

                st.write("accuracy:", accuracy)
                st.write("recall:", recall)
                st.write("precesion_score:", precesion_score)
                st.write("f_score:", f_score)

                cm = confusion_matrix(y_test, y_pred)
                if graphique_confusion:
                    plt.figure(figsize=(6, 4))
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=["transaction authentique", "transaction frauduleuse"],
                                yticklabels=["transaction authentique", "transaction frauduleuse"])
                    plt.xlabel("Prédictions")
                    plt.ylabel("Vraies valeurs")
                    plt.title("Matrice de Confusion")
                    st.pyplot()
                    st.write("confusion matrix des class ")
                    st.write(cm)



            ####### model support vecteur machine


    if classifier == "SVM":
            st.sidebar.subheader("hyper paramétres de model")
            hy_c = st.sidebar.number_input(
                "choisir la valeur de régularisation", 0.01, 10.0
            )
            gamma = st.sidebar.radio(
                ("dégré des coeffusion"), ("scale","auto")

            )

            n_max_tr=st.sidebar.number_input("nobmre itération",100,1000,step=10
            )



            if st.sidebar.button("exécution",key="classify"):
                st.subheader("support vecteur machine resultat")

                model = SVC(C=hy_c,
                             max_iter=n_max_tr,
                             gamma=gamma


                )

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                ###### calculer les métrics
                accuracy = np.round(accuracy_score(y_test, y_pred), 2)
                recall = np.round(recall_score(y_test, y_pred), 2)
                precesion_score = np.round(precision_score(y_test, y_pred), 2)
                f_score = np.round(f1_score(y_test, y_pred), 2)

                st.write("accuracy:", accuracy)
                st.write("recall:", recall)
                st.write("precesion_score:", precesion_score)
                st.write("f_score:", f_score)

                cm = confusion_matrix(y_test, y_pred)
                if graphique_confusion:
                    plt.figure(figsize=(6, 4))
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=["transaction authentique", "transaction frauduleuse"],
                                yticklabels=["transaction authentique", "transaction frauduleuse"])
                    plt.xlabel("Prédictions")
                    plt.ylabel("Vraies valeurs")
                    plt.title("Matrice de Confusion")
                    st.pyplot()
                    st.write("confusion matrix des class ")
                    st.write(cm)






















if __name__ == '__main__':
    main()

