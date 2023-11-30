from django.http import HttpResponse
from django.template import loader
from django.shortcuts import render
from django.template import RequestContext
from django.templatetags.static import static
# from IPython.core.display import HTML
# HTML('''<div class="flourish-embed flourish-cards" data-src="visualisation/1810417" data-url="https://flo.uri.sh/visualisation/1810417/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
# HTML('''<div class="flourish-embed flourish-cards" data-src="visualisation/1816605" data-url="https://flo.uri.sh/visualisation/1816605/embed"><script src="https://public.flourish.studio/resources/embed.js"></script></div>''')
# math opeations
import math
import plotly.tools as tls
import plotly.figure_factory as ff
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# produce random numbers
import random
# to load json files
import json
# datetime oprations
from datetime import timedelta
# to get web contents
from urllib.request import urlopen
# for numerical analyiss
import numpy as np
# to store and process data in dataframe
import pandas as pd
# basic visualization package
import matplotlib.pyplot as plt
# advanced ploting
import seaborn as sns
# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
from plotly.subplots import make_subplots
# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters() 
# hide warnings
import warnings
warnings.filterwarnings('ignore')

# to USA states details
import us
import calendar
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 

#seaborn plot style
sns.set_style('darkgrid')
# list files
# ==========

# !ls ../input/corona-virus-report
# Full data
# =========

page_features = ['likes','visitors','daily_interest','category']
derived = ['derived_{}'.format(i) for i in range(1,26)]
essential_features = ['C_{}'.format(i) for i in range(1,6)]
base_features = ['base_time','post_length','share_count','promotion_status','H_hrs']

weekday = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
base_DT = ['BDT_Sun', 'BDT_Mon', 'BDT_Tue', 'BDT_Wed', 'BDT_Thu', 'BDT_Fri', 'BDT_Sat']
target = ['No_of_Comments_in_H_hours']
cols = page_features+derived+essential_features+base_features+weekday+base_DT+target

Testing =pd.read_csv("Features_TestSet.csv",names = cols, header = None)

# Supprimer la colonne "promotion status"
# IL n'y a que des valeurs nulles
Testing = Testing.drop("promotion_status", axis=1)

# Supprimer les colonnes 47 à 53 (ON VA utiliser seulement la date de publication pour l'analyse de nos données ! pour l'affichage)
Testing = Testing.drop(Testing.columns[45:52], axis=1)




def index1(request):
    
    return render(request,"template1.html")

def index2(request):
    #template = loader.get_template("template0.html")
    #data = {'age': [23, 25, 27, 28, 30, 31, 33, 35, 37, 38],
        #'salaire': [40000, 45000, 48000, 50000, 52000, 55000, 58000, 62000, 65000, 70000]}
    #df = pd.DataFrame(data)
    
    if(request.GET['model']=='datavisu1'):
        df1=Testing.copy()
        cat_mapping = {
            1: 'Product/service',
            2: 'Public figure',
            3: 'Retail and consumer merchandise',
            4: 'Athlete',
            5: 'Education website',
            6: 'Arts/entertainment/nightlife',
            7: 'Aerospace/defense',
            8: 'Actor/director',
            9: 'Professional sports team',
            10: 'Travel/leisure',
            11: 'Arts/humanities website',
            12: 'Food/beverages',
            13: 'Record label',
            14: 'Movie',
            15: 'Song',
            16: 'Community',
            17: 'Company',
            18: 'Artist',
            19: 'Non-governmental organization (ngo)',
            20: 'Media/news/publishing',
            21: 'Cars',
            22: 'Clothing',
            23: 'Local business',
            24: 'Musician/band',
            25: 'Politician',
            26: 'News/media website',
            27: 'Education',
            28: 'Author',
            29: 'Sports event',
            30: 'Restaurant/cafe',
            31: 'School sports team',
            32: 'University',
            33: 'Tv show',
            34: 'Website',
            35: 'Outdoor gear/sporting goods',
            36: 'Political party',
            37: 'Sports league',
            38: 'Entertainer',
            39: 'Church/religious organization',
            40: 'Non-profit organization',
            41: 'Automobiles and parts',
            42: 'Tv channel',
            43: 'Telecommunication',
            44: 'Entertainment website',
            45: 'Shopping/retail',
            46: 'Personal blog',
            47: 'App page',
            48: 'Vitamins/supplements',
            49: 'Professional services',
            50: 'Movie theater',
            51: 'Software',
            52: 'Magazine',
            53: 'Electronics',
            54: 'School',
            55: 'Just for fun',
            56: 'Club',
            57: 'Comedian',
            58: 'Sports venue',
            59: 'Sports/recreation/activities',
            60: 'Publisher',
            61: 'Tv network',
            62: 'Health/medical/pharmacy',
            63: 'Studio',
            64: 'Home decor',
            65: 'Jewelry/watches',
            66: 'Writer',
            67: 'Health/beauty',
            68: 'Music video',
            69: 'Appliances',
            70: 'Computers/technology',
            71: 'Insurance company',
            72: 'Music award',
            73: 'Recreation/sports website',
            74: 'Reference website',
            75: 'Business/economy website',
            76: 'Bar',
            77: 'Album',
            78: 'Games/toys',
            79: 'Camera/photo',
            80: 'Book',
            81: 'Producer',
            82: 'Landmark',
            83: 'Cause',
            84: 'Organization',
            85: 'Tv/movie award',
            86: 'Hotel',
            87: 'Health/medical/pharmaceuticals',
            88: 'Transportation',
            89: 'Local/travel website',
            90: 'Musical instrument',
            91: 'Radio station',
            92: 'Other',
            93: 'Computers',
            94: 'Phone/tablet',
            95: 'Coach',
            96: 'Tools/equipment',
            97: 'Internet/software',
            98: 'Bank/financial institution',
            99: 'Society/culture website',
            100: 'Small business',
            101: 'News personality',
            102: 'Teens/kids website',
            103: 'Government official',
            104: 'Photographer',
            105: 'Spas/beauty/personal care',
            106: 'Video game'
        }
        
        # Remplacer les nombres par les noms de catégories dans la colonne appropriée
        df1['category'] = df1['category'].map(cat_mapping)
        # Compter le nombre d'occurrences de chaque catégorie de page
        category_counts = df1['category'].value_counts().reset_index()
    
        # Renommer les colonnes
        category_counts.columns = ['category', 'Count']

        # Sélectionner les 10 premières catégories
        top_10_categories = category_counts.head(10)

        # Créer une figure avec Plotly
        fig = px.bar(top_10_categories, x='category', y='Count')
        fig.update_layout(title="Top 10 Page Categories")
        
        plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2="""<div>
            <br>  
            <br>
            <br>
            <p>We can see on this graph that the most popular category is about proffesional sport team maybe we can deduce that post on this subject maybe be useful but we don't quantity doesn't mean quality</p>
        </div>
        """
        plothtml3=""
        plothtml4=""
        plothtmlbis=""
        plothtmlbis2=""
        plothtmlinteract=""
        
    if(request.GET['model']=='datavisu2'):
        df2=Testing.copy()
        # Sélectionner les colonnes pertinentes pour l'analyse
        columns = ['No_of_Comments_in_H_hours', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        # Créer une nouvelle colonne pour le jour de la semaine correspondant à chaque enregistrement
        df2['day_of_week'] = df2[columns[1:]].idxmax(axis=1)
        
        # Définir l'ordre des catégories des jours de la semaine
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Convertir la colonne 'day_of_week' en une catégorie avec l'ordre spécifié
        df2['day_of_week'] = pd.Categorical(df2['day_of_week'], categories=weekday_order, ordered=True)
        
        # Calculer la somme des commentaires par jour de la semaine
        comments_by_day = df2.groupby('day_of_week')['No_of_Comments_in_H_hours'].sum()
        
        # Créer un histogramme avec Plotly en utilisant l'ordre spécifié des jours de la semaine
        fig = px.bar(comments_by_day, x=comments_by_day.index, y=comments_by_day.values, labels={'x': 'Day of week', 'y': 'No_of_Comments_in_H_hours'})
                        
        plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2="""<div>
            <br>  
            <br>
            <br>
            <p>We can see that the post that are more published is during the weekend our dataset don't show two day info but we can surely say that weekend is a point of reflexion for posting or not our posts</p>
        </div>
        """
        plothtml3=""
        plothtml4=""
        plothtmlbis=""
        plothtmlbis2=""
        plothtmlinteract=""
    if(request.GET['model']=='datavisu3'):
        df3=Testing.copy()
        
        # Filtrer les publications avec une longueur de 0
        df3 = df3[df3['post_length'] > 0]
        # Sélectionner les colonnes pertinentes pour l'analyse
        columns = ['post_length', 'No_of_Comments_in_H_hours']
        
        # Définir les tranches de longueur de publication
        bins = [0, 100, 200, 300, 400, 500, float('inf')]
        labels = ['0-100', '101-200', '201-300', '301-400', '401-500', '>500']
        
        # Ajouter une colonne pour les tranches de longueur de publication
        df3['post_length_bin'] = pd.cut(df3['post_length'], bins=bins, labels=labels, right=False)
        
        # Calculer la somme des commentaires pour chaque tranche de longueur de publication
        comments_by_length = df3.groupby('post_length_bin')['No_of_Comments_in_H_hours'].sum().reset_index()
        
        # Créer un graphique à barres pour la somme des commentaires par tranche de longueur de publication
        fig = go.Figure(data=[
            go.Bar(
                x=comments_by_length['post_length_bin'],
                y=comments_by_length['No_of_Comments_in_H_hours']
            )
        ])
        
        # Paramètres de mise en forme du graphique
        fig.update_layout(
            title='Post Length linked to Comment in H hours',
            xaxis_title='Post length',
            yaxis_title='Sum of No of Comments in H hours'
            )

        
        plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2="""<div>
            <br>  
            <br>
            <br>
            <p>We can see that post with less words are more commun than long post maybe privilege longer post will be good for some communities and these post are lacking obviously</p>
        </div>
        """
        plothtml3=""
        plothtml4=""
        plothtmlbis=""
        plothtmlbis2=""
        plothtmlinteract=""
    
    if(request.GET['model']=='datavisu4'):
        df4=Testing.copy()
        
        # Définir les tranches de nombre de commentaires
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500]
        labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350', '351-400', '401-450', '451-500', '501-550', '551-600', '601-650', '651-700', '701-750', '751-800', '801-850', '851-900', '901-950', '951-1000', '1001-1050', '1051-1100', '1101-1150', '1151-1200', '1201-1250', '1251-1300', '1301-1350', '1351-1400', '1401-1450', '1451-1500']
        
        # Ajouter une colonne pour les tranches de nombre de commentaires
        df4['comment_count_bin'] = pd.cut(df4['No_of_Comments_in_H_hours'], bins=bins, labels=labels, right=False)
        
        # Calculer la moyenne du nombre de partages de post par tranche de nombre de commentaires
        mean_shares_by_comments = df4.groupby('comment_count_bin')['share_count'].median().reset_index()
        
        # Créer un histogramme avec la moyenne du nombre de partages de post par tranche de nombre de commentaires
        fig = px.bar(mean_shares_by_comments, x='comment_count_bin', y='share_count', title='Median Share Count by Number of Comments')
        
        # Paramètres de mise en forme de l'histogramme
        fig.update_layout(
            xaxis_title='Number of Comments in Hours',
            yaxis_title='Median Share Count'
        )

        plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2="""<div>
            <br>  
            <br>
            <br>
            <p>The histogram shows a gradual increase in median share counts as the number of comments increases. This could indicate that posts with a higher number of comments tend to have a higher median share count, potentially because they attract more attention and interest from the audience.</p>
        </div>
        """
        
       
        plothtml3=""
        plothtml4=""
        plothtmlbis=""
        plothtmlbis2=""
        plothtmlinteract=""
        
    if(request.GET['model']=='datavisu5'):
        df5=Testing.copy()
        
        # Définir les tranches de nombre de commentaires
        bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500]
        labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350', '351-400', '401-450', '451-500', '501-550', '551-600', '601-650', '651-700', '701-750', '751-800', '801-850', '851-900', '901-950', '951-1000', '1001-1050', '1051-1100', '1101-1150', '1151-1200', '1201-1250', '1251-1300', '1301-1350', '1351-1400', '1401-1450', '1451-1500']
        
        # Ajouter une colonne pour les tranches de nombre de commentaires
        df5['comment_count_bin'] = pd.cut(df5['No_of_Comments_in_H_hours'], bins=bins, labels=labels, right=False)
        
        # Calculer la moyenne du nombre de partages de post par tranche de nombre de commentaires
        mean_likes_by_comments = df5.groupby('comment_count_bin')['likes'].median().reset_index()
        
        # Créer un histogramme avec la moyenne du nombre de partages de post par tranche de nombre de commentaires
        fig = px.bar(mean_likes_by_comments, x='comment_count_bin', y='likes', title='Median likes Count by Number of Comments')
        
        # Paramètres de mise en forme de l'histogramme
        fig.update_layout(
            xaxis_title='Number of Comments in Hours',
            yaxis_title='Median likes Count'
        )
        
        # Calculer la moyenne du nombre de partages de post par tranche de nombre de commentaires
        mean_visitors_by_comments = df5.groupby('comment_count_bin')['visitors'].median().reset_index()
        
        # Créer un histogramme avec la moyenne du nombre de partages de post par tranche de nombre de commentaires
        fig2 = px.bar(mean_visitors_by_comments, x='comment_count_bin', y='visitors', title='Median visitors Count by Number of Comments')
        
        # Paramètres de mise en forme de l'histogramme
        fig2.update_layout(
            xaxis_title='Number of Comments in Hours',
            yaxis_title='Median visitors Count'
        )
        
        plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2=fig2.to_html(full_html=False,default_height=500,default_width=700)
        plothtml3=""
        plothtml4=""
        plothtmlbis="""<div>
            <br>  
            <br>
            <br>
            <p>From the left histogram, it is clear that posts with more comments tend to have higher likes counts, while posts with fewer comments have lower likes counts.</p>
            <br>
            <p>it can be observed that the number of visitors (Count by Number of Comments) to the page or video has increased with the number of comments made on the page. The growth is down significant at the beginning when there are fewer comments. However, the growth is here as the number of comments increases.</p>
            </div>"""
        plothtmlbis2=""
        plothtmlinteract=""
        
    if(request.GET['model']=='datavisu6'):
        df6 =Testing.copy()
      
                
        # Example: Creating a scatter plot for CC4 vs Target Variable
        fig2 = px.scatter(df6, x='C_4', y='No_of_Comments_in_H_hours', title='Scatter Plot of CC4 vs Target Variable')
        
        
        # Example: Creating a scatter plot for CC2 vs CC5
        fig3 = px.scatter(df6, x='C_2', y='C_5', title='Scatter Plot of CC2 vs CC5')
       

        
        plothtml=fig3.to_html(full_html=False,default_height=500,default_width=700)
        plothtml2=fig2.to_html(full_html=False,default_height=500,default_width=700)
        
       
        plothtml3=""
        plothtml4=""
        plothtmlbis="""<div>
        <p>From the first plot, we can observe several interesting trends:

        A large portion of the products has CC2 values ranging from 0 to 2000 and CC5 values ranging from -2000 to 2000. This suggests that these products have relatively stable and balanced engagement over time.
        
        Another section of the products has CC2 values ranging from 0 to 2000 and CC5 values ranging from -5000 to -3000. This indicates that these products experienced a sudden and significant drop in engagement (CC3).
        
        There is also a smaller group of products with high CC2 values and negative CC5 values. These products may have experienced a sudden surge in engagement, causing their CC2 value to increase dramatically while their CC3 value dropped significantly.</p>
        <br>
        <p>Based on the second graph, there seems to be a negative relationship between the number of comments (CC4) and the target variable.

        When the number of comments is low (between 500 and 1000), the target variable seems to have higher values. As the number of comments increases, the target variable starts to decrease. This suggests that higher engagement in comments (leading to more comments) is negatively correlated with the target variable.</p>
        </div>"""
        plothtmlbis2=""
        plothtmlinteract=""
    
    context={
        'plothtml':plothtml,
        'plothtml2':plothtml2,
        'plothtml3':plothtml3,
        'plothtml4':plothtml4,
        'plothtmlbis':plothtmlbis,
        'plothtmlbis2':plothtmlbis2,
        'plothtmlinteract':plothtmlinteract,
        }
    return render(request, "template1.html", context)
    

def index3(request):
    if (request.GET['model']=="model1"):
        df1=Testing.copy()
        data =pd.read_csv("Features_TestSet.csv",names = cols, header = None)
        data = data.drop(['promotion_status'], axis=1)
        X= data.iloc[:,:52]   #all features] #independent columns
        y = data.iloc[:,-1] # pick last column for the target feature
        
        results=[{'model': 'Linear Regression', 'best_params': {}, 'best_score': 878.723979981303, 'test_score': 885.7007446929715}, {'model': 'Ridge Regression', 'best_params': {}, 'best_score': 877.9228700989937, 'test_score': 886.0335012018314}, {'model': 'Random Forest', 'best_params': {}, 'best_score': 476.2067588851948, 'test_score': 439.48283124555695}, {'model': 'SVM Regressor', 'best_params': {}, 'best_score': 1111.8966357463878, 'test_score': 1096.9469769300076}]
        # Plotting test MSE for each model
        model_names = [result['model'] for result in results]
        test_mse = [result['test_score'] for result in results]
        fig = px.bar(x=model_names, y=test_mse, labels={'x':'Models', 'y':'Test MSE'}, title='Comparison of Model Performances')
        fig.update_xaxes(tickangle=45)

            
        
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4=""
        plothtml4 = plothtml4.strip().replace('\n', '')
        plothtmlbis=""
        plothtmlbis2=""" We can see that SVM Regressor is the worst model to apply in our case but we can also see that two model are pretty close like Linear and Ridge regression who are also not verry good to use. 
        
        In conslusion, we can deduce with the mse that random forest is the best way to study our data because of his lower test MSE"""
        plothtmlinteract=""
        
    if (request.GET['model']=="model2"):
        
        
        data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
        data = data.drop(['promotion_status'], axis=1)
        X= data.iloc[:,:52]   #all features] #independent columns
        y = data.iloc[:,-1] # pick last column for the target feature
        
        from sklearn.ensemble import ExtraTreesClassifier
        import numpy as np
        import matplotlib.pyplot as plt
        import plotly.graph_objs as go
        model = ExtraTreesClassifier()
        model.fit(X,y)
        
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        top_features = feat_importances.nlargest(10)
        
        fig = go.Figure(data=[go.Bar(y=top_features.index, x=top_features.values, orientation='h')])
        
        fig.update_layout(title="Top 10 Feature Importances with ExtraTreesClassifier",
        xaxis_title="Importance",
        yaxis_title="Features")
        
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4=""
        plothtml4 = plothtml4.strip().replace('\n', '')
        plothtmlbis=""
        plothtmlbis2=""" Like our data visualization we can see the importance of C_. columns and other column logical like share post or post lenght"""
        plothtmlinteract=""
    if (request.GET['model']=="model3"):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression, mutual_info_regression
        from sklearn.feature_selection import chi2
        
        data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
        data = data.drop(['promotion_status'], axis=1)
        X= data.iloc[:,:52]   #all features] #independent columns
        Y = data.iloc[:,-1] # pick last column for the target feature
        
        from sklearn.preprocessing import MinMaxScaler

        # Scale the features to ensure non-negativity
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        bestfeatures = SelectKBest(score_func=chi2, k=10)
        fit = bestfeatures.fit(X_scaled, Y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        scores = pd.concat([dfcolumns, dfscores], axis=1)
        scores.columns = ['specs', 'score']
        top_features = scores.nlargest(10, 'score')
        
        fig = px.bar(top_features, x='score', y='specs', orientation='h', title="Top 10 Feature Scores")
        
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4=""
        plothtml4 = plothtml4.strip().replace('\n', '')
        plothtmlbis=""
        plothtmlbis2=""" Like our data visualization we can see the importance of C_. columns this time but also the importance of some derived values.
        
        Maybe this method of model is not the best but we can see some result of this.
        
        """
        plothtmlinteract=""
    if (request.GET['model']=="model4"):
        
        data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
        data = data.drop(['promotion_status'], axis=1)
        X= data.iloc[:,:52]   #all features] #independent columns
        y = data.iloc[:,-1] # pick last column for the target feature
        # Fit PCA to your data
        pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
        principal_components = pca.fit_transform(data)
        
        # Create a scatter plot of the first two principal components
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Scatter(
        x=principal_components[:, 0],
        y=principal_components[:, 1],
        mode='markers',
        marker=dict(
        color=y,
        colorscale='viridis',
        colorbar=dict(title='Target')
        )
        ))
        
        fig.update_layout(
        title='PCA - First Two Principal Components',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2'
        )
        
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4="""<div>
        <br>
        <br>
        <p>Based"" on the PCA graph provided, it can be observed that there is a significant variance among the principal components. This indicates that there is a substantial amount of variation among the variables based on their principal component values.

        Principal Component 1 appears to capture a large amount of variation, suggesting that it plays a significant role in features. It is interesting to note that variable with high values for Principal Component 1 also tend to have high values for Principal Component 2. This could suggest that the target have characteristics that place them in both categories of feature.</p>
        </div>"""
        plothtml4 = plothtml4.strip().replace('\n', '')
        plothtmlbis=""
        plothtmlbis2=""""""
        plothtmlinteract=""
    if (request.GET['model']=='model6'):
        import plotly.graph_objects as go
        from sklearn.preprocessing import MinMaxScaler
        data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
        data = data.drop(['promotion_status'], axis=1)
        # Encodage des variables catégoriques
        encoded_data = pd.get_dummies(data, columns=['category'])
        # Sélectionnez toutes les colonnes sauf 'category'
        columns_to_normalize = [col for col in encoded_data.columns if col != 'category']
        
        # Créez un sous-ensemble de données avec les colonnes à normaliser
        data_to_normalize = encoded_data[columns_to_normalize]
        
        # Initialisez le scaler
        scaler = MinMaxScaler()
        
        # Normalisez les colonnes sélectionnées
        normalized_data = scaler.fit_transform(data_to_normalize)
        
        # Remplacez les colonnes normalisées dans le jeu de données d'origine
        encoded_data[columns_to_normalize] = normalized_data
        correlation_matrix = encoded_data.corr()

        filtered_correlation_matrix = correlation_matrix.mask(correlation_matrix < 0.10)
        
        fig = go.Figure(data=go.Heatmap(
        z=filtered_correlation_matrix.values,
        x=filtered_correlation_matrix.columns,
        y=filtered_correlation_matrix.index,
        colorscale='Viridis'))
        
        fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Features",
        yaxis_title="Features"
        )
        fig.update_layout(
            title="Correlation Heatmap",
            xaxis_title="Features",
            yaxis_title="Features",
            width=900, # largeur de 30 pouces pour une résolution de 300 DPI
            height=900 # hauteur de 30 pouces pour une résolution de 300 DPI
        )
        
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4=""""""
        
        plothtmlbis=""
        plothtmlbis2=""""""
        plothtmlinteract=""
        
    if (request.GET['model']=="model5"):
        a=[1.77610261e-04 ,2.59082107e-04, 2.91595966e-04, 8.57772034e-04,
         1.04077522e-03, 1.15762043e-03, 2.74223061e-03, 2.90487709e-02,
         4.42187599e-02, 2.45216323e-01]
        b=['category_Tv channel', 'derived_23', 'category_Education website',
               'category_Politician', 'category_Movie', 'C_3', 'derived_8',
               'derived_13', 'C_5', 'C_2']
        
        fig = px.bar(x=a, y=b, orientation='h')
        
        fig.update_layout(
        xaxis_title="Permutation Importance",
        yaxis_title="Features",
        title="Permutation Importance"
        )
        plothtml=""
        plothtml2=""
        plothtml3=fig.to_html(full_html=False,default_height=500,default_width=700)
        plothtml4="""<div>
            <br>  
            <br>
            <br>
            <p>Based on this heatmap, we can conclude that the most important features for the model are 'C_2', 'C_5', 'derived_13', 'derived_8', 'C_3', and 'category_Movie'. The least important features are 'category_Tv channel', 'category_Education website', and 'category_Politician'.</p>
        </div>"""
        
        plothtmlbis=""
        plothtmlbis2=""""""
        plothtmlinteract=""
        
    
    context={
        'plothtml':plothtml,
        'plothtml2':plothtml2,
        'plothtml3':plothtml3,
        'plothtml4':plothtml4,
        'plothtmlbis':plothtmlbis,
        'plothtmlbis2':plothtmlbis2,
        'plothtmlinteract':plothtmlinteract,
        }
    return render(request,'template1.html',context)

def about2(request):
    
    page_features = ['likes','visitors','daily_interest','category']
    derived = ['derived_{}'.format(i) for i in range(1,26)]
    essential_features = ['C_{}'.format(i) for i in range(1,6)]
    base_features = ['base_time','post_length','share_count','promotion_status','H_hrs']
    
    weekday = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    base_DT = ['BDT_Sun', 'BDT_Mon', 'BDT_Tue', 'BDT_Wed', 'BDT_Thu', 'BDT_Fri', 'BDT_Sat']
    target = ['No_of_Comments_in_H_hours']
    cols = page_features+derived+essential_features+base_features+weekday+base_DT+target
    
    df = pd.read_csv('Features_TestSet.csv',names = cols, header = None,nrows=500)
    
    # SEULEMENT POUR L'AFFICHAGE
    # Parcourir chaque colonne du DataFrame
    for col in df.columns:
        # Vérifier si la colonne contient des valeurs numériques
        if df[col].dtype == 'float64':
            # Vérifier si les valeurs de la colonne sont des entiers
            if df[col].apply(lambda x: x.is_integer()).all():
                # Arrondir les valeurs entières
                df[col] = df[col].astype(int).astype(str).replace('\.0', '', regex=True)
    
    # Supprimer la colonne "promotion status"
    # IL n'y a que des valeurs nulles
    df = df.drop("promotion_status", axis=1)
    
    # Supprimer les colonnes 47 à 53 (ON VA utiliser seulement la date de publication pour l'analyse de nos données ! pour l'affichage)
    df = df.drop(df.columns[45:52], axis=1)
    
    cat_mapping = {
        1: 'Product/service',
        2: 'Public figure',
        3: 'Retail and consumer merchandise',
        4: 'Athlete',
        5: 'Education website',
        6: 'Arts/entertainment/nightlife',
        7: 'Aerospace/defense',
        8: 'Actor/director',
        9: 'Professional sports team',
        10: 'Travel/leisure',
        11: 'Arts/humanities website',
        12: 'Food/beverages',
        13: 'Record label',
        14: 'Movie',
        15: 'Song',
        16: 'Community',
        17: 'Company',
        18: 'Artist',
        19: 'Non-governmental organization (ngo)',
        20: 'Media/news/publishing',
        21: 'Cars',
        22: 'Clothing',
        23: 'Local business',
        24: 'Musician/band',
        25: 'Politician',
        26: 'News/media website',
        27: 'Education',
        28: 'Author',
        29: 'Sports event',
        30: 'Restaurant/cafe',
        31: 'School sports team',
        32: 'University',
        33: 'Tv show',
        34: 'Website',
        35: 'Outdoor gear/sporting goods',
        36: 'Political party',
        37: 'Sports league',
        38: 'Entertainer',
        39: 'Church/religious organization',
        40: 'Non-profit organization',
        41: 'Automobiles and parts',
        42: 'Tv channel',
        43: 'Telecommunication',
        44: 'Entertainment website',
        45: 'Shopping/retail',
        46: 'Personal blog',
        47: 'App page',
        48: 'Vitamins/supplements',
        49: 'Professional services',
        50: 'Movie theater',
        51: 'Software',
        52: 'Magazine',
        53: 'Electronics',
        54: 'School',
        55: 'Just for fun',
        56: 'Club',
        57: 'Comedian',
        58: 'Sports venue',
        59: 'Sports/recreation/activities',
        60: 'Publisher',
        61: 'Tv network',
        62: 'Health/medical/pharmacy',
        63: 'Studio',
        64: 'Home decor',
        65: 'Jewelry/watches',
        66: 'Writer',
        67: 'Health/beauty',
        68: 'Music video',
        69: 'Appliances',
        70: 'Computers/technology',
        71: 'Insurance company',
        72: 'Music award',
        73: 'Recreation/sports website',
        74: 'Reference website',
        75: 'Business/economy website',
        76: 'Bar',
        77: 'Album',
        78: 'Games/toys',
        79: 'Camera/photo',
        80: 'Book',
        81: 'Producer',
        82: 'Landmark',
        83: 'Cause',
        84: 'Organization',
        85: 'Tv/movie award',
        86: 'Hotel',
        87: 'Health/medical/pharmaceuticals',
        88: 'Transportation',
        89: 'Local/travel website',
        90: 'Musical instrument',
        91: 'Radio station',
        92: 'Other',
        93: 'Computers',
        94: 'Phone/tablet',
        95: 'Coach',
        96: 'Tools/equipment',
        97: 'Internet/software',
        98: 'Bank/financial institution',
        99: 'Society/culture website',
        100: 'Small business',
        101: 'News personality',
        102: 'Teens/kids website',
        103: 'Government official',
        104: 'Photographer',
        105: 'Spas/beauty/personal care',
        106: 'Video game'
    }
    
    # Remplacer les nombres par les noms de catégories dans la colonne appropriée
    df['category'] = df['category'].map(cat_mapping)
    # Sélectionnez les colonnes numériques
    numeric_columns = df.select_dtypes(include=[np.number])
    
    # Convertissez les colonnes en entiers (si nécessaire)
    numeric_columns = numeric_columns.astype(int)
    
    # Arrondissez les colonnes numériques à l'unité
    rounded_data = numeric_columns.round()
    
    # Remplacez les colonnes arrondies dans le jeu de données d'origine
    df[numeric_columns.columns] = rounded_data
    
    context = {
        'full_table1':df
     }
    
    return render(request, 'about2.html',context)
    
def about(request):
    page_features = ['likes','visitors','daily_interest','category']
    derived = ['derived_{}'.format(i) for i in range(1,26)]
    essential_features = ['C_{}'.format(i) for i in range(1,6)]
    base_features = ['base_time','post_length','share_count','promotion_status','H_hrs']
    
    weekday = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    base_DT = ['BDT_Sun', 'BDT_Mon', 'BDT_Tue', 'BDT_Wed', 'BDT_Thu', 'BDT_Fri', 'BDT_Sat']
    target = ['No_of_Comments_in_H_hours']
    cols = page_features+derived+essential_features+base_features+weekday+base_DT+target
    
    full_table = pd.read_csv('C:/Users/louis/Downloads/facebook+comment+volume+dataset/Dataset/Testing/Features_TestSet.csv',names = cols, header = None,nrows=500)
    context={
        'full_table':full_table
        }
    return render(request, 'about.html',context)

def interact(request):
    choice1 = None
    choice2 = None
    plothtml="Your model can be long te be started please wait a minute"
    plothtml2=""
    if request.method == 'POST':
        choice1 = request.POST.get('choice1', '')
        choice2 = request.POST.get('choice2', '')
        
        if choice1=='SKB':
            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import f_regression, mutual_info_regression
            from sklearn.feature_selection import chi2
            import plotly.express as px
            data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
            category_mapping = {
                1: 'Product/service',
                2: 'Public figure',
                3: 'Retail and consumer merchandise',
                4: 'Athlete',
                5: 'Education website',
                6: 'Arts/entertainment/nightlife',
                7: 'Aerospace/defense',
                8: 'Actor/director',
                9: 'Professional sports team',
                10: 'Travel/leisure',
                11: 'Arts/humanities website',
                12: 'Food/beverages',
                13: 'Record label',
                14: 'Movie',
                15: 'Song',
                16: 'Community',
                17: 'Company',
                18: 'Artist',
                19: 'Non-governmental organization (ngo)',
                20: 'Media/news/publishing',
                21: 'Cars',
                22: 'Clothing',
                23: 'Local business',
                24: 'Musician/band',
                25: 'Politician',
                26: 'News/media website',
                27: 'Education',
                28: 'Author',
                29: 'Sports event',
                30: 'Restaurant/cafe',
                31: 'School sports team',
                32: 'University',
                33: 'Tv show',
                34: 'Website',
                35: 'Outdoor gear/sporting goods',
                36: 'Political party',
                37: 'Sports league',
                38: 'Entertainer',
                39: 'Church/religious organization',
                40: 'Non-profit organization',
                41: 'Automobiles and parts',
                42: 'Tv channel',
                43: 'Telecommunication',
                44: 'Entertainment website',
                45: 'Shopping/retail',
                46: 'Personal blog',
                47: 'App page',
                48: 'Vitamins/supplements',
                49: 'Professional services',
                50: 'Movie theater',
                51: 'Software',
                52: 'Magazine',
                53: 'Electronics',
                54: 'School',
                55: 'Just for fun',
                56: 'Club',
                57: 'Comedian',
                58: 'Sports venue',
                59: 'Sports/recreation/activities',
                60: 'Publisher',
                61: 'Tv network',
                62: 'Health/medical/pharmacy',
                63: 'Studio',
                64: 'Home decor',
                65: 'Jewelry/watches',
                66: 'Writer',
                67: 'Health/beauty',
                68: 'Music video',
                69: 'Appliances',
                70: 'Computers/technology',
                71: 'Insurance company',
                72: 'Music award',
                73: 'Recreation/sports website',
                74: 'Reference website',
                75: 'Business/economy website',
                76: 'Bar',
                77: 'Album',
                78: 'Games/toys',
                79: 'Camera/photo',
                80: 'Book',
                81: 'Producer',
                82: 'Landmark',
                83: 'Cause',
                84: 'Organization',
                85: 'Tv/movie award',
                86: 'Hotel',
                87: 'Health/medical/pharmaceuticals',
                88: 'Transportation',
                89: 'Local/travel website',
                90: 'Musical instrument',
                91: 'Radio station',
                92: 'Other',
                93: 'Computers',
                94: 'Phone/tablet',
                95: 'Coach',
                96: 'Tools/equipment',
                97: 'Internet/software',
                98: 'Bank/financial institution',
                99: 'Society/culture website',
                100: 'Small business',
                101: 'News personality',
                102: 'Teens/kids website',
                103: 'Government official',
                104: 'Photographer',
                105: 'Spas/beauty/personal care',
                106: 'Video game'
            }
            data['category'] = data['category'].map(category_mapping)
            encoded_data = pd.get_dummies(data, columns=['category'])
            
            from sklearn.preprocessing import MinMaxScaler
            
            # Sélectionnez toutes les colonnes sauf 'category'
            columns_to_normalize = [col for col in encoded_data.columns if col != 'category']
            
            # Créez un sous-ensemble de données avec les colonnes à normaliser
            data_to_normalize = encoded_data[columns_to_normalize]
            
            # Initialisez le scaler
            scaler = MinMaxScaler()
            
            # Normalisez les colonnes sélectionnées
            normalized_data = scaler.fit_transform(data_to_normalize)
            
            # Remplacez les colonnes normalisées dans le jeu de données d'origine
            encoded_data[columns_to_normalize] = normalized_data
            X= encoded_data.iloc[:,:52]   #all features] #independent columns
            Y = encoded_data.iloc[:,-1] # pick last column for the target feature
            
            from sklearn.preprocessing import MinMaxScaler

            # Scale the features to ensure non-negativity
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            bestfeatures = SelectKBest(score_func=chi2, k=10)
            fit = bestfeatures.fit(X_scaled, Y)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(X.columns)
            scores = pd.concat([dfcolumns, dfscores], axis=1)
            scores.columns = ['specs', 'score']
            top_features = scores.nlargest(int(choice2), 'score')
            a="Top "+choice2+" Features"
            fig = px.bar(top_features, x='score', y='specs', orientation='h', title=a)
            plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
            plothtml2="There are the best features to maximize the chance to have comments on your post according to the model you used"
        elif choice1=='ETC':
            data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
            data = data.drop(['promotion_status'], axis=1)
            X= data.iloc[:,:52]   #all features] #independent columns
            y = data.iloc[:,-1] # pick last column for the target feature
            
            from sklearn.ensemble import ExtraTreesClassifier
            import numpy as np
            import matplotlib.pyplot as plt
            import plotly.graph_objs as go
            model = ExtraTreesClassifier()
            model.fit(X,y)
            
            feat_importances = pd.Series(model.feature_importances_, index=X.columns)
            top_features = feat_importances.nlargest(int(choice2))
            
            fig = go.Figure(data=[go.Bar(y=top_features.index, x=top_features.values, orientation='h')])
            
            fig.update_layout(title="Top 10 Feature Importances with ExtraTreesClassifier",
            xaxis_title="Importance",
            yaxis_title="Features")
            plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
            plothtml2="There are the best features to maximize the chance to have comments on your post according to the model you used"
        elif choice1=='LR':
            data =pd.read_csv("Features_Variant_1.csv",names = cols, header = None)
            category_mapping = {
                1: 'Product/service',
                2: 'Public figure',
                3: 'Retail and consumer merchandise',
                4: 'Athlete',
                5: 'Education website',
                6: 'Arts/entertainment/nightlife',
                7: 'Aerospace/defense',
                8: 'Actor/director',
                9: 'Professional sports team',
                10: 'Travel/leisure',
                11: 'Arts/humanities website',
                12: 'Food/beverages',
                13: 'Record label',
                14: 'Movie',
                15: 'Song',
                16: 'Community',
                17: 'Company',
                18: 'Artist',
                19: 'Non-governmental organization (ngo)',
                20: 'Media/news/publishing',
                21: 'Cars',
                22: 'Clothing',
                23: 'Local business',
                24: 'Musician/band',
                25: 'Politician',
                26: 'News/media website',
                27: 'Education',
                28: 'Author',
                29: 'Sports event',
                30: 'Restaurant/cafe',
                31: 'School sports team',
                32: 'University',
                33: 'Tv show',
                34: 'Website',
                35: 'Outdoor gear/sporting goods',
                36: 'Political party',
                37: 'Sports league',
                38: 'Entertainer',
                39: 'Church/religious organization',
                40: 'Non-profit organization',
                41: 'Automobiles and parts',
                42: 'Tv channel',
                43: 'Telecommunication',
                44: 'Entertainment website',
                45: 'Shopping/retail',
                46: 'Personal blog',
                47: 'App page',
                48: 'Vitamins/supplements',
                49: 'Professional services',
                50: 'Movie theater',
                51: 'Software',
                52: 'Magazine',
                53: 'Electronics',
                54: 'School',
                55: 'Just for fun',
                56: 'Club',
                57: 'Comedian',
                58: 'Sports venue',
                59: 'Sports/recreation/activities',
                60: 'Publisher',
                61: 'Tv network',
                62: 'Health/medical/pharmacy',
                63: 'Studio',
                64: 'Home decor',
                65: 'Jewelry/watches',
                66: 'Writer',
                67: 'Health/beauty',
                68: 'Music video',
                69: 'Appliances',
                70: 'Computers/technology',
                71: 'Insurance company',
                72: 'Music award',
                73: 'Recreation/sports website',
                74: 'Reference website',
                75: 'Business/economy website',
                76: 'Bar',
                77: 'Album',
                78: 'Games/toys',
                79: 'Camera/photo',
                80: 'Book',
                81: 'Producer',
                82: 'Landmark',
                83: 'Cause',
                84: 'Organization',
                85: 'Tv/movie award',
                86: 'Hotel',
                87: 'Health/medical/pharmaceuticals',
                88: 'Transportation',
                89: 'Local/travel website',
                90: 'Musical instrument',
                91: 'Radio station',
                92: 'Other',
                93: 'Computers',
                94: 'Phone/tablet',
                95: 'Coach',
                96: 'Tools/equipment',
                97: 'Internet/software',
                98: 'Bank/financial institution',
                99: 'Society/culture website',
                100: 'Small business',
                101: 'News personality',
                102: 'Teens/kids website',
                103: 'Government official',
                104: 'Photographer',
                105: 'Spas/beauty/personal care',
                106: 'Video game'
            }
            data['category'] = data['category'].map(category_mapping)
            encoded_data = pd.get_dummies(data, columns=['category'])
            
            from sklearn.preprocessing import MinMaxScaler
            
            # Sélectionnez toutes les colonnes sauf 'category'
            columns_to_normalize = [col for col in encoded_data.columns if col != 'category']
            
            # Créez un sous-ensemble de données avec les colonnes à normaliser
            data_to_normalize = encoded_data[columns_to_normalize]
            
            # Initialisez le scaler
            scaler = MinMaxScaler()
            
            # Normalisez les colonnes sélectionnées
            normalized_data = scaler.fit_transform(data_to_normalize)
            
            # Remplacez les colonnes normalisées dans le jeu de données d'origine
            encoded_data[columns_to_normalize] = normalized_data
            
            from sklearn.feature_selection import RFE
            from sklearn.inspection import permutation_importance
            from sklearn.linear_model import LinearRegression
            import plotly.express as px
            # Initialiser le modèle de régression linéaire
            model = LinearRegression()
            
            rfe = RFE(model, n_features_to_select=int(choice2))  # Sélectionnez le nombre de caractéristiques souhaité
            
            X = encoded_data.drop('No_of_Comments_in_H_hours', axis=1)
            y = encoded_data['No_of_Comments_in_H_hours']
            
            # Appliquer la méthode de sélection de caractéristiques RFE
            rfe.fit(X, y)
            
            # Afficher les caractéristiques sélectionnées
            selected_features_lr = X.columns[rfe.support_]
            
            # Entraîner le modèle avec les caractéristiques sélectionnées
            model.fit(X[selected_features_lr], y)
            
            # Calculer l'importance des caractéristiques par permutation
            perm_importance = permutation_importance(model, X[selected_features_lr], y, n_repeats=30)
            
            # Afficher l'importance des caractéristiques
            sorted_idx = perm_importance.importances_mean.argsort()
            fig = px.bar(x=perm_importance.importances_mean[sorted_idx], y=selected_features_lr[sorted_idx], orientation='h')
            
            fig.update_layout(
            xaxis_title="Permutation Importance",
            yaxis_title="Features",
            title="Permutation Importance"
            )
            plothtml=fig.to_html(full_html=False,default_height=500,default_width=700)
            plothtml2="There are the best features to maximize the chance to have comments on your post according to the model you used"
        # Faites ici ce que vous souhaitez faire avec les choix
        # par exemple, enregistrez-les dans une base de données ou utilisez-les dans votre logique métier
        
    return render(request, 'interact.html', {'choice1': choice1, 'choice2': choice2,'plothtml':plothtml,'plothtml2':plothtml2 })
    

# def about3(request):
#     #full_table2018=pd.read_csv()
#     if (request.GET['model']=='test1'):
#         plothtml="emf;mefe;mf"
#         plothtml2=";oerfjpkflrf"
#     if(request.GET['model']=="test"):
#         plothtml="ca marche ou pas "
#         plothtml2=" oui !"
#     context={
#         'plothtml':plothtml,
#         'plothtml2':plothtml2
#         }
#     return render(request,'template1.html',context)











