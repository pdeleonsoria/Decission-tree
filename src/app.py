from utils import db_connect
engine = db_connect()

# your code here
df=pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")


num_rows, num_columns = df.shape
print(f"Número de filas: {num_rows}")
print(f"Número de columnas: {num_columns}")
df.head()

Nas= df.isna().sum()
Nulls= df.isnull().sum()
Rep= df.nunique()

print(f"Número de nas: {Nas}")
print(f"Número de nulls: {Nulls}")
print(f"Número de valores unicos: {Rep}")

df =df.drop_duplicates().reset_index(drop = True)

plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(df, 'Outcome', color=['green', 'red'])
plt.xticks(rotation=45)
plt.show()

#TRAIN Y TEST

X= df.drop("Outcome", axis=1)
y= df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

selec_model = SelectKBest(k = 7)
selec_model.fit(X_train, y_train)
modelo = selec_model.get_support()

X_train_sel = pd.DataFrame(selec_model.transform(X_train), columns = X_train.columns.values[modelo])
X_test_sel = pd.DataFrame(selec_model.transform(X_test), columns = X_test.columns.values[modelo])

X_train_sel["Outcome"] = list(y_train)
X_test_sel["Outcome"] = list(y_test)

X_train_sel.to_csv("../data/processed/clean_train.csv", index= False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index= False) 

total_datos = pd.concat([X_train_sel, X_test_sel])
total_datos.head()

#He utilizado la k = int(len(X_train.columns) * 0.3) pero me salen solo 3 columnas, por lo que me parecen muy pocas asi que he cambiado al 0.7 y me salen 6 predictores

plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(X_train_sel, 'Outcome', color=['green', 'red'])  
plt.show()

plt.figure(figsize=(12, 6))
pd.plotting.parallel_coordinates(X_test_sel, 'Outcome', color=['green', 'red']) 
plt.show()

#Por alguna razón se han invertido los significados de 0 y 1, en el df 0 era positivo en diabetes y aqui es el 1.

X_train_sel.head()

X_test_sel.head()

#ARBOL DE DECISIÓN
train_data = pd.read_csv("../data/processed/clean_train.csv")
test_data = pd.read_csv("../data/processed/clean_test.csv")

dtc= DecisionTreeClassifier

complex_tree = dtc(max_depth=100, min_samples_leaf=1, random_state=42).fit(X_train, y_train)
simple_tree = dtc(max_depth=3, min_samples_leaf=20, max_features=X_train.shape[1]//2, random_state=42).fit(X_train, y_train)

y_pred_train_complex = complex_tree.predict(X_train)
y_pred_test_complex = complex_tree.predict(X_test)

y_pred_train_simple = simple_tree.predict(X_train)
y_pred_test_simple = simple_tree.predict(X_test)


plt.figure(figsize=(10,6))
plot_tree(simple_tree, feature_names=list(X_train.columns), class_names=['N', 'P'], filled=True)

plt.figure(figsize=(10,6))
plot_tree(complex_tree, feature_names=list(X_train.columns), class_names=['N', 'P'], filled=True)

#METRICAS PARA SABER QUE ÁRBOL DE DECISIÓN ES EL MEJOR (PODRÍA HABERLO HECHO ANTES PERO QUERÍA VER LOS DOS)

metricas = {"Exactitud": accuracy_score,"Precisión": precision_score, "Sensibilidad": recall_score, "F1": f1_score,"AUC": roc_auc_score}

def get_metrics(y_train, y_test, y_pred_train, y_pred_test):

    results = {"Train": [], "Test": [], "Diferencia": []}
    
    
    for metric_name, metric_func in metricas.items():
        train_value = metric_func(y_train, y_pred_train)
        test_value = metric_func(y_test, y_pred_test)
        results["Train"].append(train_value)
        results["Test"].append(test_value)
        results["Diferencia"].append(train_value - test_value)

    metricas_df = pd.DataFrame(results, index=metricas.keys())
    
    return metricas_df


metricas_df_simple = get_metrics(y_train, y_test, y_pred_train_simple, y_pred_test_simple)
metricas_df_complex = get_metrics(y_train, y_test, y_pred_train_complex, y_pred_test_complex)

print("Las métricas del modelo simple son")
print(metricas_df_simple)
print("Las métricas del modelo complejo son:")
print(metricas_df_complex)


#OPTIMIZAR EL ARBOL 
param_grid = {"criterion": ["gini", "entropy"],"max_depth": [3, 5, 10, None],"min_samples_split": [2, 5, 10],"min_samples_leaf": [1, 2, 4]}


grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=10)
grid_search.fit(X_train, y_train)

print("Mejores parámetros:", grid_search.best_params_)

#COMPROBAR PARÁMETROS OPTIMIZADOS 

arbol_opti= DecisionTreeClassifier(criterion = "entropy", max_depth = 5, min_samples_leaf = 4, min_samples_split = 2, random_state = 42)
arbol_opti.fit(X_train, y_train)

y_pred_train_opti = arbol_opti.predict(X_train)
y_pred_test_opti = arbol_opti.predict(X_test)

y_pred_opti = arbol_opti.predict(X_test)
metricas_df_opti = get_metrics(y_train, y_test, y_pred_train_opti, y_pred_test_opti)

print(metricas_df_opti)

#GUARDAR

dump(arbol_opti, open("../models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "wb"))


