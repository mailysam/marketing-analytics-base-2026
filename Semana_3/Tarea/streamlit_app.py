"""Aplicación Starbucks Rewards basada en streamlit_starter.py."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title='Starbucks Rewards | Uplift', page_icon='☕', layout='wide')
GREEN, ORANGE, GRAY = '#00704A', '#C4512F', '#9A988E'
FEATURES = ['recency_days', 'frequency', 'monetary', 'email_open_rate', 'tenure_days']
MODELOS = {
    'Regresión logística': LogisticRegression(max_iter=1000),
    'Árbol sin límite': DecisionTreeClassifier(random_state=42),
    'Árbol max_depth=2': DecisionTreeClassifier(max_depth=2, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1),
}

@st.cache_data
def load_data():
    ruta = Path(__file__).resolve().parent / 'data' / 'uplift_campaign.csv'
    if not ruta.exists():
        st.error('No se encontró data/uplift_campaign.csv. Agrega el CSV del laboratorio a la carpeta data/ del repositorio.')
        st.stop()
    return pd.read_csv(ruta)

@st.cache_data
def load_truth():
    ruta = Path(__file__).resolve().parent / 'uplift_campaign_FULL.csv'
    return pd.read_csv(ruta) if ruta.exists() else None

@st.cache_data(show_spinner='Entrenando modelos...')
def entrenar_t_learner(df):
    train, test = train_test_split(df, test_size=0.30, random_state=42, stratify=df['treatment_group'])
    train_t = train[train['treatment_group'] == 'treatment']
    train_c = train[train['treatment_group'] == 'control']
    test = test.copy()
    riesgo = LogisticRegression(max_iter=1000).fit(train[FEATURES], 1 - train['responded_60d'])
    test['prob_churn'] = riesgo.predict_proba(test[FEATURES])[:, 1]
    for nombre, base in MODELOS.items():
        modelo_tratado = clone(base).fit(train_t[FEATURES], train_t['responded_60d'])
        modelo_control = clone(base).fit(train_c[FEATURES], train_c['responded_60d'])
        test[nombre] = (modelo_tratado.predict_proba(test[FEATURES])[:, 1] -
                        modelo_control.predict_proba(test[FEATURES])[:, 1])
    return test

def valor_de_la_politica(test_df, seleccionados):
    sel = test_df.loc[seleccionados]
    tratado = sel.loc[sel['treatment_group'] == 'treatment', 'utilidad_neta_60d'].mean()
    control = sel.loc[sel['treatment_group'] == 'control', 'margin_60d'].mean()
    return tratado - control, len(sel)

def curva_qini(test_df, score_col, steps=40):
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden['treatment_group'] == 'treatment').values
    respondio = orden['responded_60d'].values
    acumulado_t = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_c = np.cumsum(np.where(~es_tratado, respondio, 0))
    nt = np.cumsum(es_tratado)
    nc = np.cumsum(~es_tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        tasa_t = acumulado_t[i-1] / nt[i-1] if nt[i-1] else 0
        tasa_c = acumulado_c[i-1] / nc[i-1] if nc[i-1] else 0
        xs.append(i/n)
        ys.append((tasa_t-tasa_c)*n)
    return np.array(xs), np.array(ys)

def coeficiente_qini(xs, ys, n):
    return (np.trapezoid(ys, xs) - np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs))/n

st.title('☕ Starbucks Rewards | Decisiones basadas en uplift')
st.caption('Campaña de cupón de $5 · 40,000 clientes · 60 días · Evaluación en muestra de prueba')
with st.sidebar:
    st.header('Explorar políticas')
    modelo = st.selectbox('Modelo T-learner', list(MODELOS), index=3)
    pct_contactar = st.slider('Porcentaje de clientes a contactar', 10, 100, 20, 10)
    st.caption('Se evalúa el 30% reservado como prueba, con semilla 42.')

df = load_data()
test = entrenar_t_learner(df)
resultados = []
for pct in range(10, 101, 10):
    idx = test.sort_values(modelo, ascending=False).head(int(len(test)*pct/100)).index
    valor_cliente, n = valor_de_la_politica(test, idx)
    resultados.append({'Porcentaje':pct, 'Clientes':n, 'Valor por cliente':round(valor_cliente,2), 'Valor total':round(valor_cliente*n,2)})
tabla = pd.DataFrame(resultados)
seleccion = tabla.loc[tabla['Porcentaje'] == pct_contactar].iloc[0]
optimo = tabla.loc[tabla['Valor total'].idxmax()]
xs, ys = curva_qini(test, modelo)
xr, yr = curva_qini(test, 'prob_churn')
qini = coeficiente_qini(xs, ys, len(test))
qini_r = coeficiente_qini(xr, yr, len(test))

c1,c2,c3,c4 = st.columns(4)
c1.metric('Clientes seleccionados (prueba)', f"{int(seleccion['Clientes']):,}")
c2.metric('Beneficio incremental estimado', f"${seleccion['Valor total']:,.2f}")
c3.metric('Qini del T-learner', f'{qini:+.3f}')
c4.metric('Mejor porcentaje evaluado', f"{int(optimo['Porcentaje'])}%")

st.subheader('Curva Qini')
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(xs, ys, color=GREEN, linewidth=3, label=f'{modelo} (Qini={qini:+.3f})')
ax.plot(xr, yr, color=ORANGE, linewidth=2, label=f'Riesgo/churn (Qini={qini_r:+.3f})')
ax.plot([0,1], [0,ys[-1]], color=GRAY, linestyle='--', label='Al azar')
ax.axvline(pct_contactar/100, color=GRAY, linestyle=':', alpha=.8, label=f'Cobertura seleccionada: {pct_contactar}%')
ax.set(xlabel='Proporción de la base de prueba', ylabel='Respuestas incrementales acumuladas')
ax.legend()
ax.grid(alpha=.15)
st.pyplot(fig)
plt.close(fig)

st.subheader('Tabla de políticas de contacto')
st.dataframe(tabla.style.apply(lambda row: ['background-color: #d9f0e3' if row['Porcentaje'] == pct_contactar else '' for _ in row], axis=1).format({'Valor por cliente':'${:,.2f}','Valor total':'${:,.2f}'}), use_container_width=True, hide_index=True)
st.success(f"Con {modelo}, la mejor cobertura entre las evaluadas es {int(optimo['Porcentaje'])}% y su beneficio incremental estimado es ${optimo['Valor total']:,.2f} en la muestra de prueba.")
st.caption('Beneficio estimado con la diferencia de medias entre tratados y control en cada segmento. No equivale a una ganancia garantizada en producción.')

st.subheader('Comparación de clasificadores dentro del T-learner')
truth = load_truth()
comparacion = []
for nombre in MODELOS:
    x, y = curva_qini(test, nombre)
    fila = {'Modelo': nombre, 'Qini': round(coeficiente_qini(x,y,len(test)),3)}
    if truth is not None and 'true_uplift_prob' in truth.columns:
        unido = test[['customer_id',nombre]].merge(truth[['customer_id','true_uplift_prob']],on='customer_id')
        fila['Correlación con uplift real'] = round(unido[nombre].corr(unido['true_uplift_prob']),3)
    comparacion.append(fila)
st.dataframe(pd.DataFrame(comparacion), use_container_width=True, hide_index=True)
if truth is None:
    st.info('Para mostrar la correlación con el uplift verdadero, agrega uplift_campaign_FULL.csv a la raíz del repositorio. La app funciona sin ese archivo.')
