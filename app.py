import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# Configuración de la página
st.set_page_config(page_title="PEN - Motor Bayesiano", layout="wide")

st.title("🛡️ Proyecto PEN: Predicción Estratégica Nacional")
st.subheader("Módulo I: Motor de Modelado Matemático y Bayesiano")

st.markdown("""
Este módulo implementa una **Regresión Bayesiana**. A diferencia de los modelos lineales estándar, 
este enfoque trata los parámetros como distribuciones de probabilidad, lo que permite cuantificar 
la **incertidumbre** en las predicciones de estabilidad o riesgo.
""")

# --- SECCIÓN DE DATOS ---
with st.sidebar:
    st.header("Configuración de Datos")
    data_source = st.selectbox("Fuente de Datos", ["Simulación de Estabilidad Financiera", "Cargar CSV Personalizado"])
    
    # Parámetros del modelo
    st.header("Parámetros del Algoritmo")
    alpha_init = st.slider("Precisión de la Prior (Alpha)", 1e-6, 1.0, 1e-6, format="%.6f")
    lambda_init = st.slider("Precisión del Ruido (Lambda)", 1e-6, 1.0, 1e-6, format="%.6f")

# Generación de datos sintéticos (Simulación de relevancia nacional)
def load_data():
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(100, 1), axis=0)
    # Simulamos una tendencia con ruido (ej. Índice de Confianza Económica)
    y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])
    return X, y

if data_source == "Simulación de Estabilidad Financiera":
    X, y = load_data()
    df = pd.DataFrame({"Tiempo/Variable": X.flatten(), "Indicador": y})
else:
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, 0].values.reshape(-1, 1)
        y = df.iloc[:, 1].values
    else:
        st.info("Esperando archivo... usando simulación por defecto.")
        X, y = load_data()

# --- MOTOR DE CÁLCULO (Álgebra Lineal y Bayes) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implementación de Bayesian Ridge Regression
# Matemáticamente: p(w|y,X) ∝ p(y|X,w)p(w)
model = BayesianRidge(alpha_1=alpha_init, lambda_1=lambda_init, compute_score=True)
model.fit(X_scaled, y)

# Predicción con desviación estándar (Incertidumbre)
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)
y_mean, y_std = model.predict(X_plot_scaled, return_std=True)

# --- VISUALIZACIÓN ---
st.header("Análisis de Tendencia y Pronóstico")

fig = go.Figure()

# Datos Históricos
fig.add_trace(go.Scatter(
    x=X.flatten(), y=y,
    mode='markers',
    name='Datos Históricos (Entrenamiento)',
    marker=dict(color='rgba(150, 150, 150, 0.5)')
))

# Línea de Predicción (Media)
fig.add_trace(go.Scatter(
    x=X_plot.flatten(), y=y_mean,
    mode='lines',
    name='Pronóstico del Motor',
    line=dict(color='#1f77b4', width=3)
))

# Intervalo de Confianza (Incertidumbre Bayesiana)
fig.add_trace(go.Scatter(
    x=np.concatenate([X_plot.flatten(), X_plot.flatten()[::-1]]),
    y=np.concatenate([y_mean - 1.96 * y_std, (y_mean + 1.96 * y_std)[::-1]]),
    fill='toself',
    fillcolor='rgba(31, 119, 180, 0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='Intervalo de Confianza (95%)'
))

fig.update_layout(
    title="Evolución del Indicador de Estabilidad",
    xaxis_title="Eje Temporal / Variable Predictora",
    yaxis_title="Valor del Indicador",
    template="plotly_white",
    hovermode="x"
)

st.plotly_chart(fig, use_container_width=True)

# --- PANEL DE DECISIÓN ESTRATÉGICA ---
col1, col2 = st.columns(2)

with col1:
    st.metric("Puntuación de Convergencia", f"{model.scores_[-1]:.2f}")
    st.write("**Nota Técnica:** La convergencia indica qué tan bien el modelo ha ajustado las prioris bayesianas a los datos observados.")

with col2:
    st.write("**Resumen de Pesos (Álgebra Lineal):**")
    weights = pd.DataFrame({"Atributo": ["Variable Primaria"], "Coeficiente": model.coef_})
    st.table(weights)

st.success("✅ Motor automatizado listo para la toma de decisiones.")
