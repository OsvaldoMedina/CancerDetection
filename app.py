"""CancerDetection — laboratorio universitario. Run: python -m streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from src.cancer_detection import evaluate_model
from src.inference import (ROOT, FEATURES, CLASSES, load_assets, label, predict,
                           validate_frame, read_csv_bytes, batch_predict,
                           range_warnings, threshold_classes, classification_metrics)

st.set_page_config(page_title="CancerDetection | Laboratorio", page_icon="🔬", layout="wide")
st.markdown("""<style>
.block-container {max-width:1250px;padding-top:2rem;}
[data-testid="stMetric"] {border:1px solid #a9bac9;border-radius:12px;padding:14px;}
</style>""", unsafe_allow_html=True)
st.title("CancerDetection — Laboratorio interactivo de clasificación")
st.caption("Laboratorio interactivo de clasificación · Ciencia de datos")
st.warning("Uso académico. Esta predicción no constituye un diagnóstico médico")


@st.cache_resource
def assets():
    return load_assets()


try:
    model, train, test, report = assets()
except Exception as exc:
    st.error(f"No se puede iniciar la inferencia: {exc}")
    st.info("Revisa los archivos locales y el entorno indicado en README.md. No se generan predicciones simuladas.")
    st.stop()

page = st.sidebar.radio("Navegación", ["Inicio", "Clasificación individual", "Carga CSV", "Laboratorio", "Evaluación"])
st.sidebar.info("Clases del proyecto: 0 = benigno · 1 = maligno. No introduzcas nombres, expedientes ni datos identificables.")
st.sidebar.caption("Inferencia local · Sin servicios externos · Los CSV se procesan en memoria.")

METRIC_LABELS = {"accuracy": "Exactitud", "sensitivity_recall": "Sensibilidad", "specificity": "Especificidad",
                 "precision": "Precision (VPP)", "f1": "F1", "roc_auc": "AUC ROC"}


def metrics_cards(values):
    cols = st.columns(3)
    for i, (key, text) in enumerate(METRIC_LABELS.items()):
        val = values.get(key)
        cols[i % 3].metric(text, "No definida" if val is None else f"{val:.4f}")


def confusion(values, key):
    fig = px.imshow(values["confusion_matrix"], x=["Benigno (0)", "Maligno (1)"],
                    y=["Benigno (0)", "Maligno (1)"], text_auto=True,
                    labels={"x": "Clase predicha", "y": "Clase observada", "color": "Registros"},
                    color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(title="Matriz de confusión", height=370)
    st.plotly_chart(fig, key=key, width="stretch")


def result(frame, key, threshold=None):
    classes, p = predict(model, frame)
    selected = classes[0] if threshold is None else threshold_classes(p, threshold)[0]
    st.subheader(f"Clase predicha: {CLASSES[selected]}")
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad estimada: benigno (0)", f"{1-p[0]:.2%}")
    c2.metric("Probabilidad estimada: maligno (1)", f"{p[0]:.2%}")
    fig = px.bar(x=["Benigno (0)", "Maligno (1)"], y=[1-p[0], p[0]],
                 labels={"x": "Clase", "y": "Probabilidad estimada"}, text=[f"{1-p[0]:.2%}", f"{p[0]:.2%}"])
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, key=key, width="stretch")
    st.caption("Probabilidades del modelo, no riesgo clínico validado. El rendimiento en prueba no garantiza el resultado de este registro.")
    warnings = range_warnings(frame, train)
    if not warnings.empty:
        st.warning("Hay valores fuera de los mínimos y máximos de train.csv. Son referencias estadísticas, no límites clínicos.")
        st.dataframe(warnings, hide_index=True)
    st.dataframe(frame[list(FEATURES)].T.rename(columns={frame.index[0]: "Valor utilizado"}), width="stretch")


def fill(prefix, row=None, description="Captura manual; completa las 30 variables."):
    for f in FEATURES:
        st.session_state[f"{prefix}_{f}"] = None if row is None else float(row[f])
    st.session_state[f"{prefix}_origin"] = description
    st.session_state.pop(f"{prefix}_result", None)


def editor(prefix):
    values = {}
    for title, subset in (("Valores medios · mean", FEATURES[:10]),
                          ("Errores estándar · error", FEATURES[10:20]),
                          ("Valores worst", FEATURES[20:])):
        st.subheader(title)
        cols = st.columns(2)
        for i, f in enumerate(subset):
            with cols[i % 2]:
                values[f] = st.number_input(label(f), value=None, format="%.8f", step=0.000001,
                                           key=f"{prefix}_{f}", help=f"Nombre original: {f}. Rango train: {train[f].min():.8g} a {train[f].max():.8g}.")
    return pd.DataFrame([values], columns=FEATURES)


def example_description(row):
    return f"Ejemplo de entrenamiento · source_index={int(row.source_index)} · etiqueta observada: {CLASSES[int(row.malignant)]}. No es una validación independiente."


if page == "Inicio":
    st.header("Aprender a clasificar, evaluar y cuestionar")
    st.write("Introduce mediciones, observa una inferencia real y explora cómo cambian los resultados al modificar variables o el umbral. El objetivo es comprender un flujo de aprendizaje supervisado y sus limitaciones.")
    a, b, c = st.columns(3)
    a.metric("Registros de entrenamiento", len(train))
    b.metric("Registros de prueba", len(test))
    c.metric("Variables predictoras", len(FEATURES))
    st.write(f"**Modelo seleccionado:** {report['selected_model']} (regresión logística). Selección mediante AUC ROC de validación cruzada en cinco particiones de entrenamiento.")
    st.subheader("Métricas originales · reports/metrics.json")
    metrics_cards(report["test_metrics"])
    st.info("Exactitud: proporción total de aciertos. Precision o valor predictivo positivo (VPP): proporción de malignos observados entre los clasificados como malignos. Son medidas diferentes.")
    st.subheader("Dataset y variables")
    st.write("Breast Cancer Wisconsin Diagnostic: 569 registros. Las variables describen núcleos celulares en imágenes digitalizadas de aspirados con aguja fina de masas mamarias. Son 10 características resumidas por media, error estándar y worst (media de los tres valores mayores): 30 entradas en total.")
    st.write("Radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal. Las unidades físicas no se especifican en la descripción distribuida; no se asignan unidades inventadas.")
    st.dataframe(pd.DataFrame({"Nombre original": FEATURES, "Etiqueta": [label(f) for f in FEATURES],
                               "Mínimo train": train[list(FEATURES)].min().values,
                               "Máximo train": train[list(FEATURES)].max().values}), hide_index=True)
    with st.expander("Definiciones verificadas y fuente"):
        st.write("Radio: media de distancias del centro al perímetro. Textura: desviación estándar de niveles de gris. Suavidad: variación local de longitudes radiales. Compacidad: perímetro²/área − 1. Concavidad: severidad de porciones cóncavas del contorno. Puntos cóncavos: número de porciones cóncavas. Dimensión fractal: aproximación de costa − 1.")
        st.markdown("Fuente: descripción incluida en `sklearn.datasets.load_breast_cancer().DESCR` y [documentación de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).")
    st.subheader("Del entrenamiento a la evaluación")
    st.write("1. Entrenamiento: se ajustaron y compararon modelos usando únicamente train.\n\n2. Estandarización: el StandardScaler del pipeline conserva las medias y escalas aprendidas; no se vuelve a ajustar al capturar registros.\n\n3. Predicción: las 30 entradas ordenadas atraviesan el pipeline completo.\n\n4. Evaluación: se comparan predicciones y etiquetas de test. Aquí benigno = 0 y maligno = 1; scikit-learn originalmente usa la codificación inversa.")

elif page == "Clasificación individual":
    st.header("Clasificación individual")
    st.write("Completa las 30 mediciones con punto decimal. Los campos vacíos no se sustituyen. Los rangos de ayuda se calculan solo con entrenamiento.")
    prefix = "individual"
    if not any(f"{prefix}_{f}" in st.session_state for f in FEATURES):
        fill(prefix)
    cols = st.columns(3)
    for cls in (0, 1):
        row = train[train.malignant == cls].iloc[0]
        cols[cls].button(f"Cargar ejemplo {CLASSES[cls]}", on_click=fill,
                         args=(prefix, row, example_description(row)))
    cols[2].button("Restablecer formulario", on_click=fill, args=(prefix,))
    st.info(st.session_state.get(f"{prefix}_origin", "Captura manual; completa las 30 variables."))
    with st.form("individual_form"):
        frame = editor(prefix)
        submitted = st.form_submit_button("Clasificar registro", type="primary")
    if submitted:
        st.session_state.pop(f"{prefix}_result", None)
        _, errors, _ = validate_frame(frame)
        if not errors.empty:
            st.error("Completa o corrige las mediciones indicadas.")
            st.dataframe(errors, hide_index=True)
        else:
            st.session_state[f"{prefix}_result"] = frame
    if f"{prefix}_result" in st.session_state:
        st.caption("Resultado de la última clasificación enviada; vuelve a clasificar después de editar.")
        try:
            result(st.session_state[f"{prefix}_result"], "individual_result")
        except (ValueError, FloatingPointError) as exc:
            st.error(f"No se pudo calcular el resultado: {exc}. Revisa la magnitud de las entradas.")

elif page == "Carga CSV":
    st.header("Clasificación por CSV")
    st.write("UTF-8, separador coma, punto decimal y las 30 columnas exactas. No incluyas identificadores personales. Las filas inválidas se descargan con su error y sin predicción.")
    st.download_button("Descargar plantilla CSV", pd.DataFrame(columns=FEATURES).to_csv(index=False).encode("utf-8"), "plantilla_30_variables.csv", "text/csv")
    uploaded = st.file_uploader("Seleccionar CSV de mediciones", type=["csv"])
    if uploaded:
        try:
            frame = read_csv_bytes(uploaded.getvalue())
            metadata = [c for c in ("source_index", "malignant") if c in frame]
            if metadata:
                st.info(f"Columnas separadas de los predictores: {', '.join(metadata)}. malignant se conserva solo como etiqueta observada opcional (0=benigno, 1=maligno). source_index solo se conserva como referencia.")
            results, errors = batch_predict(model, frame, train)
            if not errors.empty:
                st.error("Hay filas inválidas que no fueron clasificadas. Fila 1 significa el primer registro después del encabezado.")
                st.dataframe(errors, hide_index=True)
            st.dataframe(results, hide_index=True)
            st.caption("Fuera del rango de entrenamiento indica una advertencia estadística, no un límite clínico. Las probabilidades no son riesgo clínico validado.")
            st.download_button("Descargar resultados", results.to_csv(index=False).encode("utf-8-sig"), "resultados.csv", "text/csv")
            if "malignant" in frame:
                observed = pd.to_numeric(frame.malignant, errors="coerce")
                mask = observed.isin([0, 1]) & results.probabilidad_maligno.notna()
                st.caption(f"Evaluación opcional: {int(mask.sum())} filas con mediciones y etiqueta válidas; {int((~mask).sum())} excluidas.")
                if (~observed.isin([0, 1])).any():
                    st.warning("Hay etiquetas malignant inválidas; consulta estado_etiqueta en los resultados.")
                if mask.any():
                    yp = results.loc[mask, "clase_predicha"].map({"benigno": 0, "maligno": 1})
                    values = classification_metrics(observed[mask].astype(int), yp, results.loc[mask, "probabilidad_maligno"])
                    metrics_cards(values)
                    confusion(values, "csv_confusion")
                    st.caption("Estas métricas describen únicamente las filas cargadas con etiqueta válida; no demuestran validación independiente ni validez clínica.")
        except Exception as exc:
            st.error(f"No se pudo procesar el CSV: {exc}")

elif page == "Laboratorio":
    st.header("Experimentar con variables y umbrales")
    st.warning("Modificar entradas muestra el comportamiento del modelo. No demuestra causalidad y puede producir combinaciones poco realistas.")
    threshold = st.slider("Umbral para clasificar como maligno", 0.0, 1.0, 0.50, 0.01)
    st.code("clase = 1 si P(clase 1) >= umbral; de lo contrario, clase = 0", language=None)
    st.subheader("Métricas exploratorias del umbral · test.csv")
    _, probabilities = predict(model, test)
    values = classification_metrics(test.malignant, threshold_classes(probabilities, threshold), probabilities)
    metrics_cards(values)
    confusion(values, "threshold_confusion")
    st.info("Explorar umbrales con test.csv no equivale a validarlos. Una selección formal debe realizarse con entrenamiento o validación; reutilizar test para elegir introduce sesgo. Las métricas originales se muestran por separado en Inicio y Evaluación. AUC utiliza probabilidades y no cambia al mover este umbral.")
    options = list(range(len(train)))
    selected = st.selectbox("Seleccionar ejemplo de entrenamiento", options,
                            format_func=lambda i: f"source_index={int(train.iloc[i].source_index)} · {CLASSES[int(train.iloc[i].malignant)]}")
    original = train.iloc[[selected]][list(FEATURES)].reset_index(drop=True)
    prefix = "lab"
    if st.session_state.get("lab_selected") != selected or any(f"{prefix}_{f}" not in st.session_state for f in FEATURES):
        fill(prefix, train.iloc[selected], example_description(train.iloc[selected]))
        st.session_state.lab_selected = selected
    st.button("Restablecer ejemplo original", on_click=fill,
              args=(prefix, train.iloc[selected], example_description(train.iloc[selected])))
    st.info(example_description(train.iloc[selected]))
    with st.form("lab_form"):
        changed = editor(prefix)
        submitted = st.form_submit_button("Comparar cambios", type="primary")
    if submitted:
        st.session_state.pop("lab_result", None)
        _, errors, _ = validate_frame(changed)
        if not errors.empty:
            st.error("Corrige las entradas antes de comparar.")
            st.dataframe(errors, hide_index=True)
        else:
            st.session_state.lab_result = changed
    current = st.session_state.get("lab_result", original)
    try:
        _, before = predict(model, original)
        _, after = predict(model, current)
        comparison = pd.DataFrame({"Escenario": ["Original", "Modificado"],
                                   "P(benigno)": [1-before[0], 1-after[0]],
                                   "P(maligno)": [before[0], after[0]],
                                   "Clase con umbral": [CLASSES[c] for c in threshold_classes([before[0], after[0]], threshold)]})
        st.dataframe(comparison, hide_index=True)
        st.plotly_chart(px.bar(comparison, x="Escenario", y=["P(benigno)", "P(maligno)"], barmode="group", range_y=[0, 1]), width="stretch")
        st.caption("Comparación de la última edición enviada mediante Comparar cambios.")
        st.dataframe(pd.DataFrame({"Variable": FEATURES, "Original": original.iloc[0].values,
                                   "Modificado": current.iloc[0].values,
                                   "Diferencia": current.iloc[0].values-original.iloc[0].values}), hide_index=True)
        result(current, "lab_prediction", threshold)
    except (ValueError, FloatingPointError) as exc:
        st.error(f"No se pudo comparar: {exc}. Revisa las magnitudes introducidas.")

elif page == "Evaluación":
    st.header("Evaluación y límites del benchmark")
    st.subheader("Métricas originales · reports/metrics.json")
    metrics_cards(report["test_metrics"])
    calculated, predicted, probabilities = evaluate_model(model, test[list(FEATURES)], test.malignant)
    differences = [k for k in METRIC_LABELS if not np.isclose(calculated[k], report["test_metrics"][k], atol=1e-8, rtol=1e-6)]
    if calculated["confusion_matrix"] != report["test_metrics"]["confusion_matrix"]:
        differences.append("confusion_matrix")
    if differences:
        st.error(f"Discrepancia al recalcular: {', '.join(differences)}. Los reportes originales no se han modificado.")
    else:
        st.success("Las métricas recalculadas del pipeline coinciden con el reporte (tolerancia absoluta 1e-8, relativa 1e-6).")
    st.subheader("Evaluación recalculada · decisión original del pipeline")
    metrics_cards(calculated)
    confusion(calculated, "evaluation_confusion")
    fpr, tpr, _ = roc_curve(test.malignant, probabilities, pos_label=1)
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Modelo · AUC={calculated['roc_auc']:.4f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Referencia aleatoria", line={"dash": "dash"}))
    fig.update_layout(title="Curva ROC · maligno es la clase positiva", xaxis_title="Tasa de falsos positivos (1 − especificidad)", yaxis_title="Sensibilidad", xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig, width="stretch")
    st.write("Falso positivo: registro benigno observado que el modelo clasifica como maligno. Falso negativo: registro maligno observado que clasifica como benigno. Sensibilidad = VP/(VP+FN); especificidad = VN/(VN+FP); precision (VPP) = VP/(VP+FP); F1 combina precision y sensibilidad. Si un denominador es cero, estas métricas se muestran como 0 por convención.")
    st.subheader("Comparación original · validación cruzada solo en entrenamiento")
    comparison = pd.read_csv(ROOT / "reports/model_comparison.csv")
    st.dataframe(comparison.rename(columns={"model": "Modelo", "mean_roc_auc": "AUC ROC media", "std_roc_auc": "AUC ROC desviación estándar", "mean_precision": "VPP medio", "mean_recall": "Sensibilidad media", "mean_f1": "F1 medio"}), hide_index=True)
    st.caption("mean = media; std = desviación estándar entre particiones. average_precision es el resumen de la curva precision–recall, diferente de precision (VPP). Los nombres de modelos se conservan como en el reporte.")
    st.subheader("Limitaciones")
    st.write("Este dataset pequeño e histórico no representa todas las poblaciones, dispositivos, instituciones ni protocolos actuales. La partición aleatoria no prueba generalización geográfica o temporal. No se ha demostrado calibración clínica ni desempeño por grupos demográficos. Un buen AUC o exactitud no garantiza cada resultado individual. Esta aplicación no calcula explicaciones causales ni atribuciones individuales.")
    with st.expander("Ficha original del modelo (en inglés)"):
        st.markdown((ROOT / "reports/model_card.md").read_text(encoding="utf-8"))
