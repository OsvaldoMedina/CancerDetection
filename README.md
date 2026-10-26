# CancerDetection — Laboratorio interactivo de clasificación

Aplicación educativa en español con Streamlit, inferencia real del pipeline local y gráficos Plotly. Funciona sin claves API ni servicios externos.

> Uso académico. Esta predicción no constituye un diagnóstico médico

## Instalación con Anaconda en Windows 11

1. Abre **Anaconda Prompt** y entra en la carpeta del proyecto (usa comillas si hay espacios):

   ```bat
   cd "C:\ruta\CancerDetection"
   ```

2. Crea y activa el entorno. `environment.yml` mantiene el nombre **CancerDetection**, usa Python 3.12 e instala las versiones de `requirements.txt` mediante pip:

   ```bat
   conda env create -f environment.yml
   conda activate CancerDetection
   ```

   Si el entorno ya existe:

   ```bat
   conda activate CancerDetection
   python -m pip install -r requirements.txt
   python -m pip check
   ```

3. Inicia la aplicación desde la raíz:

   ```bat
   python -m streamlit run app.py
   ```

4. Abre `http://localhost:8501`. Detén el servidor con **Ctrl+C**. El servidor se limita a la interfaz local y tiene desactivada la telemetría de uso. La instalación inicial requiere descargar paquetes; la aplicación funciona localmente después.

## VS Code

1. Con el entorno activado, ejecuta `code .`.
2. Instala las extensiones **Python** y **Jupyter** de Microsoft.
3. Ejecuta **Python: Select Interpreter** y selecciona el Python del entorno **CancerDetection**.
4. Abre una terminal nueva, comprueba `python --version` (3.12) y ejecuta `python -m streamlit run app.py`.
5. Para consultar el notebook original, selecciona su kernel **CancerDetection**. No necesitas ejecutarlo para usar la aplicación: ejecutar todas sus celdas puede regenerar y reemplazar los artefactos del experimento original.

También puedes usar Python 3.12 con `python -m venv .venv`, activar con `.venv\Scripts\activate` e instalar `python -m pip install -r requirements.txt`. Para verificar esta entrega se utilizó `.venv` con acceso a los paquetes científicos del entorno Anaconda existente.

## Navegación y actividad de clase (15–20 minutos)

- **Inicio:** objetivo, dataset, 455 registros de entrenamiento y 114 de prueba, 30 variables y métricas originales.
- **Clasificación individual:** completa las 30 variables o carga ejemplos benignos/malignos de entrenamiento; observa clase, probabilidades y entradas utilizadas. Restablecer deja los campos vacíos.
- **Carga CSV:** descarga la plantilla, agrega registros y carga el archivo. Consulta errores por fila/columna y descarga resultados. Las filas inválidas no se clasifican.
- **Laboratorio:** selecciona un registro, cambia variables, compara probabilidades antes/después y explora un umbral inicial de 0.50.
- **Evaluación:** comparación de modelos del reporte, matriz de confusión, ROC/AUC y comprobación de reproducibilidad.

Actividad:

1. En Clasificación individual, carga un ejemplo benigno. Anota su `source_index`, etiqueta observada y las dos probabilidades; pulsa **Clasificar registro**.
2. Busca ese mismo `source_index` en Laboratorio y anota la probabilidad inicial de clase 1.
3. Modifica una variable (por ejemplo `mean radius`), pulsa **Comparar cambios** y registra la diferencia. Cambia otra variable y repite. ¿La clase cambia? ¿La combinación sigue siendo plausible?
4. Restablece el ejemplo. Mueve el umbral de 0.50 a 0.20 y 0.80. Compara falsos positivos, falsos negativos, sensibilidad, especificidad y VPP en la matriz de test.
5. Explica por qué una asociación del modelo no demuestra causalidad y por qué elegir un umbral con test introduce sesgo. Distingue exactitud de precision/VPP y probabilidades del modelo de riesgo clínico validado.

## Contrato de datos e inferencia

`src/inference.py` centraliza esquema, validación, carga, rangos y evaluación por umbral. Reutiliza `positive_probabilities` y la evaluación del módulo original. `src/cancer_detection.py` conserva las funciones de entrenamiento; la extracción de probabilidades ahora busca explícitamente la clase 1 en `classes_`.

- **Benigno = 0, maligno = 1**: el proyecto invierte las etiquetas originales de scikit-learn. Se verifica el mapeo del reporte y las etiquetas de ambos CSV mediante `source_index` frente al dataset original incluido en scikit-learn.
- Las 30 variables canónicas se contrastan con los datos procesados y con `feature_names_in_` del modelo si existe. Se ordenan antes de inferir. `source_index` y `malignant` nunca son predictores.
- Solo se deserializa `models/best_model.joblib`. El artefacto contiene StandardScaler + LogisticRegression. Las incompatibilidades de versión de scikit-learn detienen la carga; se muestra el motivo y no se reentrena ni simula una respuesta.
- Los campos empiezan vacíos y no se imputan datos. Se rechazan faltantes, NaN, infinitos y textos. Se admiten decimales pequeños; la captura muestra ocho decimales.
- Los mínimos/máximos de referencia se calculan exclusivamente con train.csv. Estar fuera genera una advertencia, no un límite clínico. Entradas de magnitudes extremas pueden exceder la aritmética del modelo y se rechazan con un mensaje.
- La decisión individual/CSV utiliza `pipeline.predict`. El laboratorio utiliza explícitamente **P(clase 1) >= umbral**. Se localiza la posición de clase 1 en `classes_`, incluso si el orden fuese diferente.
- Los ejemplos precargados provienen de **train.csv** y muestran origen y etiqueta observada. No constituyen nueva validación. El laboratorio evalúa umbrales sobre el benchmark test existente; no los valida formalmente.

### CSV

UTF-8 (BOM opcional), separador coma y punto decimal. La plantilla descargable define los 30 nombres exactos; su orden de entrada puede variar. Se rechazan encabezados faltantes, duplicados o adicionales y registros con distinto número de campos. No se admiten filas vacías ni comentarios.

`source_index` y `malignant` son las únicas columnas adicionales permitidas. Se conservan separadas de los predictores. `malignant` solo se utiliza como etiqueta observada opcional: debe ser 0 o 1 para evaluar esa fila; etiquetas inválidas se indican en `estado_etiqueta`. Una etiqueta inválida no impide clasificar mediciones válidas.

La descarga incluye `clase_predicha`, `probabilidad_benigno`, `probabilidad_maligno` y `estado_validacion`, además de las columnas de entrada. Las filas con mediciones inválidas conservan sus errores y probabilidades vacías. Los números de fila de los errores empiezan en 1 después del encabezado. Las métricas opcionales solo usan filas con mediciones y etiquetas válidas, con recuento de exclusiones. El archivo se procesa en memoria y no se guarda por defecto; evita introducir información identificable. El límite de carga es 20 MB.

## Dataset y límites

Breast Cancer Wisconsin Diagnostic: 569 observaciones con características de núcleos celulares extraídas de imágenes de aspirado con aguja fina. Diez características se resumen por media, error estándar y `worst` (media de los tres valores mayores). Las definiciones verificadas se incluyen en Inicio. No se inventan unidades físicas: no están especificadas en la descripción distribuida. Fuente: `sklearn.datasets.load_breast_cancer().DESCR` y [documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html).

La regresión logística fue seleccionada mediante AUC ROC en validación cruzada de cinco particiones de entrenamiento. La partición de prueba es fija, estratificada (80/20, semilla 42). La exactitud mide aciertos totales; precision/VPP mide positivos observados entre positivos predichos. Las probabilidades no son riesgo clínico validado. Las métricas exploratorias del umbral se separan de las originales: seleccionar formalmente un umbral requiere entrenamiento/validación.

Dataset pequeño e histórico, sin garantía de representatividad demográfica ni de generalización temporal, institucional o entre dispositivos. El desempeño del benchmark no garantiza cada registro individual. Modificar variables no demuestra causalidad y puede crear combinaciones poco realistas. No se ofrecen atribuciones personales ni se asume validez clínica.

## Archivos y preservación

Entradas originales conservadas: `notebooks/01_cancer_detection_challenge.ipynb`, `data/processed/train.csv`, `data/processed/test.csv`, `models/best_model.joblib`, `reports/metrics.json`, `reports/model_comparison.csv`, `reports/model_card.md`. No se regeneran al abrir la app.

Archivos de la aplicación: `app.py`, `src/inference.py`, `.streamlit/config.toml`. Pruebas: `tests/test_pipeline.py` (original), `tests/test_inference.py`, `tests/test_app.py`. Dependencias: `requirements.txt`, `environment.yml`.

El `.gitignore` excluye entornos locales (incluido `.venv`), cachés, secretos, archivos comprimidos, datos crudos, figuras regenerables y modelos adicionales. Conserva explícitamente los pequeños artefactos originales necesarios para ejecutar la app al clonar: `train.csv`, `test.csv`, `best_model.joblib`, `metrics.json` y `model_comparison.csv` (unos 130 KB en conjunto). También incluye el notebook principal y la ficha del modelo. Si faltan estos archivos, restaura los originales; la app no los regenera silenciosamente.

### Subir a GitHub

Desde la raíz del proyecto, crea primero un repositorio vacío en tu cuenta de GitHub y ejecuta:

```bat
git init
git add .
git status --short
git commit -m "Agregar laboratorio interactivo CancerDetection"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/CancerDetection.git
git push -u origin main
```

Reemplaza `TU_USUARIO` por tu usuario. Antes del commit, revisa que `git status --short` no incluya entornos ni archivos privados. Git no permite excluir por tamaño mediante `.gitignore`: revisa los archivos nuevos de gran tamaño antes de agregarlos. Si un archivo ya estaba versionado, agregarlo al `.gitignore` no lo retira del seguimiento; usa `git rm --cached -- ruta/al/archivo` para retirarlo del siguiente commit conservando la copia local (esto no borra su historial anterior).

## Verificación y resolución de problemas

```bat
python -m pip check
python -m pytest -q
python -m streamlit run app.py
```

Las pruebas comprueban esquema/orden, exclusión de metadatos, etiquetas, clases en orden inverso, rechazo de entradas inválidas, equivalencia exacta con el pipeline y ausencia de fit durante inferencia. También reproducen todas las métricas del reporte (tolerancia absoluta 1e-8 y relativa 1e-6), verifican la regla >= del umbral, validación CSV parcial, rangos de entrenamiento y navegación/formularios con Streamlit AppTest.

Verificación de esta entrega: **18 pruebas aprobadas**, `pip check` sin incompatibilidades y arranque local de Streamlit en el puerto 8501. Las métricas recalculadas coinciden con los reportes originales. Versiones fijadas comprobadas sobre Python 3.12.15 en Windows y el artefacto existente; no se verificó una instalación nueva completa de Conda.

Si aparece incompatibilidad del artefacto, revisa el intérprete activo e instala las versiones fijadas. No reemplaces los reportes ni reentrenes para ocultar una discrepancia. Evaluación muestra cualquier diferencia detectada. Si el puerto 8501 está ocupado, agrega `--server.port 8502` al comando y abre esa dirección.
