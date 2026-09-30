"""Small Streamlit demo for the BACE-1 molecular classification baseline."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import streamlit as st
from rdkit import Chem

try:
    # Drawing is a presentation feature. Some hosted Python/RDKit combinations
    # cannot load rdMolDraw2D, so keep it optional and preserve predictions.
    from rdkit.Chem import Draw
except ImportError:
    Draw = None

from molml.artifacts import build_model_bundle
from molml.config import load_config
from molml.predict import predict_smiles
from molml.train import fit_experiment


ROOT = Path(__file__).resolve().parent
BACE_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv"
BACE_SHA256 = "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737"
MODEL_CONFIGS = {
    "Random Forest": ("random_forest", "rf_scaffold"),
    "Regresión Logística": ("logistic_regression", "lr_scaffold"),
}

st.set_page_config(
    page_title="MolML | BACE-1 demo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;}
      .eyebrow {color:#168b9b; font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;}
      .hero {background:linear-gradient(120deg,#f1f8fb 0%,#f6fbfb 100%); border:1px solid #dce9ed;
             border-radius:18px; padding:1.5rem 1.7rem; margin: .6rem 0 1.2rem 0;}
      .muted {color:#586b78;}
      div[data-testid="stMetric"] {background:#f7fafb; border:1px solid #e1eaee; padding:14px 16px; border-radius:12px;}
      div.stButton > button {border-radius:10px; min-height:2.8rem; font-weight:650;}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=1)
def fetch_bace_dataset() -> bytes:
    """Download the pinned public benchmark and verify its recorded checksum."""
    with urlopen(BACE_URL, timeout=30) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != BACE_SHA256:
        raise RuntimeError(
            "El archivo descargado no coincide con la versión validada del benchmark."
        )
    return data


@st.cache_resource(show_spinner=False, max_entries=2)
def load_demo_model(model_label: str):
    """Train the selected fixed baseline once per app process and cache it."""
    model_name, config_name = MODEL_CONFIGS[model_label]
    raw_data = fetch_bace_dataset()
    data_path = Path(tempfile.gettempdir()) / "molml-bace-pinned.csv"
    data_path.write_bytes(raw_data)

    base = load_config(ROOT / "configs" / f"{config_name}.yaml")
    config = replace(base, model=model_name, data_path=str(data_path))
    model, parts, _, _, _ = fit_experiment(config)
    bundle = build_model_bundle(
        model,
        config,
        parts["train"]["smiles"].tolist(),
        dataset_sha256=BACE_SHA256,
    )

    # Two deterministic, held-out examples make the demo usable immediately.
    examples = {}
    test = parts["test"]
    for label, key in ((1, "reported_inhibitor"), (0, "reported_non_inhibitor")):
        rows = test.loc[test["label"] == label, "smiles"]
        if rows.empty:
            raise RuntimeError("No hay ejemplos de ambas clases en el conjunto de prueba.")
        examples[key] = str(rows.iloc[0])
    return bundle, examples


def get_test_metrics(model_label: str) -> dict:
    _, result_name = MODEL_CONFIGS[model_label]
    metrics_path = ROOT / "results" / "v1" / result_name / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))["test"]


st.markdown('<div class="eyebrow">Demo interactiva · aprendizaje automático molecular</div>', unsafe_allow_html=True)
st.title("¿Qué patrón químico reconoce el modelo?")
st.markdown(
    "Esta demo explora si una molécula se parece a las que el conjunto BACE-1 clasifica "
    "como inhibidoras. Es una primera estimación del modelo, no una prueba de unión a la proteína."
)

try:
    model_label = st.selectbox("Modelo", list(MODEL_CONFIGS), index=0)
    with st.spinner("Preparando el modelo para la demo… (solo hace falta al iniciar)"):
        bundle, examples = load_demo_model(model_label)
except Exception as exc:
    st.error(f"No se pudo iniciar la demo: {exc}")
    st.stop()

metrics = get_test_metrics(model_label)
cols = st.columns(4)
cols[0].metric("Moléculas del benchmark", "1.513")
cols[1].metric("ROC-AUC en prueba", f"{metrics['roc_auc']:.3f}")
cols[2].metric("PR-AUC en prueba", f"{metrics['pr_auc']:.3f}")
cols[3].metric("MCC en prueba", f"{metrics['mcc']:.3f}")
st.caption(
    "Resultados publicados en el repositorio para el conjunto de prueba separado por estructura química "
    "(scaffold split). Son métricas del benchmark, no una garantía para la molécula que introduzcas."
)

st.divider()
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Prueba una estructura")
    example_kind = st.radio(
        "Elige una entrada",
        ["Escribir una molécula", "Ejemplo del benchmark"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if example_kind == "Ejemplo del benchmark":
        example_label = st.selectbox(
            "Ejemplo",
            ["Ejemplo A", "Ejemplo B"],
            help="Son moléculas reservadas para prueba en el benchmark; sus etiquetas reales no se muestran aquí.",
        )
        example_key = "reported_inhibitor" if example_label == "Ejemplo A" else "reported_non_inhibitor"
        default_smiles = examples[example_key]
    else:
        default_smiles = "CCO"

    smiles = st.text_area(
        "Estructura en SMILES",
        value=default_smiles,
        height=94,
        help="SMILES es una forma compacta de escribir una estructura química como texto.",
    ).strip()
    run_prediction = st.button("Analizar molécula", type="primary", width="stretch")
    if example_kind == "Escribir una molécula":
        st.caption("Puedes empezar con `CCO` (etanol) o pegar la cadena SMILES de otra molécula.")
    st.caption("La estructura se procesa en la instancia de Streamlit para calcular el resultado; esta demo no la guarda ni usa una API separada.")

with right:
    st.subheader("¿Qué hace el programa?")
    st.markdown(
        "1. **Lee** la estructura escrita como SMILES.\n"
        "2. **Resume** sus fragmentos químicos con una huella Morgan.\n"
        "3. **Compara** ese patrón con los aprendidos a partir de moléculas del benchmark.\n"
        "4. **Devuelve** una clase estimada y señales de cobertura química."
    )

if run_prediction:
    prediction = predict_smiles(bundle, [smiles])[0]
    st.divider()
    st.subheader("Resultado")
    if prediction["status"] != "ok":
        st.error(prediction["error"] or "La entrada no es un SMILES válido.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        result_cols = st.columns([0.8, 1.2], gap="large")
        with result_cols[0]:
            if Draw is not None:
                st.image(Draw.MolToImage(mol, size=(420, 280)), width="stretch")
            else:
                st.info("La predicción funciona, pero este entorno no permite dibujar la estructura química.")
        with result_cols[1]:
            positive = prediction["prediction"] == 1
            if positive:
                st.success("El modelo la clasifica como similar a la clase etiquetada inhibidora.")
            else:
                st.info("El modelo no la clasifica como inhibidora según el umbral elegido.")
            score = prediction["probability_class_1"]
            st.metric("Puntuación del modelo para la clase inhibidora", f"{score:.1%}")
            st.progress(float(score))
            st.caption(
                "Esta puntuación sale del modelo; no es una probabilidad medida de unión, "
                "ni una medida experimental de actividad."
            )

        diag_cols = st.columns(3)
        diag_cols[0].metric("Molécula de entrenamiento más parecida", f"{prediction['nearest_train_similarity']:.2f}")
        diag_cols[1].metric("Dentro de cobertura química estimada", "Sí" if prediction["in_similarity_domain"] else "No")
        diag_cols[2].metric("Esqueleto no visto al entrenar", "Sí" if prediction["novel_scaffold"] else "No")
        with st.expander("Cómo leer estas señales"):
            st.write(
                "La similitud y el esqueleto sirven para advertir si la molécula se parece a los ejemplos "
                "de entrenamiento. Son indicadores aproximados de cobertura, no una medida de certeza. "
                "Un esqueleto nuevo o una similitud baja aconsejan interpretar la puntuación con más cautela."
            )
        with st.expander("Limitaciones científicas"):
            st.write(
                "El programa usa etiquetas del benchmark BACE-1 y huellas químicas 2D. No recibe la estructura "
                "tridimensional de BACE-1, no calcula el encaje físico de la molécula, no predice seguridad "
                "y no sustituye una medición de laboratorio."
            )
        st.code(smiles, language=None)

st.divider()
st.markdown(
    "**En resumen:** una demo educativa de un flujo de aprendizaje automático reproducible. "
    "Una predicción positiva solo indica semejanza con la clase positiva del conjunto de datos."
)
st.markdown("[Ver el código, los datos y la validación en GitHub](https://github.com/jjimenezgar/MolML-Pipeline)")
