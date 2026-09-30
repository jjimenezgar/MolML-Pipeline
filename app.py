"""Small Streamlit demo for the BACE-1 molecular classification baseline."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import matplotlib.pyplot as plt
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
    "Logistic Regression": ("logistic_regression", "lr_scaffold"),
}


def draw_structure(mol) -> bytes:
    """Render a simple 2D skeletal diagram without RDKit's optional drawing extension."""
    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    coordinates = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

    fig, ax = plt.subplots(figsize=(5.2, 3.3), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for bond in mol.GetBonds():
        start = coordinates[bond.GetBeginAtomIdx()]
        end = coordinates[bond.GetEndAtomIdx()]
        dx, dy = end.x - start.x, end.y - start.y
        length = max((dx * dx + dy * dy) ** 0.5, 1e-8)
        nx, ny = -dy / length * 0.075, dx / length * 0.075
        order = bond.GetBondTypeAsDouble()
        styles = [(0.0, "-")]
        if bond.GetIsAromatic():
            styles = [(0.0, "--")]
        elif order >= 2.5:
            styles = [(-1, "-"), (0, "-"), (1, "-")]
        elif order >= 1.5:
            styles = [(-0.65, "-"), (0.65, "-")]
        for offset, linestyle in styles:
            ax.plot(
                [start.x + offset * nx, end.x + offset * nx],
                [start.y + offset * ny, end.y + offset * ny],
                color="#334155",
                linewidth=1.8,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=1,
            )

    for atom, point in zip(mol.GetAtoms(), coordinates):
        symbol = atom.GetSymbol()
        if symbol != "C" or atom.GetFormalCharge():
            label = symbol
            charge = atom.GetFormalCharge()
            if charge:
                label += f"{'+' if charge > 0 else '−'}{abs(charge) if abs(charge) > 1 else ''}"
            ax.text(
                point.x,
                point.y,
                label,
                ha="center",
                va="center",
                fontsize=12,
                color="#0f766e" if symbol in {"N", "O", "S", "P"} else "#334155",
                fontweight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
                zorder=2,
            )

    ax.set_aspect("equal")
    ax.margins(0.22)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    image = BytesIO()
    fig.savefig(image, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return image.getvalue()

st.set_page_config(
    page_title="MolML | BACE-1 analysis",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 3.4rem; padding-bottom: 3rem;}
      .hero {background:linear-gradient(120deg,#f1f8fb 0%,#f6fbfb 100%); border:1px solid #dce9ed;
             border-radius:18px; padding:1.7rem 1.9rem; margin: .35rem 0 1.35rem 0;}
      .hero .eyebrow {color:#168b9b; font-size:.76rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .7rem 0;}
      .hero h1 {color:#17324d; font-size:2.35rem; line-height:1.15; margin:0 0 .65rem 0;}
      .hero p {color:#42596b; font-size:1.04rem; line-height:1.6; margin:0; max-width:900px;}
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
            "The downloaded file does not match the validated benchmark version."
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
            raise RuntimeError("The test split does not contain examples from both classes.")
        examples[key] = str(rows.iloc[0])
    return bundle, examples


def get_test_metrics(model_label: str) -> dict:
    _, result_name = MODEL_CONFIGS[model_label]
    metrics_path = ROOT / "results" / "v1" / result_name / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))["test"]


st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">Molecular machine learning · BACE-1 benchmark</div>
      <h1>BACE-1 activity classification</h1>
      <p>Enter a molecule as SMILES to obtain a model-predicted inhibitor label, score, and chemical-similarity diagnostics. These benchmark-based estimates require experimental validation.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

try:
    model_label = st.selectbox("Model", list(MODEL_CONFIGS), index=0)
    with st.spinner("Preparing the model… (only needed at startup)"):
        bundle, examples = load_demo_model(model_label)
except Exception as exc:
    st.error(f"The app could not start: {exc}")
    st.stop()

metrics = get_test_metrics(model_label)
cols = st.columns(4)
cols[0].metric("Benchmark molecules", "1,513")
cols[1].metric("Test ROC-AUC", f"{metrics['roc_auc']:.3f}")
cols[2].metric("Test PR-AUC", f"{metrics['pr_auc']:.3f}")
cols[3].metric("Test MCC", f"{metrics['mcc']:.3f}")
st.caption(
    "Repository results on the scaffold-separated test set. These benchmark metrics do not guarantee "
    "the performance of an individual prediction."
)

st.divider()
left, right = st.columns([1.15, 0.85], gap="large")
with left:
    st.subheader("Analyze a molecule")
    example_kind = st.radio(
        "Choose an input",
        ["Enter a molecule", "Benchmark example"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if example_kind == "Benchmark example":
        example_label = st.selectbox(
            "Example",
            ["Example A", "Example B"],
            help="Held-out benchmark molecules. Their reported labels are not shown here.",
        )
        example_key = "reported_inhibitor" if example_label == "Example A" else "reported_non_inhibitor"
        default_smiles = examples[example_key]
    else:
        default_smiles = "CCO"

    smiles = st.text_area(
        "Molecular structure (SMILES)",
        value=default_smiles,
        height=94,
        help="SMILES is a compact text notation for chemical structures.",
    ).strip()
    run_prediction = st.button("Run analysis", type="primary", width="stretch")
    if example_kind == "Enter a molecule":
        st.caption("Start with `CCO` (ethanol), or paste another molecule’s SMILES string.")
    st.caption("The SMILES is processed by this Streamlit app and is not saved or sent to a separate API.")

with right:
    st.subheader("How it works")
    st.markdown(
        "1. **Validates** the SMILES structure.\n"
        "2. **Encodes** it as a Morgan fingerprint.\n"
        "3. **Compares** its features with molecules in the benchmark.\n"
        "4. **Reports** a predicted class, model score, and chemical-similarity diagnostics."
    )

if run_prediction:
    prediction = predict_smiles(bundle, [smiles])[0]
    st.divider()
    st.subheader("Analysis")
    if prediction["status"] != "ok":
        st.error(prediction["error"] or "The input is not a valid SMILES string.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        result_cols = st.columns([0.8, 1.2], gap="large")
        with result_cols[0]:
            if Draw is not None:
                st.image(Draw.MolToImage(mol, size=(420, 280)), width="stretch")
            else:
                st.image(draw_structure(mol), width="stretch")
        with result_cols[1]:
            positive = prediction["prediction"] == 1
            if positive:
                st.success("The model classifies this molecule as similar to the inhibitor-labelled class.")
            else:
                st.info("The model does not classify this molecule as an inhibitor at the selected threshold.")
            score = prediction["probability_class_1"]
            st.metric("Model score for the inhibitor-labelled class", f"{score:.1%}")
            st.progress(float(score))
            st.caption(
                "This is a model score, not a measured probability of binding or experimental activity."
            )

        diag_cols = st.columns(3)
        diag_cols[0].metric("Nearest training-molecule similarity", f"{prediction['nearest_train_similarity']:.2f}")
        diag_cols[1].metric("Within estimated chemical domain", "Yes" if prediction["in_similarity_domain"] else "No")
        diag_cols[2].metric("Scaffold unseen during training", "Yes" if prediction["novel_scaffold"] else "No")
        with st.expander("Interpreting these diagnostics"):
            st.write(
                "Similarity and scaffold novelty indicate how closely this molecule relates to the training "
                "chemistry. They are approximate coverage diagnostics, not measures of certainty. A novel "
                "scaffold or low similarity calls for greater caution when interpreting the score."
            )
        with st.expander("Scope and limitations"):
            st.write(
                "The model uses BACE-1 benchmark labels and 2D molecular fingerprints. It does not use the "
                "3D structure of BACE-1, calculate physical binding, predict safety, or replace laboratory testing."
            )
        st.code(smiles, language=None)

st.divider()
st.markdown(
    "A reproducible molecular machine-learning workflow with transparent benchmark evaluation. "
    "A positive prediction indicates similarity to the dataset’s inhibitor-labelled class; it does not "
    "establish experimental inhibition or binding."
)
st.markdown("[Explore the code, data, and validation on GitHub](https://github.com/jjimenezgar/MolML-Pipeline)")
