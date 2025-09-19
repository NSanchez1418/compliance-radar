# -*- coding: utf-8 -*-
import os, time, requests, streamlit as st

# Solo aquí se define page_config (una sola vez)
st.set_page_config(page_title="Compliance Radar — Launcher", layout="wide")

st.sidebar.title("Launcher")
mode = st.sidebar.radio("Elige modo", ["🔎 Diagnóstico IA", "🚀 App completa"], index=0)

# Token común
try:
    from dotenv import load_dotenv; load_dotenv(override=True)
except Exception:
    pass
HF_TOKEN = (st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# ============ MODO DIAGNÓSTICO ============
if mode.startswith("🔎"):
    st.title("🔎 Diagnóstico IA (Hugging Face)")
    st.write("Token OK:", HF_TOKEN.startswith("hf_"))

    txt = st.text_area(
        "Texto de prueba",
        "Intento de soborno de $500 en Quito. Contacto con mafia en Guayaquil.",
        height=120
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Probar ZERO-SHOT"):
            try:
                url = "https://api-inference.huggingface.co/models/joeddav/xlm-roberta-large-xnli"
                payload = {
                    "inputs": txt[:1500],
                    "parameters": {"candidate_labels":
                                   ["soborno/coima","amenaza/coacción","contacto con mafia","otros"],
                                   "multi_label": False},
                    "options": {"wait_for_model": True}
                }
                r = requests.post(url, headers=HEADERS, json=payload, timeout=25)
                if r.status_code in (503, 429):
                    time.sleep(2); r = requests.post(url, headers=HEADERS, json=payload, timeout=25)
                r.raise_for_status()
                out = r.json()
                st.success("ZERO-SHOT OK")
                st.write(list(zip(out.get("labels",[]), out.get("scores",[])))[:4]
                         if isinstance(out, dict) else out)
            except Exception as e:
                st.error(f"Zero-shot error: {e}")

    with col2:
        if st.button("Probar NER"):
            try:
                url = "https://api-inference.huggingface.co/models/Davlan/bert-base-multilingual-cased-ner-hrl"
                payload = {"inputs": txt[:1500],
                           "parameters": {"aggregation_strategy": "simple"},
                           "options": {"wait_for_model": True}}
                r = requests.post(url, headers=HEADERS, json=payload, timeout=25)
                if r.status_code in (503, 429):
                    time.sleep(2); r = requests.post(url, headers=HEADERS, json=payload, timeout=25)
                r.raise_for_status()
                out = r.json()
                st.success("NER OK")
                st.write(out[:5] if isinstance(out, list) else out)
            except Exception as e:
                st.error(f"NER error: {e}")

    st.info('Si ambos botones funcionan, cambia el radio a "🚀 App completa".')

# ============ MODO APP COMPLETA ============
else:
    st.title("🚀 Lanzando app completa…")
    try:
        import compliance_app
        with st.spinner("Cargando interfaz…"):
            compliance_app.render()   # <<< IMPORTANTÍSIMO: SOLO LLAMAMOS A LA FUNCIÓN
        st.success("App cargada.")
    except Exception as e:
        st.error("❌ Error al iniciar la aplicación completa. Detalle abajo:")
        st.exception(e)
        st.info("Vuelve al modo '🔎 Diagnóstico IA' para verificar token/modelos.")



