# -*- coding: utf-8 -*-
import os, time, re, datetime as dt, requests
import pandas as pd
import streamlit as st

# --- .env/Secrets (robusto) ---
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

st.set_page_config(page_title="Compliance Radar", layout="wide")
st.title("Compliance Radar FF.AA. — MVP")
st.caption("Registro y triage de incidentes (Ejército, Marina, Aviación; tropa y oficiales).")

# ---- Controles de ejecución (barra lateral) ----
# IA real por defecto (modo seguro = OFF)
SAFE_MODE = st.sidebar.toggle("🛟 Modo seguro (sin IA)", value=False)
N_MAX = st.sidebar.number_input("Máximo de filas a procesar", min_value=1, max_value=200, value=20, step=1)

# Diagnóstico rápido de token/modelos (opcional)
with st.sidebar.expander("Diagnóstico HuggingFace"):
    # token
    HF_TOKEN = (
        st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or ""
    ).strip()
    st.write("Token OK:", HF_TOKEN.startswith("hf_"))
    # prueba rápida (sin bloquear analítica)
    test_txt = st.text_area("Texto de prueba", "Intento de soborno de $500 en Quito...", height=80)
    if st.button("Probar IA ahora"):
        try:
            z = requests.post(
                "https://api-inference.huggingface.co/models/joeddav/xlm-roberta-large-xnli",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": test_txt, "parameters":{"candidate_labels":["soborno/coima","amenaza/coacción"],"multi_label":False}, "options":{"wait_for_model": True}},
                timeout=60
            ).json()
            st.write("Zero-shot (labels/scores):", list(zip(z.get("labels",[]), z.get("scores",[])))[:3] if isinstance(z, dict) else z)
        except Exception as e:
            st.error(f"Zero-shot error: {e}")

# Si no hay token, avisamos (la app igual funciona en modo seguro)
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
if not HF_TOKEN.startswith("hf_"):
    st.warning("Configura HUGGINGFACEHUB_API_TOKEN en Secrets (Cloud) o .env (local) para usar IA.")

# --- Modelos ---
ZERO_SHOT = "joeddav/xlm-roberta-large-xnli"                  # zero-shot multilingüe
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"     # NER PER/ORG/LOC multilingüe

LABELS = [
    "soborno/coima","amenaza/coacción","contacto con mafia",
    "tráfico de combustibles","minería ilegal","narcotráfico",
    "contrabando","corrupción interna","otros"
]

# --- Llamadas a HF ---
def hf_zero_shot(text, labels=LABELS):
    if not HF_HEADERS: return []
    url = f"https://api-inference.huggingface.co/models/{ZERO_SHOT}"
    payload = {
        "inputs": text[:2000],
        "parameters":{"candidate_labels":labels,"multi_label":False},
        "options":{"wait_for_model": True}
    }
    r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    if r.status_code == 503:
        time.sleep(2); r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    return list(zip(out.get("labels",[]), out.get("scores",[]))) if isinstance(out, dict) else []

def hf_ner(text):
    if not HF_HEADERS: return []
    url = f"https://api-inference.huggingface.co/models/{NER_MODEL}"
    payload = {
        "inputs": text[:2000],
        "parameters":{"aggregation_strategy":"simple"},
        "options":{"wait_for_model": True}
    }
    r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    if r.status_code == 503:
        time.sleep(2); r = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    out = r.json()
    if isinstance(out, list):
        return [(p.get("word"), p.get("entity_group"), float(p.get("score",0))) for p in out]
    return []

# --- Regex & riesgo ---
MONEY_RE = re.compile(r"\$?\s?([0-9]{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)")
DATE_RE  = re.compile(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})\b")
VIOLENCE = {"arma","amenaza","disparo","golpe","intimidación","secuestr","extors"}

def extract_fields(text):
    return {"montos": MONEY_RE.findall(text)[:5], "fechas": DATE_RE.findall(text)[:5]}

def parse_dates_found(raw_dates):
    fmts = ("%d/%m/%Y","%d-%m-%Y","%d.%m.%Y","%d/%m/%y","%d-%m-%y","%d.%m.%y")
    parsed = []
    for s in raw_dates:
        for f in fmts:
            try:
                parsed.append(dt.datetime.strptime(s.strip(), f).date()); break
            except: pass
    return parsed

def risk_score(tipo, text, raw_dates):
    score = 0
    if tipo in {"amenaza/coacción","contacto con mafia"}: score += 3
    if tipo in {"soborno/coima","narcotráfico","minería ilegal","tráfico de combustibles","contrabando"}: score += 2
    if MONEY_RE.search(text): score += 1
    if any(w in text.lower() for w in VIOLENCE): score += 1
    parsed = parse_dates_found(raw_dates)
    today = dt.date.today()
    if any((today - d).days <= 7 for d in parsed): score += 1
    return score

# --- UI ---
tabs = st.tabs(["Reportar incidente", "Analizar (CSV)"])

with tabs[0]:
    st.subheader("Reportar incidente")
    with st.form("f"):
        rama = st.selectbox("Rama", ["Ejército","Marina","Aviación"])
        grado = st.selectbox("Grado", ["Tropa","Oficial"])
        unidad = st.text_input("Unidad / Destacamento")
        provincia = st.text_input("Provincia")
        canton = st.text_input("Cantón")
        fecha_incidente = st.date_input("Fecha del incidente", value=dt.date.today())
        relato = st.text_area("Relato del incidente", height=160,
                              placeholder="Describa lo sucedido (puede anonimizar nombres).")
        ok = st.form_submit_button("Generar registro CSV")
    if ok:
        df = pd.DataFrame([{
            "rama": rama, "grado": grado, "unidad": unidad,
            "provincia": provincia, "canton": canton,
            "fecha_incidente": fecha_incidente.isoformat(),
            "relato": relato
        }])
        st.success("Registro creado. Descárguelo y consolide varios antes de analizar.")
        st.download_button("Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="incidentes_compliance.csv", mime="text/csv")

with tabs[1]:
    st.subheader("Analizar CSV")
    file = st.file_uploader(
        "Sube CSV con columnas: rama,grado,unidad,provincia,canton,fecha_incidente,relato",
        type=["csv"]
    )
    if file:
        df = pd.read_csv(file)
        df = df.head(int(N_MAX))  # límite de filas
        req = {"rama","grado","unidad","provincia","canton","fecha_incidente","relato"}
        if not req.issubset(df.columns):
            st.error(f"Faltan columnas: {req - set(df.columns)}")
            st.stop()

        # 1) Zero-shot
        st.markdown("### 1) Tipo de incidente (Zero-shot)")
        out = []
        bar1 = st.progress(0, text="Clasificando…")
        for i, row in df.iterrows():
            text = str(row["relato"])
            if SAFE_MODE:
                tipo, score = ("otros", 0.0)
            else:
                try:
                    zs = hf_zero_shot(text)
                except Exception as e:
                    st.error(f"Zero-shot error: {e}"); zs = []
                tipo, score = (zs[0][0], float(zs[0][1])) if zs else ("otros", 0.0)
            out.append({**row.to_dict(), "tipo_predicho": tipo, "score_tipo": round(score,3)})
            bar1.progress(int((i+1)/len(df)*100))
        bar1.empty()
        df = pd.DataFrame(out)

        # 2) NER + regex
        st.markdown("### 2) Entidades (NER) + Montos/Fechas (Regex)")
        ents_col, montos_col, fechas_col = [], [], []
        bar2 = st.progress(0, text="Extrayendo…")
        for i, row in df.iterrows():
            text = str(row["relato"])
            if SAFE_MODE:
                ents = []
            else:
                try:
                    ents = hf_ner(text)
                except Exception as e:
                    st.error(f"NER error: {e}"); ents = []
            fields = extract_fields(text)
            ents_col.append(", ".join({w for w,t,_ in ents if t in ("ORG","PER","LOC")}))
            montos_col.append(", ".join(fields["montos"]))
            fechas_col.append(", ".join(fields["fechas"]))
            bar2.progress(int((i+1)/len(df)*100))
        bar2.empty()
        df["entidades"] = ents_col
        df["montos"] = montos_col
        df["fechas"] = fechas_col

        # 3) Riesgo
        st.markdown("### 3) Prioridad (Reglas de riesgo)")
        risks = []
        for _, row in df.iterrows():
            raw_dates = row["fechas"].split(", ") if row["fechas"] else []
            risks.append(risk_score(row["tipo_predicho"], str(row["relato"]), raw_dates))
        df["riesgo"] = risks

        # 4) Resumen + descarga
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Conteo por tipo**")
            st.dataframe(df["tipo_predicho"].value_counts())
        with col2:
            st.write("**Top-10 por riesgo**")
            st.dataframe(df.sort_values(["riesgo","score_tipo"], ascending=False).head(10))

        st.markdown("### 4) Tabla completa (ordenada por riesgo)")
        st.dataframe(df.sort_values(["riesgo","score_tipo"], ascending=False), use_container_width=True)

        st.download_button("Descargar resultados (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="incidentes_priorizados.csv", mime="text/csv")
    else:
        st.info("Sube el CSV para comenzar.")
