# compliance_app.py
# -*- coding: utf-8 -*-

import os, time, re, datetime as dt, requests
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass

# ============================== CONFIG & MODELOS ==============================

ZERO_SHOT = "joeddav/xlm-roberta-large-xnli"                  # Zero-shot (multilingüe)
NER_MODEL = "Davlan/bert-base-multilingual-cased-ner-hrl"     # NER multilingüe (PER/ORG/LOC)

LABELS = [
    "soborno/coima", "amenaza/coacción", "contacto con mafia",
    "tráfico de combustibles", "minería ilegal", "narcotráfico",
    "contrabando", "corrupción interna", "otros"
]

# Regex y reglas de riesgo
MONEY_RE = re.compile(r"\$?\s?([0-9]{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)")
DATE_RE  = re.compile(r"\b(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
VIOLENCE = {"arma","amenaza","disparo","golpe","intimidación","secuestr","extors"}


# ============================== FUNCIONES IA ==============================

def get_headers():
    """Lee el token desde Secrets o .env y forma los headers."""
    token = (st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
             or os.getenv("HUGGINGFACEHUB_API_TOKEN") or "").strip()
    if not token.startswith("hf_"):
        st.warning("No se detectó un token HF válido. Cárgalo en Secrets o .env.")
        return {}
    return {"Authorization": f"Bearer {token}"}


def hf_zero_shot(text, labels=LABELS, headers=None):
    """Clasifica texto con zero-shot. Devuelve lista [(label, score), ...]"""
    if not headers:
        return []
    url = f"https://api-inference.huggingface.co/models/{ZERO_SHOT}"
    payload = {
        "inputs": text[:2000],
        "parameters": {"candidate_labels": labels, "multi_label": False},
        "options": {"wait_for_model": True}
    }
    last = None
    for _ in range(2):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (503, 429):
            time.sleep(2)
            last = r.text
            continue
        r.raise_for_status()
        out = r.json()
        return list(zip(out.get("labels", []), out.get("scores", []))) if isinstance(out, dict) else []
    if last:
        st.error(f"Zero-shot: la API no respondió (detalles: {last[:120]}...)")
    return []


def hf_ner(text, headers=None):
    """NER con agregación 'simple'. Devuelve [(word, entity_group, score), ...]"""
    if not headers:
        return []
    url = f"https://api-inference.huggingface.co/models/{NER_MODEL}"
    payload = {
        "inputs": text[:2000],
        "parameters": {"aggregation_strategy": "simple"},
        "options": {"wait_for_model": True}
    }
    last = None
    for _ in range(2):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (503, 429):
            time.sleep(2)
            last = r.text
            continue
        r.raise_for_status()
        out = r.json()
        if isinstance(out, list):
            return [(p.get("word"), p.get("entity_group"), float(p.get("score", 0))) for p in out]
        break
    if last:
        st.error(f"NER: la API no respondió (detalles: {last[:120]}...)")
    return []


# ============================== UTILIDADES ==============================

def extract_fields(text: str):
    """Extrae montos (regex) y fechas (regex)."""
    return {"montos": MONEY_RE.findall(text)[:5], "fechas": DATE_RE.findall(text)[:5]}


def parse_dates_found(raw_dates):
    """Convierte strings a fechas si es posible."""
    fmts = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y", "%d-%m-%y", "%d.%m.%y", "%Y-%m-%d")
    out = []
    for s in raw_dates:
        for f in fmts:
            try:
                out.append(dt.datetime.strptime(s.strip(), f).date())
                break
            except Exception:
                pass
    return out


def risk_score(tipo: str, text: str, raw_dates):
    """Reglas sencillas de riesgo."""
    score = 0
    if tipo in {"amenaza/coacción", "contacto con mafia"}:
        score += 3
    if tipo in {"soborno/coima", "narcotráfico", "minería ilegal", "tráfico de combustibles", "contrabando"}:
        score += 2
    if MONEY_RE.search(text):
        score += 1
    if any(w in text.lower() for w in VIOLENCE):
        score += 1
    parsed = parse_dates_found(raw_dates)
    today = dt.date.today()
    if any((today - d).days <= 7 for d in parsed):
        score += 1
    return score


# ============================== UI (render) ==============================

def render():
    """Dibuja la interfaz completa (se llama desde app.py)."""
    st.title("Compliance Radar FF.AA. — MVP")
    st.caption("Registro y triage de incidentes (Ejército, Marina, Aviación/F.A.; tropa y oficiales).")

    headers = get_headers()

    tabs = st.tabs(["Reportar incidente", "Analizar (CSV)"])

    # -------- Tab 1: Formulario -> CSV -------
    with tabs[0]:
        st.subheader("Reportar incidente")
        with st.form("f"):
            rama = st.selectbox("Rama", ["Ejército", "Marina", "Fuerza Aérea"])
            grado = st.selectbox("Grado", ["Tropa", "Oficial"])
            unidad = st.text_input("Unidad / Destacamento")
            provincia = st.text_input("Provincia")
            canton = st.text_input("Cantón")
            fecha_incidente = st.date_input("Fecha del incidente", value=dt.date.today())
            relato = st.text_area("Relato del incidente", height=160,
                                  placeholder="Describa lo sucedido (puede anonimizar nombres).")
            ok = st.form_submit_button("Generar registro CSV")

        if ok:
            df_form = pd.DataFrame([{
                "rama": rama, "grado": grado, "unidad": unidad,
                "provincia": provincia, "canton": canton,
                "fecha_incidente": fecha_incidente.isoformat(),
                "relato": relato
            }])
            st.success("Registro creado. Descárguelo para consolidar varios antes de analizar.")
            csv_bytes_form = df_form.to_csv(index=False).encode("utf-8-sig")  # BOM friendly
            st.download_button("Descargar CSV", csv_bytes_form,
                               file_name="incidentes_compliance.csv",
                               mime="text/csv", key="dl_form")

    # -------- Tab 2: Analizar CSV (Zero-shot + NER + Reglas) -------
    with tabs[1]:
        st.subheader("Analizar (CSV)")
        st.caption("Columnas requeridas: rama, grado, unidad, provincia, canton, fecha_incidente, relato")
        file = st.file_uploader("Sube CSV", type=["csv"])
        N_MAX = st.number_input("Máximo de filas a procesar",
                                min_value=1, max_value=200, value=6, step=1,
                                help="Útil para demos y evitar saturar la API.")

        if file:
            try:
                df = pd.read_csv(file)
                req = {"rama", "grado", "unidad", "provincia", "canton", "fecha_incidente", "relato"}
                if not req.issubset(df.columns):
                    st.error(f"Faltan columnas: {req - set(df.columns)}")
                    return

                df = df.head(int(N_MAX)).copy()

                # 1) Zero-shot
                st.markdown("### 1) Tipo de incidente (Zero-shot)")
                out = []
                bar1 = st.progress(0, text="Clasificando…")
                for i, row in df.iterrows():
                    text = str(row["relato"])
                    try:
                        zs = hf_zero_shot(text, headers=headers)
                    except Exception as e:
                        st.error(f"Zero-shot error: {e}"); zs = []
                    tipo, score = (zs[0][0], float(zs[0][1])) if zs else ("otros", 0.0)
                    out.append({**row.to_dict(), "tipo_predicho": tipo, "score_tipo": round(score, 3)})
                    bar1.progress(int((i + 1) / len(df) * 100))
                bar1.empty()
                df = pd.DataFrame(out)

                # 2) NER + regex
                st.markdown("### 2) Entidades (NER) + Montos/Fechas (Regex)")
                ents_col, montos_col, fechas_col = [], [], []
                bar2 = st.progress(0, text="Extrayendo…")
                for i, row in df.iterrows():
                    text = str(row["relato"])
                    ents = hf_ner(text, headers=headers)
                    # Mantener PER/ORG/LOC sin repetir (orden estable)
                    ents_filtered = [w for (w, t, _s) in ents if t in ("ORG", "PER", "LOC")]
                    ents_col.append(", ".join(dict.fromkeys(ents_filtered)))
                    fields = extract_fields(text)
                    montos_col.append(", ".join(fields["montos"]))
                    fechas_col.append(", ".join(fields["fechas"]))
                    bar2.progress(int((i + 1) / len(df) * 100))
                bar2.empty()

                df["entidades"] = ents_col
                df["montos"] = montos_col
                df["fechas"] = fechas_col

                # 3) Riesgo (reglas)
                st.markdown("### 3) Prioridad (Reglas de riesgo)")
                risks = []
                for _, row in df.iterrows():
                    raw_dates = row["fechas"].split(", ") if row["fechas"] else []
                    risks.append(risk_score(row["tipo_predicho"], str(row["relato"]), raw_dates))
                df["riesgo"] = risks

                # Resumen
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Conteo por tipo**")
                    resumen = df["tipo_predicho"].value_counts().rename_axis("tipo_predicho").reset_index(name="count")
                    st.dataframe(resumen)
                with col2:
                    st.write("**Top-10 por riesgo**")
                    st.dataframe(df.sort_values(["riesgo", "score_tipo"], ascending=False).head(10))

                # 4) Tabla final + descarga robusta
                st.markdown("### 4) Tabla completa (ordenada por riesgo)")
                df_final = df.sort_values(["riesgo", "score_tipo"], ascending=False)
                st.dataframe(df_final, use_container_width=True)

                # CSV con BOM (mejor para Excel) + nombre con fecha + key única
                csv_bytes = df_final.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="Descargar resultados (CSV)",
                    data=csv_bytes,
                    file_name=f"incidentes_priorizados_{dt.date.today().isoformat()}.csv",
                    mime="text/csv",
                    key="dl_resultados",
                )

                st.success("App cargada.")
            except Exception as e:
                st.error("❌ Error general durante el análisis. Detalle:")
                st.exception(e)
        else:
            st.info("Sube el CSV para comenzar.")










