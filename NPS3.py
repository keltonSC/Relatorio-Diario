
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import streamlit as st
st.set_page_config(layout="wide", page_title="Dashboard de Leads — CV")
import pandas as pd, numpy as np, plotly.express as px, plotly.graph_objects as go

# ==== Fallback de Corretor e Gestor (Imobiliária) ====
def aplicar_fallback_corretor_gestor(df_src: pd.DataFrame) -> pd.DataFrame:
    df = df_src.copy()
    for c in ["Corretor Final", "Corretor Anterior", "Gestor (Imobiliária)"]:
        if c not in df.columns:
            df[c] = None
    mask_vazio_corr = df["Corretor Final"].astype(str).str.strip().eq("") | df["Corretor Final"].isna()
    df.loc[mask_vazio_corr, "Corretor Final"] = df.loc[mask_vazio_corr, "Corretor Anterior"]
    base_mapa = df.dropna(subset=["Corretor Final", "Gestor (Imobiliária)"]).copy()
    base_mapa = base_mapa[base_mapa["Corretor Final"].astype(str).str.strip() != ""]
    base_mapa = base_mapa[base_mapa["Gestor (Imobiliária)"].astype(str).str.strip() != ""]
    mapa = (base_mapa[["Corretor Final","Gestor (Imobiliária)"]]
            .drop_duplicates("Corretor Final")
            .set_index("Corretor Final")["Gestor (Imobiliária)"].to_dict())
    mask_vazio_gestor = df["Gestor (Imobiliária)"].astype(str).str.strip().eq("") | df["Gestor (Imobiliária)"].isna()
    df.loc[mask_vazio_gestor, "Gestor (Imobiliária)"] = df.loc[mask_vazio_gestor, "Corretor Final"].map(mapa)
    return df
import requests, json, unicodedata, locale, io, re
from requests.adapters import HTTPAdapter, Retry
from datetime import date, timedelta, datetime

import altair as alt
try:
    import mostra_interacoes
except Exception:
    mostra_interacoes = None

# Bind da função de execução das Interações
try:
    # tentamos primeiro 'run', depois 'main'
    from mostra_interacoes import run as interacoes_run  # type: ignore
    _import_inter_error = None
except Exception as _e1:
    try:
        from mostra_interacoes import main as interacoes_run  # type: ignore
        _import_inter_error = None
    except Exception as _e2:
        interacoes_run = None  # fallback: usaremos a função local se existir
        _import_inter_error = f"{_e1} / {_e2}"

    # Tentativas por assinatura:
    tried = []
    try:
        # assinatura (df_leads, df_caps, leaderboards, min_valor_premium=..., owner_col=...)
        tried.append("render_gamificacao_kenlo(df_leads, df_caps, lbs, min_valor_premium, owner_col)")
    except TypeError:
        pass
    try:
        # assinatura com kwargs (mais genérica)
        tried.append("render_gamificacao_kenlo(df_leads=..., df_caps=..., leaderboards=..., premium_threshold=..., owner_col=...)")
    except TypeError:
        pass
    try:
        # assinatura antiga só com df_leads/df_caps
        tried.append("render_gamificacao_kenlo(df_leads, df_caps)")
    except TypeError as e:
        st.error("Não consegui chamar 'render_gamificacao_kenlo' devido à incompatibilidade de assinatura.\n"
                 f"Tentativas: {tried}\nErro final: {e}")

# ================== Locale pt-BR (best effort) ==================
for loc in ('pt_BR.UTF-8','pt_BR.utf8','pt_BR','Portuguese_Brazil.1252',''):
    try:
        locale.setlocale(locale.LC_TIME, loc); break
    except: pass

# ================== Helpers genéricos ==================
def strip_accents_lower(s: str) -> str:
    if s is None: return ""
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    return s.lower()

RE_NO_SHOW = re.compile(r"(não compareceu|nao compareceu|no-?show)", re.I)
RE_AGUARDANDO = re.compile(r"\baguardando atendimento\b", re.I)
RE_AGUARDANDO_CORR = re.compile(r"(aguardando atendimento do corretor|aguardando.*corretor)", re.I)

TARGET_ORDER_DISPLAY = ["Novo","Em Atendimento","Em Proposta","Analise de Credito","Venda","Arquivado"]
LOOKUP_STATUS = {
    strip_accents_lower("Novo"): "Novo",
    strip_accents_lower("Em Atendimento"): "Em Atendimento",
    strip_accents_lower("Em Proposta"): "Em Proposta",
    strip_accents_lower("Analise de Credito"): "Analise de Credito",
    strip_accents_lower("Venda"): "Venda",
    strip_accents_lower("Arquivado"): "Arquivado"
}
CAT_TARGET = pd.CategoricalDtype(categories=TARGET_ORDER_DISPLAY, ordered=True)

def normalize_status(x):
    key = strip_accents_lower(x) if isinstance(x, str) else x
    return LOOKUP_STATUS.get(key, x if isinstance(x, str) else "Não informado")

# ===== Corretor robust extractor (para API de Leads) =====
def _extract_name_any(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    if isinstance(v, dict):
        for k in ("nome","name","usuario","user","nome_completo","full_name","responsavel","responsavel_nome"):
            val = v.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        for k in ("data","attributes"):
            if k in v and isinstance(v[k], dict):
                for kk in ("nome","name","usuario","user","nome_completo","full_name"):
                    val = v[k].get(kk)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
        return None
    if isinstance(v, list) and v:
        for it in v:
            nm = _extract_name_any(it)
            if nm: return nm
        return None
    return None

def _coalesce_broker(df_raw: pd.DataFrame) -> pd.Series:
    cand_cols = [c for c in [
        "corretor_final","corretorFinal","corretor","responsavel","responsavel_nome","usuario","user",
        "corretor_anterior","corretorAnterior","corretor_anterior_nome","corretor_antes","ultimo_corretor","ultimo_responsavel"
    ] if c in df_raw.columns]
    if not cand_cols:
        return pd.Series([None]*len(df_raw))
    out = pd.Series([None]*len(df_raw), index=df_raw.index, dtype="object")
    for c in cand_cols:
        vals = df_raw[c].apply(_extract_name_any)
        mask = out.isna() & vals.notna()
        out.loc[mask] = vals.loc[mask]
    out = out.astype("object").apply(lambda s: (s or "").strip() or None)
    return out

# ===== Export helpers =====
def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

# === Funil visual centralizado ===
def _funnel_uniform_centered(stages, title="", height_scale=1.0):
    if not stages:
        st.info("Sem dados para o funil."); return
    labels = [s["label"] for s in stages]; n = len(labels)
    widths = list(reversed(range(1, n+1))); maxw = max(widths)
    left_pad = [ (maxw - w)/2 for w in widths ]
    txt = [ f'{s["value"]} ({s["pct"]:.1f}%)' for s in stages ]
    base_h = 60*len(labels)+60; fig_h = int(base_h * height_scale)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=left_pad, y=labels, orientation='h', marker=dict(color='rgba(0,0,0,0)'), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Bar(x=widths, y=labels, orientation='h', text=txt, textposition='inside', insidetextanchor='middle', textangle=0, texttemplate='%{text}', marker=dict(line=dict(width=0)), name=title))
    fig.update_layout(barmode='stack', title=title, height=fig_h, xaxis=dict(visible=False), yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ================== HTTP session & caches ==================
@st.cache_resource(show_spinner=False)
def get_http_session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10))
    return s

@st.cache_data(show_spinner=False, ttl=300)
def cv_list_period_cached(sub, email_auth, token_auth, dt_ini, dt_fim, page_size):
    return cv_list_period(sub, email_auth, token_auth, dt_ini, dt_fim, max_pages=1, page_size=page_size)

@st.cache_data(show_spinner=False, ttl=300)
def cv_fetch_tarefas_cached(sub, email_auth, token_auth, limit=2000, offset=0):
    return _cv_fetch_lead_tarefas(sub, email_auth, token_auth, limit, offset)

# ================== LEADS (CV) ==================
def _leads_fetch_raw(sub: str, email_auth: str, token_auth: str, limit=2000, offset=0):
    url = f"https://{sub}.cvcrm.com.br/api/cvio/lead"; params = {"limit": limit, "offset": offset}
    sess = get_http_session()
    try:
        r = sess.get(url, headers={"email": email_auth, "token": token_auth}, params=params, timeout=60)
        try: return r.status_code, r.json(), r.text
        except Exception: pass
        qp = dict(params); qp.update({"email": email_auth, "token": token_auth})
        r2 = sess.get(url, params=qp, timeout=60)
        try: return r2.status_code, r2.json(), r2.text
        except Exception: return r2.status_code, None, r2.text
    except Exception as e:
        return 0, None, f"exception: {e}"

def _leads_extract_rows(payload):
    if payload is None: return []
    if isinstance(payload, list): return payload
    if isinstance(payload, dict):
        for key in ("data","items","results","conteudo","leads","content","registros"):
            v = payload.get(key)
            if isinstance(v, list) and v:
                return v
        if "items" in payload and isinstance(payload["items"], dict):
            v = payload["items"].get("data")
            if isinstance(v, list): return v
        if "retorno" in payload and isinstance(payload["retorno"], dict):
            for key in ("data","items","conteudo","leads","results","content","registros"):
                v = payload["retorno"].get(key)
                if isinstance(v, list): return v
    return []

def cv_list_period(sub: str, email_auth: str, token_auth: str, dt_ini: str, dt_fim: str, max_pages: int = 1, page_size: int = 2000):
    http, payload, raw = _leads_fetch_raw(sub, email_auth, token_auth, limit=page_size, offset=0)
    st.session_state["__leads_diag__"] = {"http": http,"top_keys": list(payload.keys()) if isinstance(payload, dict) else (["<list>"] if isinstance(payload, list) else ["<none>"])}
    rows = _leads_extract_rows(payload)
    if not rows: return pd.DataFrame()
    df_raw = pd.DataFrame(rows)

    def pick_col(candidates): return next((c for c in candidates if c in df_raw.columns), None)
    col_dt   = pick_col(["data_cad", "dataCriacao", "data", "created_at"])
    col_stat = pick_col(["situacao", "status"])
    col_emp  = pick_col(["empreendimento", "empreendimento_nome"])

    if col_stat == "situacao":
        situacao_series = df_raw["situacao"].apply(lambda s: s.get("nome") if isinstance(s, dict) else s)
    elif col_stat:
        situacao_series = df_raw[col_stat]
    else:
        situacao_series = pd.Series([None]*len(df_raw))

    def _first_empreendimento(x):
        try:
            if isinstance(x, list) and x: return x[0].get("nome")
        except Exception: pass
        return None
    if col_emp == "empreendimento":
        empreendimento_series = df_raw["empreendimento"].apply(_first_empreendimento)
    elif col_emp:
        empreendimento_series = df_raw[col_emp]
    else:
        empreendimento_series = pd.Series([None]*len(df_raw))

    data_series = pd.to_datetime(df_raw[col_dt], errors="coerce") if col_dt else pd.to_datetime(pd.Series([None]*len(df_raw)), errors="coerce")
    def pick_series(name_options):
        col = pick_col(name_options)
        return df_raw[col] if col else pd.Series([None]*len(df_raw))

    df = pd.DataFrame({
        "Imobiliária": pick_series(["imobiliaria","Imobiliária","gestor","Gestor","equipe","Equipe"]),
        "Imobiliária Anterior": pick_series(["imobiliaria_anterior","Imobiliária Anterior","gestor_anterior","Gestor Anterior","equipe_anterior","Equipe Anterior"]),
        "Data Primeiro Cadastro": data_series,
        "Situação": situacao_series,
        "Empreendimento": empreendimento_series,
        "Momento do Lead": pick_series(["momento"]),
        "Possibilidade de Venda": pick_series(["score"]),
        "Gerente": pick_series(["gerente"]),
        "Origem do Lead": pick_series(["origem"]),
        "Corretor": pick_series(["corretor"]),
        "Corretor Anterior": pick_series(["corretor_anterior"]),
        "Corretor Final": _coalesce_broker(df_raw),
        # também carregamos possíveis colunas "anteriores" caso precise exibir/uso futuro
        "corretor_anterior": df_raw.get("corretor_anterior", pd.Series([None]*len(df_raw))),
        "corretorAnterior": df_raw.get("corretorAnterior", pd.Series([None]*len(df_raw))),
        "corretor_anterior_nome": df_raw.get("corretor_anterior_nome", pd.Series([None]*len(df_raw))),
        "corretor_antes": df_raw.get("corretor_antes", pd.Series([None]*len(df_raw))),
        "ultimo_corretor": df_raw.get("ultimo_corretor", pd.Series([None]*len(df_raw))),
        "ultimo_responsavel": df_raw.get("ultimo_responsavel", pd.Series([None]*len(df_raw))),
    })

    # ---- Fallback de corretor para Arquivados ----
    try:
        base_cols = list(df_raw.columns)  # type: ignore[name-defined]
    except Exception:
        base_cols = list(df.columns)
    cand_prev = [
        "corretor_anterior","corretorAnterior","corretor_anterior_nome",
        "corretor_antes","ultimo_corretor","ultimo_responsavel"
    ]
    prev_cols = [c for c in cand_prev if c in base_cols]
    if prev_cols:
        prev_series = pd.Series([None]*len(df), index=df.index, dtype="object")
        for c in prev_cols:
            try:
                col_vals = (df_raw[c] if c in df_raw.columns else df[c])  # type: ignore[name-defined]
            except Exception:
                col_vals = df.get(c, pd.Series([None]*len(df), index=df.index))
            vals = col_vals.apply(_extract_name_any)
            prev_series = prev_series.where(prev_series.notna(), vals)

        mask_arch = (df["Situacao (Target)"] == "Arquivado") & (
            df["Corretor Final"].isna() | (df["Corretor Final"].astype[str] == "")
        )
        # corrigir acesso a astype[str] -> astype(str)
        mask_arch = (df["Situacao (Target)"] == "Arquivado") & (
            df["Corretor Final"].isna() | (df["Corretor Final"].astype(str) == "")
        )
        df.loc[mask_arch, "Corretor Final"] = df.loc[mask_arch, "Corretor Final"].combine_first(prev_series.loc[mask_arch])

    broker_cols = [c for c in ["corretor_final","corretorFinal","corretor","responsavel","responsavel_nome","usuario","user",
                                "corretor_anterior","corretorAnterior","corretor_anterior_nome","corretor_antes","ultimo_corretor","ultimo_responsavel"] if c in df_raw.columns]
    like_cols = [c for c in df_raw.columns if re.search(r"(corret|respons)", c, flags=re.I)]
    samples = {}
    for bc in sorted(set(broker_cols) | set(like_cols)):
        try: samples[bc] = df_raw[bc].dropna().head(5).tolist()
        except Exception: samples[bc] = []
    st.session_state["__leads_diag__"]["broker_columns"] = broker_cols
    st.session_state["__leads_diag__"]["broker_like_columns"] = like_cols
    st.session_state["__leads_diag__"]["broker_samples"] = samples
    try:
        st.session_state["__leads_diag__"]["arquivados_sem_corretor"] = int(((df["Situacao (Target)"]=="Arquivado") & (df["Corretor Final"].isna() | (df["Corretor Final"].astype(str)==""))).sum())
    except Exception:
        st.session_state["__leads_diag__"]["arquivados_sem_corretor"] = None

    df["Situacao (Target)"] = pd.Categorical(df["Situação"].apply(normalize_status).astype("object"), dtype=CAT_TARGET)

    # ---- Fallback de Corretor Final para Arquivados (usa colunas "anteriores") ----
    base_cols = list(df_raw.columns) if isinstance(df_raw, pd.DataFrame) else list(df.columns)
    cand_prev = ["corretor_anterior","corretorAnterior","corretor_anterior_nome","corretor_antes","ultimo_corretor","ultimo_responsavel"]
    prev_cols = [c for c in cand_prev if c in base_cols]
    if prev_cols:
        prev_series = pd.Series([None]*len(df), index=df.index, dtype="object")
        for c in prev_cols:
            try:
                col_vals = (df_raw[c] if c in df_raw.columns else df[c])
            except Exception:
                col_vals = df.get(c, pd.Series([None]*len(df), index=df.index))
            vals = col_vals.apply(_extract_name_any)
            prev_series = prev_series.where(prev_series.notna(), vals)

        mask_arch = (df["Situacao (Target)"] == "Arquivado") & (df["Corretor Final"].isna() | (df["Corretor Final"].astype(str).str.strip() == ""))
        df.loc[mask_arch, "Corretor Final"] = df.loc[mask_arch, "Corretor Final"].combine_first(prev_series.loc[mask_arch])


    df["Semana Início"] = df["Data Primeiro Cadastro"].dt.to_period('W').apply(lambda r: r.start_time).dt.date
    df["Data Semana"]  = df["Semana Início"]
    df["Mês"] = df["Data Primeiro Cadastro"].dt.to_period('M').astype(str)
    df["Situacao (Target)"] = pd.Categorical(df["Situação"].apply(normalize_status).astype("object"), dtype=CAT_TARGET)

    # ---- Fallback de Corretor Final para Arquivados (usa colunas "anteriores") ----
    base_cols = list(df_raw.columns) if isinstance(df_raw, pd.DataFrame) else list(df.columns)
    cand_prev = ["corretor_anterior","corretorAnterior","corretor_anterior_nome","corretor_antes","ultimo_corretor","ultimo_responsavel"]
    prev_cols = [c for c in cand_prev if c in base_cols]
    if prev_cols:
        prev_series = pd.Series([None]*len(df), index=df.index, dtype="object")
        for c in prev_cols:
            try:
                col_vals = (df_raw[c] if c in df_raw.columns else df[c])
            except Exception:
                col_vals = df.get(c, pd.Series([None]*len(df), index=df.index))
            vals = col_vals.apply(_extract_name_any)
            prev_series = prev_series.where(prev_series.notna(), vals)

        mask_arch = (df["Situacao (Target)"] == "Arquivado") & (df["Corretor Final"].isna() | (df["Corretor Final"].astype(str).str.strip() == ""))
        df.loc[mask_arch, "Corretor Final"] = df.loc[mask_arch, "Corretor Final"].combine_first(prev_series.loc[mask_arch])


    try:
        dt_i = pd.to_datetime(dt_ini); dt_f = pd.to_datetime(dt_fim) + pd.Timedelta(days=1)
        before = len(df); df = df[(df["Data Primeiro Cadastro"]>=dt_i) & (df["Data Primeiro Cadastro"]<dt_f)].copy()
        after = len(df); st.session_state["__leads_diag__"]["filtrados_por_data"] = before - after
    except Exception: pass
    return df

# (Facebook Ads section removida)
# ================== VISITAS (tarefas CV) ==================
def _cv_get_creds_visitas():
    for sect in ("cv_atendimento", "cv_leads"):
        try:
            sec = st.secrets[sect]; return sect, sec["subdomain"], sec["email"], sec["token"]
        except Exception: continue
    raise RuntimeError("Adicione [cv_atendimento] ou [cv_leads] no .streamlit/secrets.toml")

def _cv_fetch_lead_tarefas(sub, email_auth, token_auth, limit=2000, offset=0):
    endpoint = f"https://{sub}.cvcrm.com.br/api/cvio/lead_tarefas"; params = {"limit": limit, "offset": offset}
    sess = get_http_session()
    r = sess.get(endpoint, headers={"email": email_auth, "token": token_auth}, params=params, timeout=60)
    try: data = r.json()
    except Exception:
        qp = dict(params); qp.update({"email": email_auth, "token": token_auth})
        r = sess.get(endpoint, params=qp, timeout=60); data = r.json()
    rows = []
    if isinstance(data, dict):
        for key in ("data","tarefas","results","items","content"):
            if key in data and isinstance(data[key], list):
                rows = data[key]; break
        if not rows and isinstance(data.get("items"), dict):
            rows = data["items"].get("data", [])
    elif isinstance(data, list): rows = data
    return pd.DataFrame(rows), r.status_code

def _vis_normalize_fast(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df.copy()
    df = df.copy()
    for c in ("data","data_conclusao","data_cancelamento"):
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    if "tipo" in df.columns:
        mask_vis = df["tipo"].astype(str).str.contains("visita", case=False, na=False)
        df = df[mask_vis].copy()
    map_sit = {"P":"Agendada","C":"Realizada","X":"Cancelada","A":"Atrasada"}
    if "situacao" in df.columns:
        df["status_normalizado"] = df["situacao"].astype(str).str.upper().map(map_sit).fillna("Outros")
    else:
        df["status_normalizado"] = "Outros"
    titulo = df.get("titulo", pd.Series("", index=df.index)).astype(str).str.lower()
    motivo = df.get("motivo_cancelamento", pd.Series("", index=df.index)).astype(str).str.lower()
    cancelada = df["data_cancelamento"].notna() if "data_cancelamento" in df.columns else pd.Series(False, index=df.index)
    concluida = df["data_conclusao"].notna() if "data_conclusao" in df.columns else pd.Series(False, index=df.index)
    noshow   = cancelada & motivo.str.contains(RE_NO_SHOW, regex=True)
    reag     = titulo.str.contains("reagend", na=False)
    status_final = np.where(noshow, "No-show", np.where(cancelada, "Cancelada", np.where(concluida, "Realizada", np.where(reag, "Reagendada", "Pendente"))))
    df["status_visita"] = df["status_normalizado"]; df["status_final"]  = status_final
    return df

def _vis_coorte_fast(df_vis: pd.DataFrame, dt_ini: date, dt_fim: date) -> pd.DataFrame:
    if df_vis.empty or "data" not in df_vis.columns: return df_vis.iloc[0:0].copy()
    m = (df_vis["data"].dt.date >= dt_ini) & (df_vis["data"].dt.date <= dt_fim)
    keep_cols = [c for c in ["data","data_conclusao","data_cancelamento","status_final","responsavel","idlead","idtarefa","nome_empreendimento","titulo","hora","motivo_cancelamento","motivo_concluido","lead","telefone","email","idempreendimento"] if c in df_vis.columns]
    return df_vis.loc[m, keep_cols].copy()

def _vis_kpis_from_coorte(coorte: pd.DataFrame) -> dict:
    base_ag = len(coorte)
    if base_ag == 0:
        return {"visitas_agendadas":0,"visitas_realizadas":0,"visitas_canceladas":0,"visitas_noshow":0,"visitas_reagendadas":0,"visitas_pendentes":0,"taxa_conversao":0.0}
    vc = coorte["status_final"].value_counts(); realizadas = int(vc.get("Realizada", 0))
    return {"visitas_agendadas": base_ag,"visitas_realizadas": realizadas,"visitas_canceladas": int(vc.get("Cancelada", 0)),"visitas_noshow": int(vc.get("No-show", 0)),
            "visitas_reagendadas": int(vc.get("Reagendada", 0)),"visitas_pendentes": int(vc.get("Pendente", 0)),"taxa_conversao": round((realizadas/max(base_ag,1))*100, 1)}

def _vis_render_funil(coorte: pd.DataFrame):
    base_ag = max(len(coorte), 1)
    vc = coorte["status_final"].value_counts() if not coorte.empty else pd.Series(dtype=int)
    serie = pd.Series({"Agendada (coorte)": base_ag if base_ag>0 else 0,"Realizada": int(vc.get("Realizada",0)),
                       "Cancelada": int(vc.get("Cancelada",0)),"No-show":  int(vc.get("No-show",0)),"Reagendada": int(vc.get("Reagendada",0)),"Pendente": int(vc.get("Pendente",0))})
    st.bar_chart(serie)

def _vis_render_kpis_corretor(coorte: pd.DataFrame):
    if coorte.empty or "responsavel" not in coorte.columns:
        st.info("Sem dados de 'responsavel' para agrupar."); return
    col_resp = "responsavel"
    value_col = "idtarefa" if "idtarefa" in coorte.columns else ("idlead" if "idlead" in coorte.columns else coorte.columns[0])
    piv = (coorte.pivot_table(index=col_resp, columns="status_final", values=value_col, aggfunc="count", fill_value=0)
           .reindex(columns=["Realizada","Cancelada","No-show","Reagendada","Pendente"], fill_value=0).reset_index())
    ag_por_resp = coorte.groupby(col_resp)[value_col].count().rename("Agendada (coorte)").reset_index()
    piv = ag_por_resp.merge(piv, on=col_resp, how="left").fillna(0)
    if "data_conclusao" in coorte.columns and "data" in coorte.columns:
        realizados = coorte[coorte["status_final"]=="Realizada"].copy()
        if not realizados.empty:
            realizados["sla_dias"] = (realizados["data_conclusao"] - realizados["data"]).dt.total_seconds()/86400.0
            sla = realizados.groupby(col_resp)["sla_dias"].mean().round(1)
            piv = piv.merge(sla.rename("SLA médio (dias)"), on=col_resp, how="left")
        else:
            piv["SLA médio (dias)"] = None
    else:
        piv["SLA médio (dias)"] = None
    st.dataframe(piv.sort_values(by=["Realizada"], ascending=[False]), use_container_width=True)

def _vis_render_listas(df_vis: pd.DataFrame):
    if df_vis.empty:
        st.info("Sem registros de visita."); return
    def ultima_data_evento(row):
        dts = [];
        for c in ("data","data_conclusao","data_cancelamento"):
            if c in df_vis.columns and pd.notna(row.get(c)): dts.append(row[c])
        return max(dts) if dts else pd.NaT
    dfv = df_vis.copy(); dfv["__ultima_data"] = dfv.apply(ultima_data_evento, axis=1)
    latest = (dfv.sort_values(["idlead","__ultima_data"], ascending=[True, False]).drop_duplicates(subset=["idlead"], keep="first"))
    cols1 = [c for c in ["idlead","lead","telefone","email","responsavel","status_visita","status_final","data","data_conclusao","data_cancelamento","idempreendimento","nome_empreendimento","titulo"] if c in latest.columns]
    st.markdown("**Leads com visita (1 linha por lead — último status)**"); st.dataframe(latest[cols1], use_container_width=True)
    cols2 = [c for c in ["idlead","lead","telefone","email","responsavel","status_visita","status_final","data","data_conclusao","data_cancelamento","idempreendimento","nome_empreendimento","titulo","idtarefa","hora","motivo_cancelamento","motivo_concluido"] if c in dfv.columns]
    st.markdown("**Todas as visitas (uma linha por visita)**")
    st.dataframe(dfv[cols2].sort_values(by=["idlead","data","data_conclusao","data_cancelamento"], ascending=[True, True, True, True]), use_container_width=True)

def contagens_status_final(coorte: pd.DataFrame) -> dict:
    if coorte.empty: return dict(Agendada=0, Realizada=0, Cancelada=0, **{"No-show":0, "Reagendada":0, "Pendente":0})
    vc = coorte["status_final"].value_counts(); base = len(coorte)
    out = {"Agendada": base,"Realizada": int(vc.get("Realizada", 0)),"Cancelada": int(vc.get("Cancelada", 0)),"No-show": int(vc.get("No-show", 0)),"Reagendada": int(vc.get("Reagendada", 0)),"Pendente": int(vc.get("Pendente", 0))}
    return out

def multiselect_all(label, options): return st.multiselect(label, options, default=options) if options else []

def build_filter(df, colname, expander_title, label, icon=""):
    if colname in df.columns and not df.empty:
        with st.sidebar.expander(f"{icon} {expander_title}", expanded=False):
            opts = sorted([o for o in df[colname].dropna().unique()])
            selected = multiselect_all(label, opts)
            if selected and len(selected) < len(opts):
                return df[df[colname].isin(selected)]
    return df

# ===== Consolidated KPI builder (CV Corretores) =====
def build_consolidated_kpi_table(lead_stats: pd.DataFrame, visitas_por_resp: pd.DataFrame) -> pd.DataFrame:
    cols_lead = [c for c in ["Corretor Final","Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Análise de crédito","Venda","Arquivado"]
                 if lead_stats is not None and c in lead_stats.columns]
    cols_vis  = [c for c in ["Corretor Final","Visita agendada","Visita realizada","SLA médio (dias)"]
                 if visitas_por_resp is not None and c in visitas_por_resp.columns]
    df_lead = lead_stats[cols_lead].copy() if cols_lead else pd.DataFrame(columns=["Corretor Final"])
    df_vis  = visitas_por_resp[cols_vis].copy() if cols_vis else pd.DataFrame(columns=["Corretor Final"])
    cons = pd.merge(df_lead, df_vis, on="Corretor Final", how="outer")
    for c in ["Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Análise de crédito","Venda","Arquivado","Visita agendada","Visita realizada"]:
        if c not in cons.columns: cons[c] = 0
        cons[c] = cons[c].fillna(0).astype(int)
    if "SLA médio (dias)" not in cons.columns: cons["SLA médio (dias)"] = None
    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)): return "-"
        if isinstance(v, float): return f"{v:.1f}"
        return str(v)
    cons["KPIs (concat)"] = (
        "Leads=" + cons["Leads"].astype(str) + " | "
        + "AguCli=" + cons["Aguardando atendimento"].astype(str) + " | "
        + "AguCorr=" + cons["Aguardando atendimento do corretor"].astype(str) + " | "
        + "EmAtend=" + cons["Em atendimento"].astype(str) + " | "
        + "VisAg=" + cons["Visita agendada"].astype(str) + " | "
        + "VisReal=" + cons["Visita realizada"].astype(str) + " | "
        + "AnalCred=" + cons["Análise de crédito"].astype(str) + " | "
        + "Venda=" + cons["Venda"].astype(str) + " | "
        + "Arquiv=" + cons["Arquivado"].astype(str) + " | "
        + "SLAmed=" + cons["SLA médio (dias)"].map(fmt)
    )
    cols_final = ["Corretor Final","Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Visita agendada","Visita realizada","Análise de crédito","Venda","Arquivado","SLA médio (dias)","KPIs (concat)"]
    cols_exist = [c for c in cols_final if c in cons.columns]
    return cons[cols_exist]

# ================== App ==================


# ========= Helpers de Normalização (Gestor e Arquivados) =========
def _cv_normalize_owners_generic(df):
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in df.columns: return n
            for k, orig in cols.items():
                if k == n.lower(): return orig
        return None

    col_corretor     = col('Corretor Final','Corretor','responsavel','Responsável','corretor')
    col_corretor_ant = col('Corretor Anterior','corretor_anterior','responsavel_anterior','Responsável Anterior','ultimo_corretor','ultimo_responsavel')
    col_imo          = col('Imobiliária','imobiliaria','Equipe','equipe','Gestor','gestor')
    col_imo_ant      = col('Imobiliária Anterior','imobiliaria_anterior','Equipe Anterior','equipe_anterior','Gestor Anterior','gestor_anterior')
    col_status       = col('Situacao (Target)','Situação','situacao','status')
    col_arq          = col('Arquivado','arquivado')

    # flag de arquivado
    is_archived = pd.Series(False, index=df.index)
    if col_arq:
        is_archived = df[col_arq].astype(str).str.lower().isin(['1','true','sim','yes'])
    elif col_status:
        is_archived = df[col_status].astype(str).str.contains('Arquivado', case=False, na=False)

    # Corretor Efetivo
    if col_corretor:
        base_cor = df[col_corretor]
        ant_cor  = df[col_corretor_ant] if col_corretor_ant else None
        df['Corretor Efetivo'] = base_cor.astype(object).where(
            ~(is_archived & (base_cor.isna() | (base_cor.astype(str).str.strip() == ''))),
            ant_cor if ant_cor is not None else base_cor
        )
    elif col_corretor_ant:
        df['Corretor Efetivo'] = df[col_corretor_ant]
    else:
        df['Corretor Efetivo'] = None

    # Imobiliária Efetiva
    if col_imo or col_imo_ant:
        now = df[col_imo] if col_imo else None
        ant = df[col_imo_ant] if col_imo_ant else None
        if now is not None:
            df['Imobiliária Efetiva'] = now.astype(object).where(
                ~(is_archived & (now.isna() | (now.astype(str).str.strip() == ''))),
                ant if ant is not None else now
            )
        elif ant is not None:
            df['Imobiliária Efetiva'] = ant
        else:
            df['Imobiliária Efetiva'] = None
    else:
        df['Imobiliária Efetiva'] = None
    return df

def _apply_gestor_filter(df):
    import pandas as pd
    sel = st.session_state.get("cv_gestor_sel") or []
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or not sel:
        return df
    col = "Imobiliária Efetiva" if "Imobiliária Efetiva" in df.columns else ("Imobiliária" if "Imobiliária" in df.columns else None)
    if col is None:
        return df
    return df[df[col].astype(str).isin([str(x) for x in sel])].copy()


def main():
    st.title("📊 Dashboard de Leads — CV")

    # Top-level tabs
    main_tabs = st.tabs(["🏗 CV CRM"])
    # ------------------- CV CRM -------------------
    with main_tabs[0]:
        # ---------- Tabs CV ----------
        # ---------- Sidebar CV ----------
        with st.sidebar:
            st.markdown("### ⚙️ Controles (CV)")
            sub_default = st.secrets.get("cv_leads", {}).get("subdomain", "selectbroker")
            sub = st.text_input("Subdomínio do CV", value=sub_default, key="cv_sub")
            email_auth = st.secrets.get("cv_leads", {}).get("email", "")
            token_auth = st.secrets.get("cv_leads", {}).get("token", "")
            dt_ini = st.date_input("De (Leads/Visitas)", value=(date.today().replace(day=1)), key="cv_dt_ini")
            dt_fim = st.date_input("Até (Leads/Visitas)", value=date.today(), key="cv_dt_fim")
            page_size = st.number_input("Page size (CV)", min_value=100, max_value=5000, value=2000, step=100, key="cv_page_size")
            reload_leads = st.button("🔄 Carregar/Atualizar LEADS", key="cv_reload")
            st.markdown("---"); # ---------- LEADS load ----------
        if reload_leads or "cv_df_leads" not in st.session_state:
            with st.spinner("Carregando LEADS do CV..."):
                df_leads = cv_list_period_cached(sub, email_auth, token_auth, dt_ini.strftime("%Y-%m-%d"), dt_fim.strftime("%Y-%m-%d"), int(page_size))
            st.session_state["cv_df_leads"] = df_leads
        df = st.session_state["cv_df_leads"]

        # ---------- VISITAS auto (para coortes/kpis) ----------
        vis_ini_auto = date.today().replace(day=1); vis_fim_auto = date.today()
        if "cv_vis_auto" not in st.session_state:
            try:
                sect, sub_vis, email_vis, token_vis = _cv_get_creds_visitas()
                df_raw_vis, http_vis = cv_fetch_tarefas_cached(sub_vis, email_vis, token_vis)
                df_vis_auto = _vis_normalize_fast(df_raw_vis)
                coorte_vis_auto = _vis_coorte_fast(df_vis_auto, vis_ini_auto, vis_fim_auto)
                k_vis_auto = _vis_kpis_from_coorte(coorte_vis_auto)
                st.session_state["cv_vis_auto"] = {"df": df_vis_auto, "coorte": coorte_vis_auto, "kpis": k_vis_auto, "http": http_vis, "sect": sect}
            except Exception as e:
                st.session_state["cv_vis_auto"] = {"df": pd.DataFrame(), "coorte": pd.DataFrame(), "kpis": {"visitas_agendadas":0,"visitas_realizadas":0,"visitas_canceladas":0,"visitas_noshow":0,"visitas_reagendadas":0,"visitas_pendentes":0,"taxa_conversao":0.0}, "http": None, "sect": None}
        vis_auto = st.session_state["cv_vis_auto"]

        # ---------- Normalização para Gestor/Arquivados ----------
        df_norm = _cv_normalize_owners_generic(df)

        # Normalizar colunas textuais potencialmente vindas como dict/list
        for _col in ["Imobiliária Efetiva","Imobiliária","Imobiliária Anterior","Corretor Efetivo","Corretor Final","Corretor Anterior"]:
            if _col in df_norm.columns:
                df_norm[_col] = df_norm[_col].apply(_extract_name_any)

        st.session_state["cv_df_leads_norm"] = df_norm
        # ---------- Filtros de KPIs ----------
        with st.sidebar:
            st.markdown("---"); st.markdown("### 🧰 Filtros de KPIs (CV)")
            # Filtro por Gestor (Imobiliária)
            ser = df_norm.get("Imobiliária Efetiva", pd.Series(dtype=object)).dropna().apply(_extract_name_any).astype(str)
            gestor_opts = sorted(ser.unique().tolist())
            st.multiselect("Gestor (Imobiliária)", gestor_opts, default=gestor_opts, key="cv_gestor_sel")
            emp_opts = sorted([o for o in (df["Empreendimento"].dropna().unique().tolist() if not df.empty else [])])
            corr_leads = set(df["Corretor Final"].dropna().unique().tolist() if not df.empty else [])
            coorte_hdr_tmp = vis_auto["coorte"]
            corr_vis   = set(coorte_hdr_tmp["responsavel"].dropna().unique().tolist() if not coorte_hdr_tmp.empty and "responsavel" in coorte_hdr_tmp.columns else [])
            corr_opts  = sorted(list(corr_leads.union(corr_vis)))
            sel_emp = st.multiselect("Empreendimento (Leads)", emp_opts, default=emp_opts, key="cv_sel_emp")
            sel_corr = st.multiselect("Corretor (Leads/Visitas)", corr_opts, default=corr_opts, key="cv_sel_corr")

        df_kpi = df_norm.copy()
        sel_gestor = st.session_state.get("cv_gestor_sel") or []
        if df_kpi.size and sel_gestor:
            col_g = "Imobiliária Efetiva" if "Imobiliária Efetiva" in df_kpi.columns else ("Imobiliária" if "Imobiliária" in df_kpi.columns else None)
            if col_g:
                df_kpi = df_kpi[df_kpi[col_g].astype(str).isin([str(x) for x in sel_gestor])]
        if df_kpi.size and sel_emp: df_kpi = df_kpi[df_kpi["Empreendimento"].isin(sel_emp)]
        if df_kpi.size and sel_corr: df_kpi = df_kpi[df_kpi["Corretor Final"].isin(sel_corr)]

        st.session_state["cv_df_leads_view"] = df_kpi
        total_leads_hdr = len(df_kpi)
        def _clean_hdr(s): return strip_accents_lower(str(s)) if pd.notna(s) else ""
        sraw_hdr = df_kpi.get("Situação", pd.Series([""]*len(df_kpi))).apply(_clean_hdr) if not df_kpi.empty else pd.Series(dtype="object")
        stgt_hdr = df_kpi.get("Situacao (Target)", pd.Series([""]*len(df_kpi))) if not df_kpi.empty else pd.Series(dtype="object")
        aguard_atend_hdr = int((sraw_hdr.str.contains(RE_AGUARDANDO) & ~sraw_hdr.str.contains("corretor")).sum()) if not df_kpi.empty else 0
        em_atend_hdr = int((stgt_hdr == "Em Atendimento").sum()) if not df_kpi.empty else 0
        arquivados_hdr = int((stgt_hdr == "Arquivado").sum()) if not df_kpi.empty else 0

        spL, c1, c2, c3, c4, spR = st.columns([0.7, 2.6, 2.6, 2.6, 2.6, 0.7])
        with c1: st.metric("Total de Leads", total_leads_hdr)
        with c2: st.metric("Aguardando Atendimento", aguard_atend_hdr)
        with c3: st.metric("Em Atendimento", em_atend_hdr)
        with c4: st.metric("Arquivados", arquivados_hdr)

        # ---------- Coorte de visitas alinhada ao cabeçalho ----------
        df_vis_base_hdr = vis_auto.get("df", pd.DataFrame())
        try:
            dt_ini_hdr = pd.to_datetime(dt_ini).date(); dt_fim_hdr = pd.to_datetime(dt_fim).date()
        except Exception:
            dt_ini_hdr = date.today().replace(day=1); dt_fim_hdr = date.today()
        coorte_hdr = _vis_coorte_fast(df_vis_base_hdr, dt_ini_hdr, dt_fim_hdr)
        if not coorte_hdr.empty:
            if sel_corr and "responsavel" in coorte_hdr.columns: coorte_hdr = coorte_hdr[coorte_hdr["responsavel"].isin(sel_corr)]
            if sel_emp and "nome_empreendimento" in coorte_hdr.columns: coorte_hdr = coorte_hdr[coorte_hdr["nome_empreendimento"].isin(sel_emp)]

        # ---------- Sub-abas CV ----------
        cv_tabs = st.tabs(["🧾 Resumo", "👥 Corretores","👣 Visitas", "💬 Interações"])

        # ====== CV Resumo ======
        with cv_tabs[0]:
            st.subheader("🧾 Resumo (CV)")
            dt_ini_vis_sel = st.session_state.get("cv_vis_ini", date.today().replace(day=1))
            dt_fim_vis_sel = st.session_state.get("cv_vis_fim", date.today())
            df_vis_base = vis_auto.get("df", pd.DataFrame())
            coorte_resumo = _vis_coorte_fast(df_vis_base, dt_ini_vis_sel, dt_fim_vis_sel)
            if df_kpi.empty and coorte_resumo.empty:
                st.info("Sem dados no período atual (após filtros).")
                diag = st.session_state.get("__leads_diag__", {})
                with st.expander("🩺 Diagnóstico da API (Leads)"):
                    st.write(f"HTTP: **{diag.get('http','?')}**"); st.write(f"Chaves de topo no JSON: **{diag.get('top_keys',[])}**")
                    if "filtrados_por_data" in diag: st.write(f"Filtrados por data: **{diag['filtrados_por_data']}**")
                    if "broker_columns" in diag: st.write(f"Colunas de corretor detectadas: **{diag['broker_columns']}**")
                    if "broker_samples" in diag: st.json({"broker_samples": diag["broker_samples"]})
            else:
                st.caption(f"📌 Neste Resumo: **Leads** seguem o período/filtros da barra lateral. **Visitas** usam as **mesmas datas da aba Visitas** ({dt_ini_vis_sel} → {dt_fim_vis_sel}), sem filtros adicionais.")
                st.markdown("### 📋 Indicadores (ordem solicitada)")
                total_leads = len(df_kpi)
                def _clean(s): return strip_accents_lower(str(s)) if pd.notna(s) else ""
                sraw = df_kpi.get("Situação", pd.Series([""]*len(df_kpi))).apply(_clean) if not df_kpi.empty else pd.Series(dtype="object")
                stgt = df_kpi.get("Situacao (Target)", pd.Series([""]*len(df_kpi))) if not df_kpi.empty else pd.Series(dtype="object")
                # corrigindo possível erro acima (callable/lambda)
                if not df_kpi.empty:
                    stgt = df_kpi["Situacao (Target)"]
                aguard_cli = int((sraw.str.contains(RE_AGUARDANDO) & ~sraw.str.contains("corretor")).sum()) if not df_kpi.empty else 0
                aguard_corretor  = int((sraw.str.contains(RE_AGUARDANDO_CORR)).sum()) if not df_kpi.empty else 0
                em_atendimento   = int((stgt == "Em Atendimento").sum()) if not df_kpi.empty else 0
                analise_credito  = int((stgt == "Analise de Credito").sum()) if not df_kpi.empty else 0
                vendas           = int((stgt == "Venda").sum()) if not df_kpi.empty else 0
                arquivados       = int((stgt == "Arquivado").sum()) if not df_kpi.empty else 0
                counts_vis = contagens_status_final(coorte_resumo)
                indicadores = [
                    ("Aguardando atendimento (leads)", aguard_cli),
                    ("Aguardando atendimento do corretor (leads)", aguard_corretor),
                    ("Em atendimento (leads)", em_atendimento),
                    ("Visita agendada (tarefas/coorte período)", counts_vis["Agendada"]),
                    ("Visita realizada (tarefas/coorte período)", counts_vis["Realizada"]),
                    ("Análise de crédito (leads)", analise_credito),
                    ("Venda (leads)", vendas),
                    ("Arquivado (leads)", arquivados),
                ]
                df_lista = pd.DataFrame(indicadores, columns=["Indicador","Quantidade"])
                df_lista["% (sobre total de leads)"] = ((df_lista["Quantidade"] / max(total_leads,1)) * 100).round(1)
                cL, cM, cR = st.columns([1,6,1])
                with cM:
                    styled = (df_lista.style.hide(axis="index").set_properties(**{"text-align":"center"}).set_table_styles([{"selector":"th", "props":"text-align:center;"}]).format({"% (sobre total de leads)": "{:.1f}%"}))
                    st.markdown(styled.to_html(), unsafe_allow_html=True)
                st.markdown("---"); st.markdown(f"### 🔻 Funil de Visitas — coorte {dt_ini_vis_sel} a {dt_fim_vis_sel}")
                base_vis = max(counts_vis["Agendada"], 1)
                stages_vis = [
                    {"label":"Visita agendada (coorte)", "value": counts_vis["Agendada"], "pct": (counts_vis["Agendada"]/base_vis)*100},
                    {"label":"Visita realizada",         "value": counts_vis["Realizada"], "pct": (counts_vis["Realizada"]/base_vis)*100},
                    {"label":"Cancelada",                "value": counts_vis["Cancelada"], "pct": (counts_vis["Cancelada"]/base_vis)*100},
                    {"label":"No-show",                  "value": counts_vis["No-show"],   "pct": (counts_vis["No-show"]/base_vis)*100},
                    {"label":"Reagendada",               "value": counts_vis["Reagendada"],"pct": (counts_vis["Reagendada"]/base_vis)*100},
                    {"label":"Pendente",                 "value": counts_vis["Pendente"],  "pct": (counts_vis["Pendente"]/base_vis)*100},
                ]
                fL, fC, fR = st.columns([1, 8, 1])
                with fC: _funnel_uniform_centered(stages_vis, title="", height_scale=1.4)

        # ====== CV Corretores ======
        with cv_tabs[1]:
            st.subheader("👥 Corretores (CV)")

            # --- Alerta para Arquivados sem corretor ---
            diag = st.session_state.get("__leads_diag__", {})
            miss_arch = diag.get("arquivados_sem_corretor", 0) or 0
            if miss_arch > 0:
                st.warning(f"⚠️ Há {miss_arch} lead(s) **Arquivado(s)** sem **Corretor Final**. A API pode trazer esse nome em um campo alternativo (ex.: *corretor_anterior*). Veja abaixo os candidatos e me diga qual coluna usar como fallback definitivo.")
                with st.expander("🔎 Colunas candidatas relacionadas a corretor/responsável (amostras do payload)"):
                    like_cols = diag.get("broker_like_columns", []) or diag.get("broker_columns", [])
                    samples = diag.get("broker_samples", {})
                    if not like_cols:
                        st.info("Não encontrei colunas candidatas além das já usadas. Posso inspecionar o payload completo se você preferir.")
                    else:
                        for c in sorted(like_cols):
                            st.markdown(f"**{c}**")
                            st.write(samples.get(c, [])[:5])

            alias = st.secrets.get("corretor_alias", {})
            def alias_map(s): raw = str(s or ""); return alias.get(raw, raw)

            # ---- LEADS por corretor ----
            if df_kpi.empty:
                leads_por_corretor = pd.DataFrame(columns=["Corretor Final","Leads"])
                lead_stats = pd.DataFrame(columns=["Corretor Final","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Análise de crédito","Venda","Arquivado"])
            else:
                leads_por_corretor = (df_kpi["Corretor Final"].fillna("—").map(alias_map).value_counts().rename_axis("Corretor Final").reset_index(name="Leads"))
                def _clean(s): return strip_accents_lower(str(s)) if pd.notna(s) else ""
                tmp = df_kpi.copy(); tmp["Corretor Final"] = tmp["Corretor Final"].map(alias_map); tmp["sit_clean"] = tmp["Situação"].apply(_clean)
                def cnt_mask(mask): return (tmp[mask].groupby("Corretor Final")["Situação"].count().rename("val"))
                lead_stats = pd.DataFrame({"Corretor Final": leads_por_corretor["Corretor Final"]})
                m_agu_cli   = tmp["sit_clean"].str.contains(RE_AGUARDANDO) & ~tmp["sit_clean"].str.contains("corretor")
                m_agu_corr  = tmp["sit_clean"].str.contains(RE_AGUARDANDO_CORR)
                m_em_atend  = (tmp["Situacao (Target)"] == "Em Atendimento")
                m_analise   = (tmp["Situacao (Target)"] == "Analise de Credito")
                m_venda     = (tmp["Situacao (Target)"] == "Venda")
                m_arquiv    = (tmp["Situacao (Target)"] == "Arquivado")
                def add_col(df_out, serie, nome):
                    serie = serie.reindex(df_out["Corretor Final"]).fillna(0).astype(int); df_out[nome] = serie.values
                add_col(lead_stats, cnt_mask(m_agu_cli),  "Aguardando atendimento")
                add_col(lead_stats, cnt_mask(m_agu_corr), "Aguardando atendimento do corretor")
                add_col(lead_stats, cnt_mask(m_em_atend), "Em atendimento")
                add_col(lead_stats, cnt_mask(m_analise),  "Análise de crédito")
                add_col(lead_stats, cnt_mask(m_venda),    "Venda")
                add_col(lead_stats, cnt_mask(m_arquiv),   "Arquivado")
                lead_stats = leads_por_corretor.merge(lead_stats, on="Corretor Final", how="left")

            def visitas_por_responsavel_from_coorte(coorte: pd.DataFrame) -> pd.DataFrame:
                if coorte.empty or "responsavel" not in coorte.columns:
                    return pd.DataFrame(columns=["Corretor Final","Visita agendada","Visita realizada","SLA médio (dias)"])
                val_col = "idtarefa" if "idtarefa" in coorte.columns else ("idlead" if "idlead" in coorte.columns else coorte.columns[0])
                base = coorte.groupby("responsavel")[val_col].count().rename("Visita agendada").reset_index()
                realiz = (coorte[coorte["status_final"]=="Realizada"].groupby("responsavel")[val_col].count().rename("Visita realizada").reset_index())
                dfR = base.merge(realiz, on="responsavel", how="left")
                dfR["Visita realizada"] = dfR["Visita realizada"].fillna(0).astype(int)
                if "data_conclusao" in coorte.columns and "data" in coorte.columns:
                    realizados = coorte[coorte["status_final"]=="Realizada"].copy()
                    if not realizados.empty:
                        realizados["sla_dias"] = (realizados["data_conclusao"] - realizados["data"]).dt.total_seconds()/86400.0
                        sla = realizados.groupby("responsavel")["sla_dias"].mean().round(1)
                        dfR = dfR.merge(sla.rename("SLA médio (dias)"), on="responsavel", how="left")
                    else:
                        dfR["SLA médio (dias)"] = None
                else:
                    dfR["SLA médio (dias)"] = None
                dfR = dfR.rename(columns={"responsavel":"Corretor Final"})
                dfR["Corretor Final"] = dfR["Corretor Final"].map(alias_map)
                return dfR

            # Visitas (coorte do mês atual auto, filtrada pelos KPIs quando aplicável)
            coorte = vis_auto["coorte"]
            if not coorte.empty:
                if sel_corr and "responsavel" in coorte.columns: coorte = coorte[coorte["responsavel"].isin(sel_corr)]
                if sel_emp and "nome_empreendimento" in coorte.columns: coorte = coorte[coorte["nome_empreendimento"].isin(sel_emp)]
            visitas_por_resp = visitas_por_responsavel_from_coorte(coorte)

            # ---- Visão consolidada (primeiro) ----
            cons_df = build_consolidated_kpi_table(lead_stats, visitas_por_resp)
            tmp_leads = df_kpi.groupby("Corretor Final")["Empreendimento"].count().rename("Leads_total").reset_index() if not df_kpi.empty else pd.DataFrame(columns=["Corretor Final","Leads_total"])
            tmp_vis   = coorte.groupby("responsavel")["idlead"].nunique().rename("Leads_com_visita").reset_index() if (not coorte.empty and "responsavel" in coorte.columns and "idlead" in coorte.columns) else pd.DataFrame(columns=["responsavel","Leads_com_visita"])
            if not tmp_vis.empty: tmp_vis = tmp_vis.rename(columns={"responsavel":"Corretor Final"})
            cons_df = cons_df.merge(tmp_leads, on="Corretor Final", how="left").merge(tmp_vis, on="Corretor Final", how="left")
            cons_df["Leads_total"] = cons_df["Leads_total"].fillna(cons_df["Leads"]).fillna(0).astype(int)
            cons_df["Leads_com_visita"] = cons_df["Leads_com_visita"].fillna(0).astype(int)
            cons_df["Leads_sem_visita"] = (cons_df["Leads_total"] - cons_df["Leads_com_visita"]).clip(lower=0).astype(int)
            cons_df["% Leads c/ visita"] = (cons_df["Leads_com_visita"] / cons_df["Leads_total"].replace(0,1) * 100).round(1)
            cons_df["Visita % Realizada"] = (cons_df["Visita realizada"] / cons_df["Visita agendada"].replace(0,1) * 100).round(1)

            st.markdown("### 📋 Visão consolidada (KPIs em colunas)")
            cc1, cc2 = st.columns([2,2])
            with cc1:
                q = st.text_input("🔎 Buscar corretor", value="", key="cv_search_corr")
            with cc2:
                sort_opt = st.selectbox("Ordenar por", ["Visita realizada","Leads","% Leads c/ visita","Visita % Realizada","SLA médio (dias)"], index=0, key="cv_sort_opt")
            if q:
                cons_view = cons_df[cons_df["Corretor Final"].astype(str).str.contains(q, case=False, na=False)].copy()
            else:
                cons_view = cons_df.copy()
            asc = False if sort_opt in ["Visita realizada","Leads","% Leads c/ visita","Visita % Realizada"] else True
            if sort_opt in cons_view.columns:
                cons_view = cons_view.sort_values(by=sort_opt, ascending=asc, na_position="last")

            cons_display = cons_view
            if "KPIs (concat)" in cons_view.columns:
                kidx = list(cons_view.columns).index("KPIs (concat)")
                cons_display = cons_view.iloc[:, :kidx]
            st.dataframe(cons_display, use_container_width=True)
            st.download_button("⬇️ Baixar (Consolidado) .xlsx", data=dataframe_to_excel_bytes(cons_display), file_name="cv_corretores_consolidado.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ====== Tabela Principal (imutável) ======
            if "Corretor Final" not in lead_stats.columns: lead_stats["Corretor Final"] = pd.Series(dtype="object")
            if "Corretor Final" not in visitas_por_resp.columns: visitas_por_resp["Corretor Final"] = pd.Series(dtype="object")
            lead_stats["__k"] = lead_stats["Corretor Final"].astype(str).apply(strip_accents_lower)
            visitas_por_resp["__k"] = visitas_por_resp["Corretor Final"].astype(str).apply(strip_accents_lower)
            full = lead_stats.merge(visitas_por_resp, on="__k", how="outer", suffixes=("_lead", "_vis")).drop(columns=["__k"], errors="ignore")
            if "Corretor Final_lead" in full.columns and "Corretor Final_vis" in full.columns:
                full["Corretor Final"] = full["Corretor Final_lead"].combine_first(full["Corretor Final_vis"]); full = full.drop(columns=["Corretor Final_lead","Corretor Final_vis"])
            elif "Corretor Final_lead" in full.columns: full = full.rename(columns={"Corretor Final_lead":"Corretor Final"})
            elif "Corretor Final_vis" in full.columns: full = full.rename(columns={"Corretor Final_vis":"Corretor Final"})
            for c in ["Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Visita agendada","Visita realizada","Análise de crédito","Venda","Arquivado"]:
                if c not in full.columns: full[c] = 0
            if "SLA médio (dias)" not in full.columns: full["SLA médio (dias)"] = None
            for c in ["Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Visita agendada","Visita realizada","Análise de crédito","Venda","Arquivado"]:
                full[c] = full[c].fillna(0).astype(int)
            if not df_kpi.empty and "Corretor Final" in df_kpi.columns:
                leads_por_cor = df_kpi.groupby("Corretor Final")["Empreendimento"].count().rename("Leads_total")
                ids_com_visita = pd.Series(dtype=int)
                if not coorte.empty and "responsavel" in coorte.columns and "idlead" in coorte.columns:
                    ids_com_visita = coorte.groupby("responsavel")["idlead"].nunique().rename("Leads_com_visita")
                m = leads_por_cor.to_frame().merge(ids_com_visita, left_index=True, right_index=True, how="left").fillna(0).astype(int)
                m["Leads_sem_visita"] = (m["Leads_total"] - m["Leads_com_visita"]).clip(lower=0); m = m.reset_index().rename(columns={"index":"Corretor Final"})
                full = full.merge(m[["Corretor Final","Leads_sem_visita"]], on="Corretor Final", how="left"); full["Leads_sem_visita"] = full["Leads_sem_visita"].fillna(0).astype(int)
            else:
                full["Leads_sem_visita"] = 0
            full = full.sort_values(by=["Leads","Visita realizada"], ascending=[False, False])
            cols_ordem = ["Corretor Final","Leads","Aguardando atendimento","Aguardando atendimento do corretor","Em atendimento","Visita agendada","Visita realizada","Análise de crédito","Venda","Arquivado","Leads_sem_visita","SLA médio (dias)"]
            cols_exist = [c for c in cols_ordem if c in full.columns]
            st.markdown("### 📄 Tabela principal (imutável)")
            st.dataframe(full[cols_exist], use_container_width=True)
            st.download_button("⬇️ Baixar (Principal) .xlsx", data=dataframe_to_excel_bytes(full[cols_exist]), file_name="cv_corretores_principal.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ---- Drilldown por corretor ----
            st.markdown("### 🔎 Drilldown rápido por corretor")
            sel_cor = st.selectbox("Selecione um corretor", sorted(cons_df["Corretor Final"].dropna().unique().tolist()) if not cons_df.empty else [], key="cv_drill_sel")
            if sel_cor:
                leads_cor = df_kpi[df_kpi["Corretor Final"].astype(str) == sel_cor].copy()
                cols_leads_show = [c for c in ["Data Primeiro Cadastro","Empreendimento","Situação","Situacao (Target)","Origem do Lead","Gerente","Momento do Lead"] if c in leads_cor.columns]
                st.markdown("**Leads do período**")
                if cols_leads_show:
                    st.dataframe(leads_cor[cols_leads_show].sort_values(by="Data Primeiro Cadastro", ascending=False), use_container_width=True)
                coorte_sel = coorte_hdr.copy()
                if not coorte_sel.empty and "responsavel" in coorte_sel.columns:
                    coorte_sel = coorte_sel[coorte_sel["responsavel"].astype(str) == sel_cor]
                st.markdown("**Visitas (coorte do cabeçalho)**")
                if not coorte_sel.empty:
                    cols_vis_show = [c for c in ["data","status_final","nome_empreendimento","titulo","data_conclusao","data_cancelamento","idlead","idtarefa"] if c in coorte_sel.columns]
                    st.dataframe(coorte_sel[cols_vis_show].sort_values(by=["data","data_conclusao","data_cancelamento"], ascending=[False, False, False]), use_container_width=True)
                else:
                    st.info("Sem visitas para esse corretor no período.")
        # ====== CV VISITAS ======
        with cv_tabs[2]:

            st.subheader("👣 Visitas — Coorte de Agendamento (Tarefas) — CV")
            c1, c2 = st.columns(2)
            with c1: dt_ini_vis = st.date_input("De (coorte VISITAS por DATA)", value=date.today().replace(day=1), key="cv_vis_ini")
            with c2: dt_fim_vis = st.date_input("Até (coorte VISITAS por DATA)", value=date.today(), key="cv_vis_fim")
            ss_key_vis = f"CV_VIS_{dt_ini_vis}_{dt_fim_vis}"
            if ss_key_vis not in st.session_state:
                try:
                    sect, sub_vis, email_vis, token_vis = _cv_get_creds_visitas()
                    df_raw_vis, http_vis = cv_fetch_tarefas_cached(sub_vis, email_vis, token_vis)
                    st.caption(f"Visitas • Credenciais: [{sect}] • /lead_tarefas HTTP {http_vis} • Linhas: {len(df_raw_vis)}")
                    df_vis = _vis_normalize_fast(df_raw_vis); coorte_vis = _vis_coorte_fast(df_vis, dt_ini_vis, dt_fim_vis)
                    k_vis = _vis_kpis_from_coorte(coorte_vis); st.session_state[ss_key_vis] = {"df": df_vis, "coorte": coorte_vis, "kpis": k_vis}
                except Exception as e:
                    st.error("Erro ao carregar Visitas (tarefas). Verifique secrets e conectividade."); st.exception(e); st.stop()
            vis = st.session_state[ss_key_vis]; k = vis["kpis"]
            kc = st.columns(6)
            with kc[0]: st.metric("Agendadas (coorte)", k["visitas_agendadas"])
            with kc[1]: st.metric("Realizadas",          k["visitas_realizadas"])
            with kc[2]: st.metric("Canceladas",          k["visitas_canceladas"])
            with kc[3]: st.metric("No-show",             k["visitas_noshow"])
            with kc[4]: st.metric("Reagendadas",         k["visitas_reagendadas"])
            with kc[5]: st.metric("Conversão (%)",       k["taxa_conversao"])
            st.markdown("---"); st.subheader("🔻 Funil por Coorte (Visitas)"); _vis_render_funil(vis["coorte"])
            st.subheader("👥 KPIs por Corretor (Visitas)"); _vis_render_kpis_corretor(vis["coorte"])
            st.subheader("🧾 Listas de Leads (Visitas)"); _vis_render_listas(vis["df"])


        with cv_tabs[3]:
            st.subheader("💬 Interações — API CV (nova)")
            if interacoes_run is None:
                try:
                    render_interacoes_cv_tab()
                    st.caption("ℹ️ Usando fallback interno (mostra_interacoes.py não encontrado).")
                except Exception as _e:
                    st.error("Falha ao renderizar Interações (fallback). Verifique o 'mostra_interacoes.py' ou credenciais do CV.")
                    st.exception(_e)
            else:
                interacoes_run()

if __name__ == "__main__":
    main()

# ------- Fallback mínimo: Interações -------
def render_interacoes_cv_tab():
    st.info("Fallback de Interações ativo. Mostrando leads em cache (se houver).")
    df_leads = st.session_state.get("cv_df_leads_view") or st.session_state.get("cv_df_leads_norm") or st.session_state.get("cv_df_leads")
    if isinstance(df_leads, pd.DataFrame) and not df_leads.empty:
        st.dataframe(df_leads.head(200), use_container_width=True)
    else:
        st.caption("Sem dataframe de leads em cache. Carregue na visão principal.")