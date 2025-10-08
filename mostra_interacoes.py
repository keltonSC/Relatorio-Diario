#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import requests
import altair as alt
from datetime import date, timedelta, datetime

# ⚠️ Não chame st.set_page_config aqui (para não conflitar quando importado pelo NPS.py)

def run():
    # ===== Credenciais (st.secrets) =====
    sec = st.secrets.get("cv_atendimento") or st.secrets.get("cv_leads")
    if not sec:
        st.error("⚠️ Configure [cv_atendimento] ou [cv_leads] em .streamlit/secrets.toml")
        st.stop()
    sub, email, token = sec["subdomain"], sec["email"], sec["token"]
    BASE = f"https://{sub}.cvcrm.com.br"
    URL  = f"{BASE}/api/v1/comercial/leads/interacoes"
    HEADERS = {"email": email, "token": token}

    st.caption("Período e controles desta aba são independentes das demais.")

    # ===== Sidebar (com chaves exclusivas) =====
    st.sidebar.header("Período (Interações)")
    fim = st.sidebar.date_input("Fim", value=date.today(), key="inter_fim")
    ini = st.sidebar.date_input("Início", value=fim - timedelta(days=60), key="inter_ini")

    col_b1, col_b2 = st.sidebar.columns(2)
    btn_fetch = col_b1.button("🔄 Atualizar dados (API)", use_container_width=True, key="inter_btn_fetch")
    btn_force = col_b2.button("♻️ Forçar atualização", use_container_width=True, key="inter_btn_force",
                              help="Ignora cache do período")

    aplicar_fallback = st.sidebar.checkbox(
        "Aplicar fallback do corretor (recomendado)",
        value=True, key="inter_cb_fallback",
        help="Quando 'corretor' vier vazio, usa o corretor da última interação do mesmo lead."
    )

    # ===== Cache por período (namespacing em session_state) =====
    @st.cache_data(show_spinner=False, ttl=300)
    def _peek_total_pages():
        r = requests.get(URL, headers=HEADERS, params={"pagina": 1, "limit": 30}, timeout=30)
        payload = r.json() if r.content else {}
        return int(payload.get("total_de_paginas") or 1)

    @st.cache_data(show_spinner=False, ttl=300)
    def _fetch_page(pagina:int, limit:int=30):
        r = requests.get(URL, headers=HEADERS, params={"pagina": pagina, "limit": limit}, timeout=60)
        pl = r.json() if r.content else {}
        return pl.get("dados", []) or []

    @st.cache_data(show_spinner=True, ttl=300)
    def fetch_period_descending(ini_d: date, fim_d: date, limit:int=30):
        total_pages = _peek_total_pages()
        rows = []
        ini_ts = pd.Timestamp(ini_d)
        fim_ts = pd.Timestamp(fim_d) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        prog = st.progress(0.0, text=f"Coletando… (páginas {total_pages} → 1)")
        for i, page in enumerate(range(total_pages, 0, -1), start=1):
            chunk = _fetch_page(page, limit=limit)
            if not chunk:
                prog.progress(min(1.0, i/total_pages), text=f"Pág {page} vazia")
                continue

            dfp = pd.DataFrame(chunk)
            base_col = "referencia_data" if "referencia_data" in dfp.columns else ("data_cad" if "data_cad" in dfp.columns else None)
            if base_col:
                dfp["_Data"] = pd.to_datetime(dfp[base_col], errors="coerce")
                dfp_orig = dfp.copy()
                dfp = dfp[(dfp["_Data"].notna()) & (dfp["_Data"] >= ini_ts) & (dfp["_Data"] <= fim_ts)]
                if not dfp.empty:
                    rows.append(dfp)
                if dfp_orig["_Data"].notna().any() and dfp_orig["_Data"].min() < ini_ts:
                    prog.progress(1.0, text=f"Parou na pág {page} — antes do início")
                    break
            else:
                rows.append(dfp)

            prog.progress(min(1.0, i/total_pages), text=f"Coletando pág {page-1}")

        if rows:
            df = pd.concat(rows, ignore_index=True).sort_values("_Data", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame()
        return df, total_pages

    def carregar_periodo(ini_d: date, fim_d: date, force: bool=False):
        key = f"{ini_d.isoformat()}_{fim_d.isoformat()}"
        if force or st.session_state.get("inter_period_key") != key or "inter_df" not in st.session_state:
            df, total_pages = fetch_period_descending(ini_d, fim_d, limit=30)
            st.session_state["inter_df"] = df
            st.session_state["inter_period_key"] = key
            st.session_state["inter_total_pages"] = total_pages
            st.session_state["inter_ultima_origem"] = "API"
            st.session_state["inter_ultima_coleta_ts"] = datetime.now()
        else:
            st.session_state.setdefault("inter_ultima_origem", "cache")
            st.session_state.setdefault("inter_ultima_coleta_ts", None)

    # ===== Disparo da coleta (com chaves isoladas) =====
    if btn_fetch or btn_force or ("inter_df" not in st.session_state):
        with st.spinner("Coletando dados do período…"):
            carregar_periodo(ini, fim, force=btn_force)

    df = st.session_state.get("inter_df", pd.DataFrame())
    total_pages = st.session_state.get("inter_total_pages", 0)

    st.caption(
        f"Período: **{ini:%d/%m/%Y} → {fim:%d/%m/%Y}** • "
        f"Páginas totais: **{total_pages}** • "
        f"Linhas carregadas: **{len(df):,}**".replace(",", ".")
    )
    origem = st.session_state.get("inter_ultima_origem", "cache")
    ts = st.session_state.get("inter_ultima_coleta_ts", None)
    st.markdown(
        f"**Origem:** {('🟢 API' if origem=='API' else '🟡 cache')}"
        + (f" • Atualizado em: {ts:%d/%m/%Y %H:%M:%S}" if ts else "")
    )

    if df.empty:
        st.info("Sem dados no recorte. Ajuste o período e clique em **Atualizar dados (API)**.")
        st.stop()

    # ===== Fallback corretor (não conflita com outras abas) =====
    def aplicar_fallback_corretor(df_src: pd.DataFrame) -> pd.DataFrame:
        dfv = df_src.copy()
        for c in ["corretor", "corretor_interacao", "idlead"]:
            if c not in dfv.columns:
                dfv[c] = None
        dft = dfv.sort_values("_Data", ascending=False).copy()
        dft["corretor_fallback"] = dft["corretor"].where(dft["corretor"].astype(str).str.strip().ne(""))
        dft["corretor_fallback"] = dft["corretor_fallback"].fillna(
            dft["corretor_interacao"].where(dft["corretor_interacao"].astype(str).str.strip().ne(""))
        )
        mapa = (
            dft.dropna(subset=["idlead"])
               .drop_duplicates("idlead")[["idlead","corretor_fallback"]]
               .set_index("idlead")["corretor_fallback"].to_dict()
        )
        mask_vazio = dfv["corretor"].astype(str).str.strip().eq("") | dfv["corretor"].isna()
        dfv.loc[mask_vazio, "corretor"] = dfv.loc[mask_vazio, "idlead"].map(mapa)
        return dfv

    if "_Data" not in df.columns:
        df["_Data"] = pd.NaT
    df["_Dia"] = pd.to_datetime(df["_Data"], errors="coerce").dt.date
    df_view = aplicar_fallback_corretor(df) if aplicar_fallback else df

    # ===== KPIs =====
    total = len(df_view)
    dias = df_view["_Dia"].nunique()
    media_dia = round(total / max(1, dias), 2)
    corretores_ativos = df_view["corretor"].astype(str).nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de interações", total)
    c2.metric("Dias com interação", dias)
    c3.metric("Média por dia", media_dia)
    c4.metric("Corretores ativos", corretores_ativos)

    # ===== Ranking simples =====
    st.subheader("🏆 Ranking — Total de interações por corretor")
    rank = (df_view.groupby("corretor").size().sort_values(ascending=False)
            .rename("interações").reset_index())
    for i, row in rank.head(20).iterrows():
        st.write(f"{i+1}. **{row['corretor'] or '—'}** — {int(row['interações'])} interações")

    # ===== Tabelas lado a lado =====
    st.subheader("Top corretores por interações")
    colA, colB = st.columns(2)

    _df = df_view.copy()
    _df["corretor"] = _df["corretor"].fillna("").replace("", "—")

    with colA:
        tab_top = (_df.groupby("corretor", dropna=False).size()
                   .reset_index(name="interações")
                   .sort_values("interações", ascending=False, ignore_index=True))
        tab_top["interações"] = tab_top["interações"].astype(int)
        st.dataframe(tab_top, use_container_width=True, hide_index=True)

    with colB:
        if "idlead" in _df.columns:
            tab_leads = (_df.groupby("corretor", dropna=False)["idlead"].nunique()
                         .reset_index(name="leads_únicos")
                         .sort_values("leads_únicos", ascending=False, ignore_index=True))
            tab_leads["leads_únicos"] = tab_leads["leads_únicos"].astype(int)
            st.dataframe(tab_leads, use_container_width=True, hide_index=True)
        else:
            st.info("Coluna 'idlead' não disponível para leads únicos.")

    # ===== Gráfico Interações por dia =====st.subheader("📈 Interações por dia (total)")

    # Agrupa por dia e ordena
    by_day_full = df_view.groupby("_Dia").size().reset_index(name="interacoes").sort_values("_Dia")

    # Garante no máximo 31 dias (pega os 31 dias mais recentes do período)
    if len(by_day_full) > 31:
        by_day = by_day_full.tail(31).copy()
    else:
        by_day = by_day_full.copy()

    # Preenche dias faltantes para não "pular" nenhum rótulo
    if not by_day.empty:
        full_range = pd.date_range(start=by_day["_Dia"].min(), end=by_day["_Dia"].max(), freq="D").date
        by_day = pd.DataFrame({"_Dia": full_range}).merge(by_day, on="_Dia", how="left").fillna({"interacoes": 0})
    else:
        by_day = by_day_full.copy()

    # Cria rótulo categórico só com o dia (string) e lista de ordenação
    by_day["DiaStr"] = pd.to_datetime(by_day["_Dia"]).dt.strftime("%d")
    tick_labels = by_day["DiaStr"].tolist()

    # Número de dias e largura dinâmica (~28px por dia, mínimo 620px)
    n_days = len(by_day)
    chart_width = max(620, int(n_days * 28))

    base_total = alt.Chart(by_day).encode(
        x=alt.X(
            "DiaStr:N",
            title=None,
            sort=tick_labels,  # mantém a ordem cronológica
            axis=alt.Axis(grid=False, labelAngle=0, labelOverlap=False)
        ),
        y=alt.Y("interacoes:Q", title=None, axis=alt.Axis(grid=False)),
        tooltip=[
            alt.Tooltip("_Dia:T", title="Data", format="%d/%m"),
            alt.Tooltip("interacoes:Q", title="Interações")
        ]
    )

    bar_size = 20 if n_days <= 15 else (16 if n_days <= 22 else 12)

    chart_total = (
        base_total.mark_bar(size=bar_size) +
        base_total.mark_text(dy=-6).encode(text="interacoes:Q")
    ).configure_view(stroke=None).configure_axis(grid=False).configure_legend(disable=True)      .properties(height=260, width=chart_width)

    st.altair_chart(chart_total, use_container_width=True)

    # ===== Um dia específico =====st.subheader("👥 Interações por corretor (somente 1 dia)")
    dias_disp = sorted(df_view["_Dia"].dropna().unique(), reverse=True)
    dia_escolhido = st.selectbox("Escolha o dia", dias_disp, index=0,
                                 format_func=lambda d: d.strftime("%d/%m/%Y"), key="inter_dia_sel")
    df_dia = df_view[df_view["_Dia"] == dia_escolhido].copy()

    if df_dia.empty:
        st.info("Sem interações nesse dia.")
    else:
        # Agrupa por corretor e aplica rótulo: PrimeiroNome + ÚltimoSobrenome,
        # com exceções específicas solicitadas
        por_corretor = df_dia.groupby("corretor").size().reset_index(name="interacoes")

        def label_nome(nome):
            full = str(nome or "").strip()
            low = full.lower()
            if not full:
                return "—"
            # exceções específicas
            if low == "francisco osterne graça mendonça":
                return "Osterne Graça"
            if low == "francisco anildo mota correia":
                return "Anildo Mota"
            parts = [p for p in full.split() if p]
            if len(parts) == 1:
                return parts[0]
            return f"{parts[0]} {parts[-1]}"

        por_corretor["rotulo"] = por_corretor["corretor"].apply(label_nome)

        n_corr = len(por_corretor)
        chart_width = max(700, int(n_corr * 120))

        base_c = alt.Chart(por_corretor).encode(
            x=alt.X("rotulo:N", title=None,
                    axis=alt.Axis(labelAngle=-25, labelLimit=0, labelOverlap=False)),
            y=alt.Y("interacoes:Q", title=None, axis=alt.Axis(grid=False)),  # sem título lateral
            tooltip=[
                alt.Tooltip("corretor:N", title="Corretor (completo)"),
                alt.Tooltip("interacoes:Q", title="Interações")
            ]
        )
        chart_cor = (base_c.mark_bar() + base_c.mark_text(dy=-6).encode(text="interacoes:Q"))             .configure_view(stroke=None).configure_axis(grid=False)             .properties(height=320, width=chart_width)
        st.altair_chart(chart_cor, use_container_width=True)

    # ===== Amostra do dia =====
    st.subheader("🔎 Amostra do dia selecionado")
    cols_show = ["_Data","corretor","idcorretor","idinteracao","idlead","tipo","situacao","descricao"]
    cols_show = [c for c in cols_show if c in df_dia.columns]
    st.dataframe(df_dia.sort_values("_Data", ascending=False)[cols_show].head(500),
                 use_container_width=True, hide_index=True)

if __name__ == "__main__":
    st.set_page_config(page_title="Visão de Interações (Standalone)", layout="wide")
    st.title("💬 Visão de Interações — Standalone")
    run()
