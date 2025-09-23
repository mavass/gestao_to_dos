# app.py — Gestor de Entregáveis (Streamlit + SQLite)
# Execução: `streamlit run app.py`
# Este app foca apenas em ENTREGÁVEIS (sem reuniões), com prioridades renomeadas
# e ajuste de rótulos conforme solicitado.

import os
import sqlite3
from contextlib import closing
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

import base64
import json
import requests

DB_PATH = 'todos.db'

# -----------------------------
# Banco de Dados
# -----------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo TEXT NOT NULL,
    empresa TEXT CHECK(empresa IN ('Alyvia','Laudite','Arlequim','Pessoal')) NOT NULL DEFAULT 'Pessoal',
    status TEXT CHECK(status IN ('Novo','Em andamento','Bloqueado','Concluído')) NOT NULL DEFAULT 'Novo',
    prioridade TEXT CHECK(prioridade IN ('Urgente','Alta','Média','Baixa','Muito Baixa')) NOT NULL DEFAULT 'Média',
    due_date TEXT,
    planned_for TEXT,
    notas TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_tasks_empresa ON tasks (empresa);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_planned_for ON tasks (planned_for);
CREATE TRIGGER IF NOT EXISTS trg_tasks_updated_at
AFTER UPDATE ON tasks FOR EACH ROW
BEGIN
    UPDATE tasks SET updated_at = datetime('now') WHERE id = old.id;
END;
"""

# Nova tabela: lembretes de longo prazo
SCHEMA_SQL += """
CREATE TABLE IF NOT EXISTS long_term (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    empresa TEXT CHECK(empresa IN ('Alyvia','Laudite','Arlequim','Pessoal')) NOT NULL DEFAULT 'Pessoal',
    pessoa_area TEXT NOT NULL,
    lembrete TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_long_term_empresa ON long_term (empresa);
CREATE INDEX IF NOT EXISTS idx_long_term_pessoa_area ON long_term (pessoa_area);
"""

# Nova tabela: FUPs do Time (nova categoria de inputs)
SCHEMA_SQL += """
CREATE TABLE IF NOT EXISTS fups_team (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    empresa TEXT CHECK(empresa IN ('Alyvia','Laudite','Arlequim','Pessoal')) NOT NULL DEFAULT 'Pessoal',
    pessoa_area TEXT NOT NULL,
    fup TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_fups_empresa ON fups_team (empresa);
CREATE INDEX IF NOT EXISTS idx_fups_pessoa_area ON fups_team (pessoa_area);
"""


# --- util para ler secrets com segurança ---
def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default


    # 1) pegar SHA atual do arquivo (se existir)
    api_base = f"https://api.github.com/repos/{repo}/contents/{repo_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    params = {"ref": branch}
    resp = requests.get(api_base, headers=headers, params=params)
    sha = resp.json().get("sha") if resp.status_code == 200 else None

    # 2) ler binário e enviar
    with open(path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "message": f"chore(db): update {repo_path} via app",
        "content": content_b64,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put = requests.put(api_base, headers=headers, data=json.dumps(payload))
    return put.status_code in (200, 201)

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(drop_and_recreate: bool = False):
    with closing(get_conn()) as conn, conn:
        if drop_and_recreate:
            # Reseta tudo
            conn.executescript("""
            DROP TABLE IF EXISTS tasks;
            DROP TABLE IF EXISTS long_term;
            DROP TABLE IF EXISTS fups_team;
            """)
        conn.executescript(SCHEMA_SQL)


# -----------------------------
# Tasks — CRUD
# -----------------------------

def insert_task(d: Dict[str, Any]) -> int:
    keys = ','.join(d.keys())
    qmarks = ','.join('?' for _ in d)
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(f"INSERT INTO tasks ({keys}) VALUES ({qmarks})", list(d.values()))
        task_id = cur.lastrowid
    sync_db_to_github()
    return task_id

def update_task(task_id: int, d: Dict[str, Any]):
    set_clause = ','.join([f"{k}=?" for k in d.keys()])
    with closing(get_conn()) as conn, conn:
        conn.execute(f"UPDATE tasks SET {set_clause} WHERE id=?", list(d.values()) + [task_id])
    sync_db_to_github()

def delete_task(task_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    sync_db_to_github()

def insert_long_term(empresa: str, pessoa_area: str, lembrete: str) -> int:
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO long_term (empresa, pessoa_area, lembrete) VALUES (?, ?, ?)",
            (empresa, pessoa_area.strip(), lembrete.strip())
        )
        new_id = cur.lastrowid
    sync_db_to_github()
    return new_id

def delete_long_term(item_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM long_term WHERE id=?", (item_id,))
    sync_db_to_github()

def insert_fup(empresa: str, pessoa_area: str, fup: str) -> int:
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO fups_team (empresa, pessoa_area, fup) VALUES (?, ?, ?)",
            (empresa, pessoa_area.strip(), fup.strip())
        )
        new_id = cur.lastrowid
    sync_db_to_github()
    return new_id

def delete_fup(item_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM fups_team WHERE id=?", (item_id,))
    sync_db_to_github()

def fetch_tasks(where: str = '', params: tuple = ()) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        query = "SELECT * FROM tasks" + (f" WHERE {where}" if where else "") + " ORDER BY prioridade, due_date, planned_for, id DESC"
        df = pd.read_sql_query(query, conn, params=params)
    return df


# -----------------------------
# Longo Prazo — CRUD
# -----------------------------

def insert_long_term(empresa: str, pessoa_area: str, lembrete: str) -> int:
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO long_term (empresa, pessoa_area, lembrete) VALUES (?, ?, ?)",
            (empresa, pessoa_area.strip(), lembrete.strip())
        )
        return cur.lastrowid


def delete_long_term(item_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM long_term WHERE id=?", (item_id,))


def fetch_long_term(where: str = "", params: tuple = ()) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        q = "SELECT * FROM long_term" + (f" WHERE {where}" if where else "")
        q += " ORDER BY empresa, pessoa_area, id DESC"
        return pd.read_sql_query(q, conn, params=params)


# -----------------------------
# FUPs do Time — CRUD (nova categoria)
# -----------------------------

def insert_fup(empresa: str, pessoa_area: str, fup: str) -> int:
    with closing(get_conn()) as conn, conn:
        cur = conn.execute(
            "INSERT INTO fups_team (empresa, pessoa_area, fup) VALUES (?, ?, ?)",
            (empresa, pessoa_area.strip(), fup.strip())
        )
        return cur.lastrowid


def delete_fup(item_id: int):
    with closing(get_conn()) as conn, conn:
        conn.execute("DELETE FROM fups_team WHERE id=?", (item_id,))


def fetch_fups(where: str = "", params: tuple = ()) -> pd.DataFrame:
    with closing(get_conn()) as conn:
        q = "SELECT * FROM fups_team" + (f" WHERE {where}" if where else "")
        q += " ORDER BY empresa, pessoa_area, id DESC"
        return pd.read_sql_query(q, conn, params=params)


# -----------------------------
# Utilidades
# -----------------------------
EMPRESAS = ['Alyvia','Laudite','Arlequim','Pessoal']
STATUSES = ['Novo','Em andamento','Bloqueado','Concluído']
PRIORIDADES = ['Urgente','Alta','Média','Baixa','Muito Baixa']


def iso_or_none(d: Optional[date]) -> Optional[str]:
    return d.isoformat() if d else None


def parse_date(s: str) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, '%Y-%m-%d').date()
    except Exception:
        return None


def _get_openai_key() -> str:
    """Lê a chave exclusivamente de st.secrets['OPENAI_API_KEY'].
    No deploy, configure em Settings → Secrets (não no repositório)."""
    try:
        import streamlit as st
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip()
    except Exception:
        return ""


def openai_triage(texto: str) -> Optional[List[Dict[str, Any]]]:
    api_key = _get_openai_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""
Você é um assistente que transforma texto solto em ENTREGÁVEIS (sem reuniões).
Retorne APENAS um JSON com a chave 'tarefas' contendo objetos com as chaves:
- titulo (string até 180 chars)
- empresa (Alyvia|Laudite|Arlequim|Pessoal)
- prioridade (Urgente|Alta|Média|Baixa|Muito Baixa)
- due_date (YYYY-MM-DD ou null)
- planned_for (YYYY-MM-DD ou null)  # 'Começar até'
- notas (string, opcional)

Regras:
- NÃO invente datas.
- NÃO crie reuniões.
- Se não souber empresa, use 'Pessoal'.
Entrada:
---
{texto}
---
Exemplo de saída:
{{"tarefas":[{{"titulo":"Preparar demo","empresa":"Alyvia","prioridade":"Alta","due_date":null,"planned_for":"2025-08-13","notas":""}}]}}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content":"Você retorna APENAS um JSON válido com a chave 'tarefas'."},
                {"role":"user","content":prompt}
            ],
            temperature=0.2
        )
        import json
        data = json.loads(resp.choices[0].message.content)
        tarefas = data.get("tarefas", [])
        out = []
        for t in tarefas:
            out.append({
                "titulo": (t.get("titulo","") or "Entregável sem título").strip()[:180],
                "empresa": t.get("empresa") if t.get("empresa") in EMPRESAS else "Pessoal",
                "status": "Novo",
                "prioridade": t.get("prioridade") if t.get("prioridade") in PRIORIDADES else "Média",
                "due_date": t.get("due_date") or None,
                "planned_for": t.get("planned_for") or None,
                "notas": t.get("notas") or ""
            })
        return out
    except Exception:
        return None


# -----------------------------
# UI Helpers
# -----------------------------

def task_form(defaults: Dict[str, Any], key_prefix: str) -> Dict[str, Any]:
    c1,c2 = st.columns([2,1])
    titulo = c1.text_input('Título', value=defaults.get('titulo',''), key=f'{key_prefix}_titulo')
    empresa = c2.selectbox('Empresa', EMPRESAS, index=EMPRESAS.index(defaults.get('empresa','Pessoal')), key=f'{key_prefix}_empresa')

    c3,c4 = st.columns(2)
    prioridade = c3.selectbox('Prioridade', PRIORIDADES, index=PRIORIDADES.index(defaults.get('prioridade','Média')), key=f'{key_prefix}_prioridade')
    status = c4.selectbox('Status', STATUSES, index=STATUSES.index(defaults.get('status','Novo')), key=f'{key_prefix}_status')

    c6,c7 = st.columns(2)
    due_date = c6.date_input('Prazo (due date)', value=parse_date(defaults.get('due_date')), key=f'{key_prefix}_due', format='YYYY-MM-DD')
    planned_for = c7.date_input('Começar até', value=parse_date(defaults.get('planned_for')), key=f'{key_prefix}_plan', format='YYYY-MM-DD')

    notas = st.text_area('Notas', value=defaults.get('notas',''), key=f'{key_prefix}_notas')
    return {
        'titulo': titulo,
        'empresa': empresa,
        'status': status,
        'prioridade': prioridade,
        'due_date': iso_or_none(due_date) if isinstance(due_date, date) else None,
        'planned_for': iso_or_none(planned_for) if isinstance(planned_for, date) else None,
        'notas': notas or None
    }


def row_actions(row: pd.Series, prefix: str = ""):
    p = f"{prefix}_{row.id}" if prefix else str(row.id)

    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button('✅ Concluir', key=f'done_{p}'):
        update_task(int(row.id), {'status': 'Concluído'})
        st.rerun()

    if c2.button('➡️ Em andamento', key=f'prog_{p}'):
        update_task(int(row.id), {'status': 'Em andamento'})
        st.rerun()

    if c3.button('⏭️ Amanhã', key=f'tmr_{p}'):
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        update_task(int(row.id), {'planned_for': tomorrow})
        st.rerun()

    if c4.button('⚠️ Bloquear', key=f'blk_{p}'):
        update_task(int(row.id), {'status': 'Bloqueado'})
        st.rerun()

    if c5.button('🗑️ Apagar', key=f'del_{p}'):
        delete_task(int(row.id))
        st.rerun()


# -----------------------------
# App
# -----------------------------

st.set_page_config(page_title='Gestor de Entregáveis — MVP', layout='wide')
st.title('📦 Gestor de Entregáveis — MVP')

# Recria tabelas conforme schema atual (sem apagar dados por padrão)
init_db(drop_and_recreate=False)

with st.sidebar:
    # Título para identificar os diferentes inputs
    st.subheader('Triar texto (WhatsApp/E-mail/Ideias)')
    texto = st.text_area('Cole aqui o texto', height=160, key='triage_text')
    if st.button('🤖 Triar (OpenAI, se chave configurada)', key='triage_openai'):
        tarefas = openai_triage(texto) or []
        for t in tarefas:
            insert_task(t)
        st.success(f'{len(tarefas)} tarefa(s) criada(s).')

    st.divider()

    # Título + entrada de Longo Prazo
    st.subheader('🕰️ Longo Prazo — input')
    with st.expander('🕰️ Longo Prazo — novo lembrete', expanded=False):
        c_lp1, c_lp2 = st.columns([1,1])
        empresa_lp = c_lp1.selectbox('Empresa', EMPRESAS, index=EMPRESAS.index('Pessoal'), key='lp_empresa')
        pessoa_area_lp = c_lp2.text_input('Pessoa ou Área', key='lp_pessoa_area', placeholder='Ex.: Dr. João / Comercial / Financeiro')
        lembrete_lp = st.text_area('Lembrete', key='lp_lembrete', height=100, placeholder='Descreva o lembrete / fup de longo prazo')
        if st.button('➕ Adicionar lembrete de longo prazo', key='lp_add'):
            if (pessoa_area_lp or '').strip() and (lembrete_lp or '').strip():
                insert_long_term(empresa_lp, pessoa_area_lp, lembrete_lp)
                st.success('Lembrete de longo prazo adicionado.')
                st.rerun()
            else:
                st.warning('Preencha "Pessoa ou Área" e o "Lembrete".')

    st.divider()

    # Input dedicado para FUPs do Time
    st.subheader('👥 FUPs do Time — input')
    with st.expander('👥 FUPs do Time — novo FUP', expanded=False):
        c_ft1, c_ft2 = st.columns([1,1])
        empresa_ft = c_ft1.selectbox('Empresa', EMPRESAS, index=EMPRESAS.index('Pessoal'), key='ft_empresa')
        pessoa_area_ft = c_ft2.text_input('Pessoa ou Área', key='ft_pessoa_area', placeholder='Ex.: Comercial / Financeiro / Fulano')
        fup_ft = st.text_area('FUP', key='ft_fup', height=100, placeholder='Descreva o FUP (follow-up) do time')
        if st.button('➕ Adicionar FUP', key='ft_add'):
            if (pessoa_area_ft or '').strip() and (fup_ft or '').strip():
                insert_fup(empresa_ft, pessoa_area_ft, fup_ft)
                st.success('FUP adicionado.')
                st.rerun()
            else:
                st.warning('Preencha "Pessoa ou Área" e o "FUP".')

    st.divider()

    # "Novo entregável (manual)" é o ÚLTIMO input da sidebar
    st.subheader('Novo entregável (manual)')
    with st.expander('Novo entregável (manual)', expanded=False):
        defaults = {
            'titulo':'',
            'empresa':'Pessoal',
            'status':'Novo',
            'prioridade':'Média',
            'due_date': None,
            'planned_for': None,
            'notas':''
        }
        form_vals = task_form(defaults, key_prefix='create')
        if st.button('➕ Adicionar entregável', key='add_task'):
            insert_task(form_vals)
            st.success('Entregável criado.')

    st.divider()

    if st.button("🧨 Resetar banco (APAGA TUDO)"):
        init_db(drop_and_recreate=True)
        st.success("Banco recriado.")

# Abas principais (FUPs do Time como 2ª)
_tab_labels = ['🗓️ Hoje', '👥 FUPs do Time', '📚 Backlog', '⭐ Prioridade', '✅ Finalizados', '🕰️ Longo Prazo']
tab1, tabFUP, tab2, tab3, tab4, tab5 = st.tabs(_tab_labels)

# --- Hoje ---
with tab1:
    st.subheader('Hoje — Atividades em andamento ou do dia')
    today = date.today().isoformat()

    # Usar date(trim(...)) para evitar problemas com TEXT/espacos/formatos
    df1 = fetch_tasks(
        "status != 'Concluído' AND ("
        "  date(trim(planned_for)) = date(?) OR "
        "  date(trim(due_date))    = date(?) OR "
        "  status = 'Em andamento'"
        ") AND (due_date IS NULL OR date(trim(due_date)) >= date(?))",
        (today, today, today)
    )
    if df1.empty:
        st.info('Sem atividades do dia ou em andamento.')
    else:
        for _, row in df1.iterrows():
            with st.expander(f"[{row.prioridade}] {row.titulo}", expanded=False):
                st.markdown(f"**Empresa:** {row.empresa}  •  **Status:** {row.status}")
                st.markdown(f"**Prazo:** {row.due_date or '—'}  •  **Começar até:** {row.planned_for or '—'}")
                if row.notas:
                    st.write(row.notas)
                row_actions(row, prefix="hoje")

    # Atrasados: apenas due_date < hoje (comparando como data)
    st.subheader('Hoje — Atrasados')
    atrasados = fetch_tasks(
        "status != 'Concluído' AND due_date IS NOT NULL AND date(trim(due_date)) < date(?)",
        (today,)
    )
    if atrasados.empty:
        st.info('Nenhum entregável atrasado.')
    else:
        for _, row in atrasados.iterrows():
            with st.expander(f"[ATRASADO] [{row.prioridade}] {row.titulo}", expanded=False):
                st.markdown(f"**Empresa:** {row.empresa}  •  **Status:** {row.status}")
                st.markdown(f"**Prazo:** {row.due_date or '—'}  •  **Começar até:** {row.planned_for or '—'}")
                if row.notas:
                    st.write(row.notas)
                row_actions(row, prefix="atrasados")

    # Backlog (exclui o que já apareceu acima)
    st.subheader('Hoje — Fila (backlog)')
    all_open = fetch_tasks("status != 'Concluído'")
    ids_df1 = set(df1['id'].tolist()) if not df1.empty else set()
    ids_atrasados = set(atrasados['id'].tolist()) if not atrasados.empty else set()
    excluidos = ids_df1 | ids_atrasados
    df2 = all_open[~all_open['id'].isin(excluidos)]
    if df2.empty:
        st.info('Nenhum item na fila.')
    else:
        for _, row in df2.iterrows():
            st.write(f"• [{row.empresa}] [{row.prioridade}] {row.titulo} — (Prazo: {row.due_date or '—'} | Começar até: {row.planned_for or '—'})")

# --- FUPs no Time (2ª aba) ---
with tabFUP:
    st.subheader('FUPs do Time')
    df_fup = fetch_fups()
    if df_fup.empty:
        st.info('Nenhum FUP cadastrado.')
    else:
        # Subtítulos por Empresa, e dentro agrupar por Pessoa/Área (estrutura semelhante a Longo Prazo)
        for emp in EMPRESAS:
            sub_emp = df_fup[df_fup['empresa'] == emp]
            if sub_emp.empty:
                continue
            st.markdown(f"### {emp} — {len(sub_emp)} item(ns)")
            for pessoa_area, sub_pa in sub_emp.groupby('pessoa_area'):
                with st.expander(f"👤 {pessoa_area} — {len(sub_pa)} FUP(s)", expanded=False):
                    for _, row in sub_pa.iterrows():
                        cols = st.columns([0.1, 0.75, 0.15])
                        cols[0].markdown(f"**#{int(row.id)}**")
                        cols[1].write(row.fup)
                        if 'created_at' in row and row['created_at']:
                            cols[1].caption(f"Criado em: {row.created_at}")
                        if cols[2].button('🗑️ Apagar', key=f'fup_del_{row.id}'):
                            delete_fup(int(row.id))
                            st.rerun()

# --- Backlog ---
with tab2:
    st.subheader('Backlog por empresa')
    df = fetch_tasks("status != 'Concluído'")
    if df.empty:
        st.info('Sem itens no backlog.')
    else:
        for emp in EMPRESAS:
            sub = df[df['empresa'] == emp]
            if sub.empty:
                continue
            st.markdown(f"### {emp} — {len(sub)} item(ns)")
            for _, row in sub.iterrows():
                header = f"• [{row.prioridade}] {row.titulo} — (Prazo: {row.due_date or '—'} | Começar até: {row.planned_for or '—'})"
                with st.expander(header, expanded=False):
                    edit_vals = task_form(row, key_prefix=f'edit_{row.id}')
                    if st.button('💾 Salvar', key=f'save_{row.id}'):
                        update_task(int(row.id), edit_vals)
                        st.success('Atualizado.')
                    row_actions(row, prefix="backlog")

# --- Prioridade ---
with tab3:
    st.subheader('Fila por Prioridade')
    dfp = fetch_tasks("status != 'Concluído'")
    if dfp.empty:
        st.info('Nada pendente.')
    else:
        for p in PRIORIDADES:
            sub = dfp[dfp['prioridade']==p]
            st.markdown(f"### {p} — {len(sub)} item(ns)")
            for _, row in sub.iterrows():
                st.write(f"• [{row.empresa}] {row.titulo}  —  (Prazo: {row.due_date or '—'} | Começar até: {row.planned_for or '—'})")

# --- Finalizados ---
with tab4:
    st.subheader('Tarefas Finalizadas')
    dff = fetch_tasks("status = 'Concluído'")
    if dff.empty:
        st.info('Nenhuma tarefa finalizada.')
    else:
        for _, row in dff.iterrows():
            with st.expander(f"#{row.id} [{row.empresa}] {row.titulo}", expanded=False):
                st.markdown(f"**Prioridade:** {row.prioridade}  •  **Prazo:** {row.due_date or '—'}  •  **Começar até:** {row.planned_for or '—'}")
                if row.notas:
                    st.write(row.notas)
                # Somente mudar status permitido
                if st.button('↩️ Reabrir (mudar para Novo)', key=f'reopen_{row.id}'):
                    update_task(int(row.id), {'status':'Novo'})
                    st.rerun()

# --- Longo Prazo ---
with tab5:
    st.subheader('Lembretes de Longo Prazo')
    df_lp = fetch_long_term()
    if df_lp.empty:
        st.info('Nenhum lembrete cadastrado.')
    else:
        # Agrupa por Empresa -> Pessoa/Área
        for emp in EMPRESAS:
            sub_emp = df_lp[df_lp['empresa'] == emp]
            if sub_emp.empty:
                continue
            st.markdown(f"### {emp} — {len(sub_emp)} item(ns)")

            # Dentro de cada empresa, agrupar por Pessoa/Área
            for pessoa_area, sub_pa in sub_emp.groupby('pessoa_area'):
                with st.expander(f"👤 {pessoa_area} — {len(sub_pa)} lembrete(s)", expanded=False):
                    for _, row in sub_pa.iterrows():
                        cols = st.columns([0.1, 0.75, 0.15])
                        cols[0].markdown(f"**#{int(row.id)}**")
                        cols[1].write(row.lembrete)
                        if 'created_at' in row and row['created_at']:
                            cols[1].caption(f"Criado em: {row.created_at}")
                        if cols[2].button('🗑️ Apagar', key=f'lp_del_{row.id}'):
                            delete_long_term(int(row.id))
                            st.rerun()


