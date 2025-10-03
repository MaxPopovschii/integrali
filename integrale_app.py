import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import datetime
import pandas as pd
import json

# === FUNZIONI MODULARI ===
def parse_function(expr_input, a, b):
    x = sp.Symbol('x')
    try:
        f_expr = sp.sympify(expr_input)
        f_lambdified = sp.lambdify(x, f_expr, 'numpy')
        # Test valutazione su un punto
        _ = f_lambdified((a + b) / 2)
        return f_expr, f_lambdified, None
    except Exception as e:
        return None, None, e

def compute_integral(f_expr, x, a, b):
    try:
        integral = sp.integrate(f_expr, (x, a, b)).evalf()
        return integral, None
    except Exception as e:
        return None, e

def plot_2d(X_vals, Y_vals, expr_input, integral, a, b):
    fig2d, ax = plt.subplots()
    ax.plot(X_vals, Y_vals, label=f'f(x) = {expr_input}', color='blue')
    ax.fill_between(X_vals, Y_vals, alpha=0.3, color='orange')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"Area sotto la curva = {integral:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)
    return fig2d

def plot_3d(f_lambdified, a, b, n_points, z0):
    theta = np.linspace(0, 2*np.pi, 100)
    X = np.linspace(a, b, int(n_points/3))
    Theta, X_mesh = np.meshgrid(theta, X)
    R = f_lambdified(X_mesh)
    if not isinstance(R, np.ndarray):
        return None, "'R' non è un array numpy valido."
    Z = R * np.sin(Theta)
    Y_rot = R * np.cos(Theta)
    fig3d = go.Figure(data=[go.Surface(x=X_mesh, y=Y_rot, z=Z, colorscale='Viridis', opacity=0.8)])
    section_x, section_y, section_z = [], [], []
    for xi in X:
        ri = f_lambdified(xi)
        if np.abs(z0) <= np.abs(ri) and ri != 0:
            r_sez = np.sqrt(ri**2 - z0**2)
            theta_sec = np.linspace(0, 2*np.pi, 100)
            y_sec = r_sez * np.cos(theta_sec)
            z_sec = np.full_like(theta_sec, z0)
            section_x.extend([xi]*len(theta_sec))
            section_y.extend(y_sec)
            section_z.extend(z_sec)
    fig3d.add_trace(go.Scatter3d(
        x=section_x,
        y=section_y,
        z=section_z,
        mode='markers',
        marker=dict(size=3, color='red'),
        name=f'Sezione z = {z0:.2f}'
    ))
    fig3d.update_layout(
        title="Solido di rivoluzione + Sezione (Cavalieri)",
        width=1000,
        height=700,
        scene=dict(
            xaxis_title='x',
            yaxis_title='r·cos(θ)',
            zaxis_title='r·sin(θ)',
            xaxis=dict(range=[a, b]),
            camera=dict(eye=dict(x=2.5, y=2.5, z=2.0))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig3d, None



# === CONFIG PAGINA ===
st.set_page_config(page_title="Studio Integrali & Cavalieri", layout="wide", page_icon="📐")

# === SIDEBAR MODERNA ===
with st.sidebar:
    # Tema chiaro/scuro (Streamlit supporta solo via config, workaround: info)
    st.markdown("""
    <span style='font-size:15px;'>
    <b>Tema:</b> <i>Segui impostazioni di sistema</i><br>
    <span style='color:gray;font-size:12px;'>Per cambiare tema vai su <b>Impostazioni Streamlit</b> (in alto a destra)</span>
    </span>
    """, unsafe_allow_html=True)
    st.markdown("""
    <details>
    <summary><b>ℹ️ About & Credits</b></summary>
    <ul>
    <li>App sviluppata da <b>Max Popovschii</b></li>
    <li>Open Source su <a href='https://github.com/MaxPopovschii/integrali' target='_blank'>GitHub</a></li>
    <li>Icone da <a href='https://icons8.com' target='_blank'>Icons8</a></li>
    <li>Powered by Streamlit, SymPy, Plotly, Matplotlib</li>
    </ul>
    </details>
    <details>
    <summary><b>💬 Feedback</b></summary>
    <span style='font-size:13px;'>
    Per suggerimenti o segnalare bug:<br>
    <a href='mailto:max.popovschii@gmail.com'>max.popovschii@gmail.com</a><br>
    Oppure apri una issue su <a href='https://github.com/MaxPopovschii/integrali/issues' target='_blank'>GitHub</a>.
    </span>
    </details>
    <br>
    <details>
    <summary><b>❓ Guida rapida</b></summary>
    <ul>
    <li>Scegli una funzione o inseriscila manualmente</li>
    <li>Imposta i limiti di integrazione</li>
    <li>Regola la risoluzione dei grafici</li>
    <li>Visualizza e scarica grafici e dati</li>
    <li>Esplora il solido 3D e le sezioni</li>
    </ul>
    </details>
    """, unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/integral-symbol.png", width=64)
    st.title("Integrali 3D App")
    st.markdown("""
    <span style='font-size:16px;'>
    <b>Autore:</b> Max Popovschii  
    <b>Ultimo aggiornamento:</b> {date}
    </span>
    """.format(date=datetime.date.today()), unsafe_allow_html=True)
    st.markdown("""
    <hr>
    <b>Funzionalità:</b>
    <ul>
    <li>Calcolo integrali definiti</li>
    <li>Grafici 2D e 3D interattivi</li>
    <li>Download immagini e dati</li>
    <li>Principio di Cavalieri</li>
    </ul>
    <hr>
    <b>Link utili:</b><br>
    <a href='https://it.wikipedia.org/wiki/Principio_di_Cavalieri' target='_blank'>Cavalieri (Wikipedia)</a><br>
    <a href='https://github.com/MaxPopovschii/integrali' target='_blank'>Codice sorgente su GitHub</a>
    <hr>
    <b>Contatti:</b> <a href='mailto:max.popovschii@gmail.com'>max.popovschii@gmail.com</a>
    """, unsafe_allow_html=True)
    st.markdown("""
    <hr>
    <span style='font-size:13px;color:gray;'>
    © {year} Max Popovschii
    </span>
    """.format(year=datetime.date.today().year), unsafe_allow_html=True)

# === HEADER PRINCIPALE ===
st.markdown("""
# 📐 Studio degli Integrali con Visualizzazione 3D (Cavalieri)
<span style='font-size:18px;'>
Questa app ti permette di:
<ul>
<li>Calcolare integrali definiti</li>
<li>Visualizzare il grafico e l’area sotto la curva</li>
<li>Esplorare il principio di <b>Cavalieri</b> con sezioni di solidi di rivoluzione</li>
</ul>
</span>
""", unsafe_allow_html=True)
# Funzioni predefinite e intervalli consigliati
funzioni_predefinite = {
    "sin(x)": (0, np.pi),
    "cos(x)": (0, np.pi),
    "x**2": (0, 1),
    "sqrt(x)": (0, 1),
    "exp(-x**2)": (-2, 2),
    "1/(1+x**2)": (-5, 5),
    "log(x+1)": (0, 2),
    "abs(x)": (-1, 1),
    "x**3": (-1, 1),
    "exp(x)": (0, 1),
    "tan(x)": (0, np.pi/4),
    "1/x": (1, 2),
    "cos(x)**2": (0, np.pi),
    "sin(x)**2": (0, np.pi),
    "1/(x+1)": (0, 2),
    "x*exp(-x)": (0, 5),
    "1/sqrt(x)": (0.01, 1),
    "arctan(x)": (-2, 2),
    "sinh(x)": (-1, 1),
    "cosh(x)": (-1, 1)
}

# Layout a colonne per input e opzioni
col1, col2 = st.columns([2, 1])

# --- SESSION STATE per salvataggio/caricamento/reset ---
if 'scelta' not in st.session_state:
    st.session_state['scelta'] = list(funzioni_predefinite.keys())[0]
if 'expr_input' not in st.session_state:
    st.session_state['expr_input'] = st.session_state['scelta']
if 'a' not in st.session_state:
    st.session_state['a'] = float(funzioni_predefinite[st.session_state['scelta']][0])
if 'b' not in st.session_state:
    st.session_state['b'] = float(funzioni_predefinite[st.session_state['scelta']][1])
if 'n_points' not in st.session_state:
    st.session_state['n_points'] = 1000

def reset_params():
    st.session_state['scelta'] = list(funzioni_predefinite.keys())[0]
    st.session_state['expr_input'] = st.session_state['scelta']
    st.session_state['a'] = float(funzioni_predefinite[st.session_state['scelta']][0])
    st.session_state['b'] = float(funzioni_predefinite[st.session_state['scelta']][1])
    st.session_state['n_points'] = 1000
    st.session_state['history'] = []

def save_session():
    data = {
        'scelta': st.session_state['scelta'],
        'expr_input': st.session_state['expr_input'],
        'a': st.session_state['a'],
        'b': st.session_state['b'],
        'n_points': st.session_state['n_points']
    }
    return json.dumps(data, indent=2).encode()

def load_session(uploaded):
    try:
        data = json.loads(uploaded.getvalue())
        st.session_state['scelta'] = data.get('scelta', st.session_state['scelta'])
        st.session_state['expr_input'] = data.get('expr_input', st.session_state['expr_input'])
        st.session_state['a'] = data.get('a', st.session_state['a'])
        st.session_state['b'] = data.get('b', st.session_state['b'])
        st.session_state['n_points'] = data.get('n_points', st.session_state['n_points'])
        st.success('✅ Sessione caricata!')
    except Exception as e:
        st.error(f'Errore caricamento sessione: {e}')

with col1:
    scelta = st.selectbox(
        "Scegli una funzione predefinita oppure scrivi la tua: ℹ️",
        list(funzioni_predefinite.keys()) + ["Personalizzata"],
        help="Scegli una funzione nota o scrivine una personalizzata in termini di x. Esempio: exp(-x**2) + x",
        key='scelta',
        format_func=lambda x: x + (" (custom)" if x == "Personalizzata" else "")
    )
    if scelta != "Personalizzata":
        st.session_state['expr_input'] = scelta
        a_default, b_default = funzioni_predefinite[scelta]
    else:
        st.session_state['expr_input'] = st.text_input(
            "✏️ Inserisci la funzione f(x):",
            value=st.session_state['expr_input'],
            help="Scrivi la funzione in termini di x. Esempio: exp(-x**2) + x",
            key='expr_input',
            placeholder="Esempio: exp(-x**2) + x",
            label_visibility="visible"
        )
        a_default, b_default = 0.0, float(np.pi)
    st.session_state['a'] = st.number_input(
        "📉 Limite inferiore (a):",
        value=float(st.session_state['a']),
        help="Estremo inferiore dell'integrale. Deve essere un numero reale.",
        key='a',
        placeholder="ad esempio 0"
    )
    st.session_state['b'] = st.number_input(
        "📈 Limite superiore (b):",
        value=float(st.session_state['b']),
        help="Estremo superiore dell'integrale. Deve essere un numero reale.",
        key='b',
        placeholder="ad esempio pi"
    )
    st.button('🔄 Reset parametri', on_click=reset_params, help='Ripristina i parametri iniziali', type='secondary')
with col2:
    st.session_state['n_points'] = st.slider(
        "Risoluzione grafici (n punti)",
        min_value=100, max_value=3000, value=st.session_state['n_points'], step=100,
        help="Numero di punti per la discretizzazione dei grafici. Più punti = grafico più liscio ma più lento.",
        key='n_points'
    )
    st.download_button('💾 Salva sessione', data=save_session(), file_name='integrale_sessione.json', mime='application/json', help='Scarica i parametri attuali in un file JSON')
    uploaded = st.file_uploader('📂 Carica sessione', type='json', help='Carica una sessione salvata')
    if uploaded:
        load_session(uploaded)

# === CRONOLOGIA CALCOLI ===
if 'history' not in st.session_state:
    st.session_state['history'] = []

expr_input = st.session_state['expr_input']
a = st.session_state['a']
b = st.session_state['b']
n_points = st.session_state['n_points']

x = sp.Symbol('x')



# --- GESTIONE ERRORI E PARSING MODULARE ---
f_expr, f_lambdified, parse_err = parse_function(expr_input, a, b)
if parse_err:
    st.error(f"<span style='color:#d9534f;font-size:18px;'>❌ Errore nella funzione inserita:</span> {parse_err}", unsafe_allow_html=True)
    st.stop()
integral, int_err = compute_integral(f_expr, x, a, b)
if int_err:
    st.error(f"<span style='color:#d9534f;font-size:18px;'>❌ Errore nel calcolo dell'integrale:</span> {int_err}", unsafe_allow_html=True)
    st.stop()

# --- PASSAGGI SIMBOLICI ---
st.subheader("📝 Passaggi simbolici del calcolo integrale")
try:
    integral_indef = sp.integrate(f_expr, x)
    st.latex(r"\int " + sp.latex(f_expr) + r"\,dx = " + sp.latex(integral_indef) + " + C")
    st.latex(r"\int_{" + str(a) + "}^{" + str(b) + "} " + sp.latex(f_expr) + r"\,dx = " +
             sp.latex(integral_indef.subs(x, b)) + " - " + sp.latex(integral_indef.subs(x, a)) +
             " = " + str(integral))
except Exception:
    st.info("Passaggi simbolici non disponibili per questa funzione.")

# === GRAFICO 2D ===
st.subheader("📊 Grafico 2D con area integrale")
try:
    X_vals = np.linspace(a, b, n_points)
    Y_vals = f_lambdified(X_vals)
    fig2d = plot_2d(X_vals, Y_vals, expr_input, integral, a, b)
    st.pyplot(fig2d)
    # Download grafico 2D
    buf = BytesIO()
    fig2d.savefig(buf, format="png")
    st.download_button("📥 Scarica grafico 2D", data=buf.getvalue(), file_name="grafico_integrale.png", mime="image/png")
    # Download dati CSV/JSON
    df = pd.DataFrame({"x": X_vals, "f(x)": Y_vals})
    st.download_button("⬇️ Scarica dati (CSV)", data=df.to_csv(index=False).encode(), file_name="dati_integrale.csv", mime="text/csv")
    st.download_button("⬇️ Scarica dati (JSON)", data=df.to_json(orient="records").encode(), file_name="dati_integrale.json", mime="application/json")
except Exception as e:
    st.error(f"<span style='color:#d9534f;font-size:18px;'>Errore nella generazione del grafico 2D:</span> {e}", unsafe_allow_html=True)


# Badge risultato integrale + copia + cronologia
st.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;'>
    <span style='font-size:22px;color:#198754;'>✅</span>
    <span style='font-size:18px;background:#e6ffe6;border-radius:8px;padding:6px 16px;box-shadow:0 1px 2px #0001;'>
        Valore dell'integrale da <b>{a}</b> a <b>{b}</b>: <b id='integrale_val'>{integral:.5f}</b>
    </span>
    <button onclick="navigator.clipboard.writeText(document.getElementById('integrale_val').innerText)" style='margin-left:8px;padding:4px 10px;border-radius:6px;border:none;background:#d1e7dd;color:#198754;cursor:pointer;font-size:15px;'>📋 Copia</button>
</div>
<script>/* accessibility: focus on copy */document.querySelectorAll('button').forEach(b=>b.setAttribute('aria-label','Copia risultato'));</script>
""", unsafe_allow_html=True)
st.balloons()

# Aggiorna cronologia calcoli
if len(st.session_state['history']) == 0 or st.session_state['history'][-1] != (expr_input, a, b, float(integral)):
    st.session_state['history'].append((expr_input, a, b, float(integral)))
    if len(st.session_state['history']) > 5:
        st.session_state['history'] = st.session_state['history'][-5:]

with st.expander('🕑 Cronologia ultimi calcoli', expanded=False):
    if st.session_state['history']:
        for i, (ex, aa, bb, val) in enumerate(reversed(st.session_state['history']), 1):
            st.markdown(f"<span style='color:#888;'>{i}.</span> <b>∫<sub>{aa}</sub><sup>{bb}</sup> {ex} dx</b> = <span style='color:#198754;'>{val:.5f}</span>", unsafe_allow_html=True)
    else:
        st.info('Nessun calcolo recente.')

# === GRAFICO 3D (Cavalieri) ===
st.subheader("🌀 Solido di rivoluzione (Principio di Cavalieri)")
try:
    # Calcolo Z per slider
    theta = np.linspace(0, 2*np.pi, 100)
    X = np.linspace(a, b, int(n_points/3))
    Theta, X_mesh = np.meshgrid(theta, X)
    R = f_lambdified(X_mesh)
    if isinstance(R, np.ndarray):
        Z = R * np.sin(Theta)
        z0 = st.slider("📏 Altezza della sezione (z)", float(np.min(Z)), float(np.max(Z)), step=0.1, value=float(np.median(Z)), help="Sposta la sezione per vedere il principio di Cavalieri.")
        fig3d, err3d = plot_3d(f_lambdified, a, b, n_points, z0)
        if err3d:
            st.error(f"<span style='color:#d9534f;font-size:18px;'>Errore 3D:</span> {err3d}", unsafe_allow_html=True)
        else:
            st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.error("<span style='color:#d9534f;font-size:18px;'>❌ Errore nei dati: 'R' non è un array numpy valido.</span>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"<span style='color:#d9534f;font-size:18px;'>Errore nella generazione del grafico 3D:</span> {e}", unsafe_allow_html=True)
