import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Config pagina
st.set_page_config(page_title="Studio Integrali & Cavalieri", layout="wide")

st.title("📐 Studio degli Integrali con Visualizzazione 3D (Cavalieri)")

st.markdown("""
Questa app ti permette di:
- Calcolare integrali definiti
- Visualizzare il grafico e l’area sotto la curva
- Esplorare il principio di **Cavalieri** con sezioni di solidi di rivoluzione
""")

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

scelta = st.selectbox(
    "Scegli una funzione predefinita oppure scrivi la tua:",
    list(funzioni_predefinite.keys()) + ["Personalizzata"]
)

if scelta != "Personalizzata":
    expr_input = scelta
    a_default, b_default = funzioni_predefinite[scelta]
else:
    expr_input = st.text_input("✏️ Inserisci la funzione f(x):", value="sin(x)")
    a_default, b_default = 0.0, float(np.pi)

a = st.number_input("📉 Limite inferiore (a):", value=float(a_default))
b = st.number_input("📈 Limite superiore (b):", value=float(b_default))

x = sp.Symbol('x')

try:
    # Parsing funzione
    f_expr = sp.sympify(expr_input)
    f_lambdified = sp.lambdify(x, f_expr, 'numpy')
    integral = sp.integrate(f_expr, (x, a, b)).evalf()

    # Mostra i passaggi simbolici del calcolo integrale
    st.subheader("📝 Passaggi simbolici del calcolo integrale")
    try:
        integral_indef = sp.integrate(f_expr, x)
        st.latex(r"\int " + sp.latex(f_expr) + r"\,dx = " + sp.latex(integral_indef) + " + C")
        st.latex(r"\int_{" + str(a) + "}^{" + str(b) + "} " + sp.latex(f_expr) + r"\,dx = " +
                 sp.latex(integral_indef.subs(x, b)) + " - " + sp.latex(integral_indef.subs(x, a)) +
                 " = " + str(integral))
    except Exception as e:
        st.info("Passaggi simbolici non disponibili per questa funzione.")

    # === GRAFICO 2D ===
    X_vals = np.linspace(a, b, 1000)
    Y_vals = f_lambdified(X_vals)

    fig2d, ax = plt.subplots()
    ax.plot(X_vals, Y_vals, label=f'f(x) = {expr_input}', color='blue')
    ax.fill_between(X_vals, Y_vals, alpha=0.3, color='orange')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"Area sotto la curva = {integral:.5f}")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    st.subheader("📊 Grafico 2D con area integrale")
    st.pyplot(fig2d)

    st.success(f"✅ Valore dell'integrale da {a} a {b}: {integral:.5f}")

    # === GRAFICO 3D (Cavalieri) ===
    st.subheader("🌀 Solido di rivoluzione (Principio di Cavalieri)")

    theta = np.linspace(0, 2*np.pi, 100)
    X = np.linspace(a, b, 300)
    Theta, X_mesh = np.meshgrid(theta, X)
    Y = f_lambdified(X)
    R = f_lambdified(X_mesh)
    
    # Assicurarsi che R, Y, Z siano array numpy e correttamente formattati
    if isinstance(R, np.ndarray):
        Z = R * np.sin(Theta)
        Y_rot = R * np.cos(Theta)

        fig3d = go.Figure(data=[go.Surface(x=X_mesh, y=Y_rot, z=Z, colorscale='Viridis', opacity=0.8)])

        # Slider per sezione (altezza z)
        z0 = st.slider("📏 Altezza della sezione (z)", float(np.min(Z)), float(np.max(Z)), step=0.1, value=float(np.median(Z)))

        # === Sezione circolare vera: tutti i cerchi a quota z0 ===
        section_x = []
        section_y = []
        section_z = []

        for xi in X:
            ri = f_lambdified(xi)
            if np.abs(z0) <= np.abs(ri) and ri != 0:
                # Calcola il raggio del cerchio a quota z0
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
                camera=dict(eye=dict(x=2.5, y=2.5, z=2.0))  # Più lontano per "ingrandire"
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        st.plotly_chart(fig3d, use_container_width=True)
    else:
        st.error("❌ Errore nei dati: 'R' non è un array numpy valido.")
    
except Exception as e:
    st.error(f"❌ Errore: {e}")
