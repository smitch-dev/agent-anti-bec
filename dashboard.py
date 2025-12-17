import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(page_title="AIBEC Dashboard v1.1", layout="wide")

# --- CSS ALLÉGÉ (Compatible Mode Sombre/Clair) ---
st.markdown("""
    <style>
    .stMetric {
        border: 1px solid #4B5563;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️AI-BEC-AGENT Monitoring des Menaces (by smitch)")

if os.path.exists("detection_history.csv"):
    # On charge les données
    df = pd.read_csv("detection_history.csv")
    
    # --- SECTION 1 : METRIQUES ---
    total = len(df)
    bec = len(df[df['decision'] == 'BEC_ALERT'])
    legit = len(df[df['decision'] == 'LEGITIMATE'])
    taux_menace = (bec / total * 100) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    # Plus de couleur de fond forcée ici pour éviter les conflits de thème
    col1.metric("Total Scannés", total)
    col2.metric("Alertes BEC", bec, delta=f"{taux_menace:.1f}%", delta_color="inverse")
    col3.metric("Légitimes", legit)
    col4.metric("Risque Moyen", f"{df['final_score'].mean():.2f}")

    st.divider()

    # --- SECTION 2 : GRAPHIQUES ---
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("📈 Évolution du Risque")
        st.line_chart(df['final_score'])

    with right_column:
        st.subheader("📊 Répartition")
        fig = px.pie(
            df, names='decision', color='decision',
            color_discrete_map={'BEC_ALERT': '#ef553b', 'LEGITIMATE': '#636efa'},
            hole=0.4
        )
        # On force le fond du graphique à être transparent pour le mode sombre
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # --- SECTION 3 : JOURNAL (SANS INDEX) ---
    st.subheader("📜 Journal détaillé")
    
    # Tri par date
    df_sorted = df.sort_values(by="timestamp", ascending=False)
    
    # hide_index=True enlève la colonne "index-streamlit-generated"
    st.dataframe(df_sorted, use_container_width=True, hide_index=True)

else:
    st.info("Aucune donnée disponible.")