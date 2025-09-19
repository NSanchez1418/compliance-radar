# Wrapper: intenta ejecutar la app real y, si falla, muestra el error en pantalla.
try:
    import compliance_app  # tu app real está en compliance_app.py
except Exception as e:
    import streamlit as st
    st.error("❌ Error al iniciar la aplicación. Detalle abajo:")
    st.exception(e)

