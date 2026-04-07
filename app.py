import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="LSTM Generador Natural", page_icon="📝", layout="centered")

# --- 2. CARGAR MODELO Y DICCIONARIO ---
@st.cache_resource
def cargar_cerebro():
    try:
        modelo = load_model('modelo_autocompletado_profundo.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        vocab_size = len(tokenizer.word_index) + 1
        return modelo, tokenizer, vocab_size
    except Exception as e:
        st.error(f"⚠️ Error al cargar los archivos: {e}")
        return None, None, 0
    
modelo, tokenizer, vocab_size = cargar_cerebro()

# --- 3. LÓGICAS DE PREDICCIÓN ---
# Lógica 1: Para el Modo Manual (Top 3 Palabras)
def obtener_sugerencias(texto_semilla, temperatura, factor_penalizacion):
    if not modelo or not tokenizer: return []
    secuencias = tokenizer.texts_to_sequences([texto_semilla])
    if len(secuencias[0]) == 0: return [] 
    palabras_previas = secuencias[0]
    tokens = pad_sequences([secuencias[0]], maxlen=5, padding='pre')
    probas = modelo.predict(tokens, verbose=0)[0]
    probas[0] = 0.0 

    # Penalizar repeticiones de la misma frase
    for id_gen in palabras_previas:
        if id_gen < len(probas):
            probas[id_gen] /= factor_penalizacion
    probas = np.log(probas + 1e-7) / float(temperatura)
    exp_probas = np.exp(probas)
    probas = exp_probas / np.sum(exp_probas)    
    mejores_indices = np.argsort(probas)[-3:][::-1]

    sugerencias = []
    for indice in mejores_indices:
        for palabra, i in tokenizer.word_index.items():
            if i == indice:
                sugerencias.append(palabra)
                break
    return sugerencias

# Lógica 2: Código Original para el Modo Autopiloto (Generación Completa)
def generar_secuencia(texto_semilla, n_palabras, temperatura, factor_penalizacion):
    if not modelo or not tokenizer: return texto_semilla
    palabras_generadas = []
    texto_actual = texto_semilla

    for _ in range(n_palabras):
        secuencias = tokenizer.texts_to_sequences([texto_actual])
        if len(secuencias[0]) == 0: break # Si hay una palabra no reconocida, se detiene
        tokens = pad_sequences([secuencias[0]], maxlen=5, padding='pre')
        probas = modelo.predict(tokens, verbose=0)[0]
        probas[0] = 0.0
        for id_gen in palabras_generadas:
            if id_gen < len(probas):
                probas[id_gen] /= factor_penalizacion
        probas = np.log(probas + 1e-7) / float(temperatura)
        exp_probas = np.exp(probas)
        probas = exp_probas / np.sum(exp_probas)        
        # Muestreo Estocástico para el Autopiloto
        indice = np.random.choice(range(vocab_size), p=probas)
        palabras_generadas.append(indice)

        palabra_final = ""
        for palabra, i in tokenizer.word_index.items():
            if i == indice:
                palabra_final = palabra
                break
        texto_actual += " " + palabra_final
    return texto_actual

# --- 4. INTERFAZ VISUAL (FRONTEND) ---
st.title("🧠 Red Neuronal LSTM")
st.caption("Autocompletado y Generación de Lenguaje Natural.")

# ⚙️ Barra Lateral (Controles del Motor)
st.sidebar.header("⚙️ Ajustes del Motor")
st.sidebar.markdown("Ajusta las matemáticas de la red neuronal en tiempo real.")

temperatura = st.sidebar.slider(
    "🔥 Temperatura (Creatividad)", 
    min_value=0.1, max_value=2.0, value=0.8, step=0.1,
    help="Valores bajos = respuestas obvias. Valores altos = respuestas creativas."
)

penalizacion = st.sidebar.slider(
    "🛑 Penalización de Repetición", 
    min_value=1.0, max_value=5.0, value=3.0, step=0.5,
    help="Castiga a la red si intenta repetir la misma palabra ('el el el')."
)
# 🖥️ Área Central (Pestañas)
tab_manual, tab_auto = st.tabs(["⌨️ Modo Manual (Teclado)", "🚀 Modo Autopiloto (Generación)"])

# ==== PESTAÑA 1: MODO MANUAL ====
with tab_manual:
    st.subheader("Teclado Predictivo")
    st.info("Escribe y la red neuronal te dará 3 opciones para la siguiente palabra.")
    
    # 1. Usamos 'key' para que Streamlit guarde el texto automáticamente en su memoria
    if 'txt_manual' not in st.session_state:
        st.session_state.txt_manual = ""

    texto_actual = st.text_area("Caja de escritura:", height=150, key="txt_manual")

    # 2. EL TOQUE VISUAL QUE PEDISTE: Muestra la frase en tiempo real abajo
    if texto_actual:
        st.markdown("### 📝 Frase construida:")
        st.success(f"*{texto_actual}*") # Esto lo pone en una cajita verde elegante

    if st.button("🧠 Adivinar siguiente palabra", use_container_width=True):
        if texto_actual:
            with st.spinner('Analizando contexto...'):
                sugerencias = obtener_sugerencias(texto_actual, temperatura, penalizacion)

                if sugerencias:
                    st.write("**Haz clic en una palabra para añadirla:**")
                    cols = st.columns(3)

                    for idx, palabra in enumerate(sugerencias):
                        # 3. EL ARREGLO DE MEMORIA: Suma la palabra directamente a la llave del text_area
                        def actualizar_texto(palabra_elegida=palabra):
                            st.session_state.txt_manual += " " + palabra_elegida

                        cols[idx].button(
                            palabra, 
                            key=f"btn_{idx}_{palabra}", 
                            use_container_width=True, 
                            on_click=actualizar_texto
                        )
                else:
                    st.warning("Escribe una palabra válida del documento.")

# ==== PESTAÑA 2: MODO AUTOPILOTO ====
with tab_auto:
    st.subheader("Generación de Secuencias")
    st.info("Escribe una semilla y deja que el Autompletado complete la frase.")
    texto_semilla = st.text_input("Palabras de inicio (Semilla):", placeholder="Ej: El detalle tiene un gran...")
    num_palabras = st.slider("📏 ¿Cuántas palabras quieres que genere?", min_value=1, max_value=20, value=5)

    if st.button("🚀 Iniciar Generación", type="primary", use_container_width=True):
        if texto_semilla:
            with st.spinner('Escribiendo la historia...'):
                resultado_final = generar_secuencia(texto_semilla, num_palabras, temperatura, penalizacion)
                
                st.success("¡Texto Generado!")
                st.write("---")
                st.write(resultado_final)
                st.write("---")
        else:
            st.warning("Por favor ingresa una semilla para arrancar el motor.")