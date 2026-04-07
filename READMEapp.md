# 🧠 Red Neuronal LSTM - Generador de Lenguaje Natural

## Descripción General

Aplicación interactiva basada en **Streamlit** que implementa un modelo de red neuronal **LSTM** (Long Short-Term Memory) para autocompletado y generación de lenguaje natural. La aplicación permite a los usuarios escribir texto y recibir sugerencias de palabras basadas en el contexto, o generar secuencias completas de forma automática.

**Características principales:**
- 🔮 Predicción de palabras en tiempo real
- 🚀 Generación automática de texto (Modo Autopiloto)
- 🎚️ Controles ajustables para creatividad y penalización de repeticiones
- 💾 Modelo LSTM preentrenado y optimizado

---

## 📋 Requisitos

- Python 3.8+
- Streamlit
- TensorFlow/Keras
- NumPy

---

## 📁 Estructura del Proyecto

```
App_Autocompletado/
├── app.py                              # Archivo principal de la aplicación
├── modelo_autocompletado_profundo.h5   # Modelo LSTM preentrenado
├── tokenizer.pkl                       # Diccionario de palabras (tokenizador)
├── requirements.txt                    # Dependencias del proyecto
├── README.md                           # Este archivo
└── .streamlit/                         # Configuración de Streamlit
```

---

## 🚀 Cómo Ejecutar

1. Activa el entorno virtual (si existe):
```bash
venv\Scripts\activate
```
2. Instala todas las dependencias con:
```bash
pip install -r requirements.txt

3. Ejecuta la aplicación:
```bash
streamlit run app.py
```

3. La aplicación se abrirá en tu navegador (por defecto en `http://localhost:8501`)

---

## 📚 Componentes y Funciones

### 1. **Configuración Inicial**

#### `st.set_page_config()`
- Configura el título de la página: "LSTM Generador Natural"
- Establece el icono: 📝
- Layout: centrado para mejor visualización

---

### 2. **Carga de Modelo y Tokenizador**

#### `cargar_cerebro()`
```python
@st.cache_resource
def cargar_cerebro():
```

**Propósito:** Carga el modelo LSTM y el tokenizador de forma eficiente.

**Parámetros:** Ninguno

**Retorna:**
- `modelo`: Modelo Keras cargado
- `tokenizer`: Tokenizador con el vocabulario del documento
- `vocab_size`: Tamaño total del vocabulario

**Características:**
- Usa decorador `@st.cache_resource` para cachear en memoria (mejora rendimiento)
- Manejo de errores con try-except
- Carga desde archivos: `modelo_autocompletado_profundo.h5` y `tokenizer.pkl`

---

### 3. **Lógica de Predicción - Modo Manual**

#### `obtener_sugerencias(texto_semilla, temperatura, factor_penalizacion)`

**Propósito:** Sugiere las 3 palabras más probables seguidas de un texto dado.

**Parámetros:**
- `texto_semilla` (str): Texto ingresado por el usuario
- `temperatura` (float): Control de creatividad (0.1-2.0)
  - Valores bajos (0.1): Predicciones obvias y conservadoras
  - Valores altos (2.0): Predicciones creativas y variadas
- `factor_penalizacion` (float): Penaliza repeticiones (1.0-5.0)
  - Evita que la IA repita palabras consecutivas

**Retorna:** Lista con las 3 palabras más probables

**Proceso Interno:**
1. Convierte el texto a secuencias numéricas usando el tokenizador
2. Rellena con padding a máximo 5 tokens
3. Obtiene predicciones del modelo (probabilidades)
4. Penaliza palabras ya presentes en el contexto
5. Aplica escala de temperatura para ajustar probabilidades
6. Convierte a distribución de probabilidad normalizada
7. Selecciona los 3 índices con mayor probabilidad
8. Convierte índices a palabras usando el tokenizador inverso

---

### 4. **Lógica de Predicción - Modo Autopiloto**

#### `generar_secuencia(texto_semilla, n_palabras, temperatura, factor_penalizacion)`

**Propósito:** Genera una secuencia completa de palabras a partir de una semilla.

**Parámetros:**
- `texto_semilla` (str): Frase inicial (semilla)
- `n_palabras` (int): Cantidad de palabras a generar (1-20)
- `temperatura` (float): Control de creatividad
- `factor_penalizacion` (float): Control de penalización

**Retorna:** Texto completo generado (semilla + palabras generadas)

**Proceso Interno:**
1. Inicializa lista de palabras generadas
2. Para cada iteración (hasta n_palabras):
   - Tokeniza el texto actual
   - Genera predicción del modelo
   - Aplica penalización y temperatura
   - Realiza **muestreo estocástico** (random sampling) usando `np.random.choice()`
     - Esto permite variabilidad, no siempre elige la palabra más probable
   - Decodifica el índice a palabra
   - Añade la palabra al texto actual
3. Retorna el texto completo generado

---

## 🎨 Interfaz de Usuario

### **Sección Superior**
- Título: "🧠 Red Neuronal LSTM"
- Subtítulo: "Autocompletado y Generación de Lenguaje Natural"

### **Barra Lateral (⚙️ Ajustes del Motor)**

Controles interactivos en tiempo real:

#### 1. 🔥 Temperatura (Creatividad)
- **Rango:** 0.1 - 2.0
- **Valor por defecto:** 0.8
- **Función:** Ajusta cuán creativa es la IA
- **Matemática:** Divide los logits por la temperatura

#### 2. 🛑 Penalización de Repetición
- **Rango:** 1.0 - 5.0
- **Valor por defecto:** 3.0
- **Función:** Penaliza palabras repetidas
- **Efecto:** Divide la probabilidad de palabras previas

---

### **Área Central - Dos Pestañas**

#### **Pestaña 1: ⌨️ Modo Manual (Teclado)**

**Componentes:**
1. **Caja de Escritura** - `st.text_area()`
   - Altura: 150 píxeles
   - Multilínea: permite escribir texto extenso
   - Memory cache: Streamlit guarda automáticamente el contenido

2. **Vista en Vivo** - Muestra la frase construida en una caja verde

3. **Botón "🧠 Adivinar siguiente palabra"**
   - Al hacer clic, procesa el texto con `obtener_sugerencias()`
   - Muestra 3 botones (columnas) con palabras sugeridas

4. **Botones de Palabras Sugeridas**
   - Al hacer clic, añade la palabra al texto automáticamente
   - Usa `st.session_state` para mantener el estado del texto

**Flujo de Uso:**
```
1. Usuario escribe texto
2. Usuario hace clic en "Adivinar siguiente palabra"
3. Aparecen 3 opciones
4. Usuario selecciona una palabra
5. La palabra se añade al texto automáticamente
6. Repetir desde paso 2
```

#### **Pestaña 2: 🚀 Modo Autopiloto (Generación)**

**Componentes:**
1. **Input de Semilla** - `st.text_input()`
   - Placeholder: "Ej: El detalle tiene un gran..."
   - Acepta la frase inicial

2. **Slider de Cantidad** - `st.slider()`
   - Rango: 1-20 palabras
   - Valor por defecto: 5 palabras
   - Permite controlar la longitud del texto generado

3. **Botón "🚀 Iniciar Generación"** (tipo primario)
   - Dispara `generar_secuencia()`
   - Muestra spinner mientras genera

4. **Área de Resultado**
   - Muestra el texto generado en una caja de éxito (verde)
   - Separadores visuales para mejor legibilidad

**Flujo de Uso:**
```
1. Usuario ingresa semilla
2. Usuario selecciona cantidad de palabras
3. Usuario hace clic en "Iniciar Generación"
4. La IA genera automáticamente el resto del texto
5. Resultado se muestra en la pantalla
```

---

## 🔧 Variables Globales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `modelo` | Keras Model | Red neuronal LSTM cargada |
| `tokenizer` | Tokenizer | Diccionario de palabras para convertir texto ↔ números |
| `vocab_size` | int | Tamaño del vocabulario (número de palabras únicas) |
| `temperatura` | float | Parámetro de creatividad (desde slider) |
| `penalizacion` | float | Parámetro de penalización (desde slider) |

---

## 📊 Flujo de Datos

```
Usuario ingresa texto
         ↓
   Tokenizador: texto → [números]
         ↓
   Padding: [números] → [números padded a 5]
         ↓
   Modelo LSTM: [números padded] → [probabilidades]
         ↓
   Aplicar Temperatura: probabilities / temperatura
         ↓
   Aplicar Penalización: penalizar palabras repetidas
         ↓
   Normalizar: Softmax
         ↓
   Seleccionar: argmax o muestreo
         ↓
   Decodificar: [números] → palabras
         ↓
   Mostrar al usuario
```

---

## 🎯 Parámetros Técnicos

### Modelo LSTM
- **Nombre:** `modelo_autocompletado_profundo.h5`
- **Arquitectura:** Red neuronal recurrente LSTM
- **Entrada:** Secuencia de hasta 5 palabras tokenizadas
- **Salida:** Distribución de probabilidades sobre el vocabulario
- **Función de activación:** Softmax

### Tokenizador
- **Formato:** Pickle (`.pkl`)
- **Nombre:** `tokenizer.pkl`
- **Contenido:** Mapeo bidireccional palabra ↔ índice numérico
- **Función:** Convierte texto → números (necesario para el modelo)

---

## ⚠️ Validaciones y Manejo de Errores

### En `cargar_cerebro()`:
- Try-except: Captura errores al cargar archivos
- Retorna `None` si hay error (fácil de verificar)

### En `obtener_sugerencias()`:
- Verifica que el modelo y tokenizador existan
- Verifica que la secuencia no esté vacía (palabra no reconocida)
- Retorna lista vacía si hay problemas

### En `generar_secuencia()`:
- Verifica que el modelo y tokenizador existan
- Detiene generación si encuentra palabra no reconocida
- Retorna solo la semilla si hay error

### En la Interfaz:
- Valida que haya texto antes de procesar
- Valida que la semilla no esté vacía en Modo Autopiloto
- Muestra mensajes informativos (info, warning, error)

---

## 💡 Consejos de Uso

| Objetivo | Ajuste | Valor Recomendado |
|----------|--------|-------------------|
| Respuestas predecibles | Baja temperatura | 0.1 - 0.5 |
| Balance | Temperatura moderada | 0.7 - 0.9 |
| Respuestas creativas | Alta temperatura | 1.5 - 2.0 |
| Evitar repeticiones | Alta penalización | 3.0 - 5.0 |
| Permitir repeticiones | Baja penalización | 1.0 - 2.0 |

---

## 📝 Ejemplo de Uso

### Modo Manual:
```
1. Escribe: "El machine learning"
2. Click en "Adivinar siguiente palabra"
3. Opciones: "es", "utiliza", "permite"
4. Selecciona "es"
5. Texto ahora: "El machine learning es"
6. Repite proceso
```

### Modo Autopiloto:
```
1. Semilla: "La inteligencia artificial"
2. Palabras a generar: 8
3. Click en "Iniciar Generación"
4. Resultado: "La inteligencia artificial es una rama de la 
   ciencia que estudia sistemas inteligentes"
```

---

## 🔐 Notas de Seguridad

- ✅ No realiza llamadas a internet
- ✅ Procesa todo localmente
- ✅ Los archivos del modelo nunca se modifican
- ✅ Información completamente privada y en tu máquina

---

## 📈 Mejoras Futuras Posibles

- [ ] Exportar texto generado a archivo
- [ ] Historial de generaciones
- [ ] Múltiples modelos para elegir
- [ ] Entrenar modelo con nuevos documentos
- [ ] Análisis estadístico de predicciones
- [ ] Interfaz para ajustar semillas guardadas

---

## 📞 Soporte

Si encuentras errores:
1. Verifica que `modelo_autocompletado_profundo.h5` y `tokenizer.pkl` existan
2. Verifica que tengas todas las dependencias instaladas
3. Intenta reinstalar con `pip install -r requirements.txt --upgrade`

