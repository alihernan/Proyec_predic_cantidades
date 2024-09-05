import requests
import os
from dotenv import load_dotenv
import pandas as pd
import joblib
import logging
from flask import Flask, request, render_template
import math

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# URL base de la API
api_url_base = "https://centralusdtapp73.epicorsaas.com/SaaS5333/api/v1/BaqSvc/HMP_Part_model_predict(ALICO)/?Parte="
username = os.getenv('USERNAME')
password = os.getenv('PASSWORD')

# Diccionario para mapear los nombres de columnas
column_mapping = {
    "PartPlant_PartNum": "Part",
    "Part_ShortChar02": "TipoProducto",
    "Part_ShortChar04": "Estructura",
    "Part_ShortChar05": "Estructura 2",
    "Part_ShortChar03": "TipoBolsa",
    "Part_ShortChar10": "TipoSellado",
    "UD02_ShortChar05": "Tipo de montaje",
    "Part_Number01": "Ancho",
    "Part_Number02": "Largo",
    "Part_Number03": "Calibre2",
    "Part_Character07": "Dispositivo 1",
    "Part_Character08": "Dispositivo 2",
    "Part_Character06": "Dispositivo 3",
    "Part_Dispositivo4_c": "Dispositivo 4",
    "UD02_ShortChar03": "Troquel 1",
    "Part_UserChar4": "Troquel 2",
    "Part_DobleCorte_c": "DobleCorte_c",
    "Part_ShortChar01": "Tipo de impresion"
}

# Diccionarios de clasificación de troqueles
troqueles_forma = {"TR013", "TR014", "TR015", "TR018", "TR019", "TR020", "TR022", "TR024", "TR026", "TR027", "TR033",
                   "TR034", "TR038", "TR039", "TR040", "TR042", "TR045", "TR046", "TR049", "TR052", "TR055", "TR056",
                   "TR058", "TR059", "TR060", "TR061", "TR062", "TR063", "TR064", "TR065", "TR066", "TR067", "TR068",
                   "TR069", "TR070", "TR072", "TR073", "TR074", "TR075", "TR076", "TR077", "TR078", "TR079", "TR080",
                   "TR081", "TR083", "TR084", "TR086", "TR089", "TR090", "TR091", "TR093", "TR096", "TR098", "TR099",
                   "TR100", "TR101", "TR102", "TR103", "TR104", "TR105"}

troqueles_alto = {"TR001", "TR002", "TR003", "TR004", "TR005", "TR008", "TR011", "TR016", "TR023", "TR025", "TR029",
                  "TR031", "TR037", "TR043", "TR047", "TR048", "TR053", "TR054", "TR057", "TR071", "TR082", "TR085",
                  "TR087", "TR088", "TR092", "TR094", "TR095", "TR097"}

troqueles_bajo = {"TR007", "TR009", "TR010"}

def get_api_data(parte):
    # Completar la URL con la parte especificada por el usuario
    url = f"{api_url_base}{parte}"
    
    try:
         # Realizar la solicitud GET a la API con autenticación básica
        headers = {
            'Authorization': f'Basic ZXh0ZXJuYWxfYXBpOjEwMjRtYi0xVA=='
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Verifica que la solicitud fue exitosa
        data = response.json()  # Asumiendo que la API devuelve un JSON
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud a la API: {e}")
        return None

def transform_data(data):
    # Renombrar las columnas según el diccionario de mapeo
    transformed_data = {column_mapping.get(k, k): v for k, v in data.items()}
    
    # Verificar si el tipo de producto es "ROLLO"
    if transformed_data['TipoProducto'].upper() == "ROLLO":
        # Asignar "NO APLICA" a los campos específicos cuando es un "ROLLO"
        transformed_data['TipoBolsa'] = 'NO APLICA'
        transformed_data['TipoSellado'] = 'NO APLICA'
        transformed_data['Valvula'] = 'NO APLICA'
        transformed_data['GradoTroquel'] = 'NO APLICA'
        transformed_data['UsoZipper'] = 'NO APLICA'
    
    # Convertir Ancho, Largo y Calibre en enteros
    transformed_data['Ancho'] = int(float(transformed_data['Ancho']))
    transformed_data['Largo'] = int(float(transformed_data['Largo']))
    transformed_data['Calibre2'] = int(float(transformed_data['Calibre2']))

    # Asignar el valor de Calibre según la nueva restricción (<=100 o >100)
    if transformed_data['Calibre2'] <= 100:
        transformed_data['Calibre'] = '<=100'
    else:
        transformed_data['Calibre'] = '>100'
    
    # Crear columna GrupoFormato (usar siempre Ancho si es ROLLO o si en el montaje contiene 'LAT')
    if transformed_data['TipoProducto'].upper() == "ROLLO" or 'LAT' in transformed_data.get('Tipo de montaje', '').upper():
        # Si es ROLLO o tiene LAT en tipo de montaje, usar el Ancho
        if transformed_data['Ancho'] >= 1000:
            transformed_data['GrupoFormato'] = 'Grupo 6'
        elif transformed_data['Ancho'] >= 800:
            transformed_data['GrupoFormato'] = 'Grupo 5'
        elif transformed_data['Ancho'] >= 600:
            transformed_data['GrupoFormato'] = 'Grupo 4'
        elif transformed_data['Ancho'] >= 400:
            transformed_data['GrupoFormato'] = 'Grupo 3'
        elif transformed_data['Ancho'] >= 200:
            transformed_data['GrupoFormato'] = 'Grupo 2'
        else:
            transformed_data['GrupoFormato'] = 'Grupo 1'
    else:
        # Si no tiene LAT, usar el Largo
        if transformed_data['Largo'] >= 1000:
            transformed_data['GrupoFormato'] = 'Grupo 6'
        elif transformed_data['Largo'] >= 800:
            transformed_data['GrupoFormato'] = 'Grupo 5'
        elif transformed_data['Largo'] >= 600:
            transformed_data['GrupoFormato'] = 'Grupo 4'
        elif transformed_data['Largo'] >= 400:
            transformed_data['GrupoFormato'] = 'Grupo 3'
        elif transformed_data['Largo'] >= 200:
            transformed_data['GrupoFormato'] = 'Grupo 2'
        else:
            transformed_data['GrupoFormato'] = 'Grupo 1'

    # Crear columna Impreso
    if 'SIN' in transformed_data.get('Tipo de impresion', '').upper() or not transformed_data.get('Tipo de impresion'):
        transformed_data['Impreso'] = 'NO'
    else:
        transformed_data['Impreso'] = 'SI'

    # Crear columna Válvula (si no es ROLLO)
    if transformed_data['TipoProducto'].upper() != "ROLLO":
        if any(keyword in transformed_data.get('Dispositivo 1', '').upper() for keyword in ['VAL', 'CONJ', 'TAPA']) or \
           any(keyword in transformed_data.get('Dispositivo 2', '').upper() for keyword in ['VAL', 'CONJ', 'TAPA']) or \
           any(keyword in transformed_data.get('Dispositivo 3', '').upper() for keyword in ['VAL', 'CONJ', 'TAPA']):
            transformed_data['Valvula'] = 'SI'
        else:
            transformed_data['Valvula'] = 'NO'

    # Crear columna UsoZipper (si no es ROLLO)
    if transformed_data['TipoProducto'].upper() != "ROLLO":
        if 'ZIPPER' in transformed_data.get('Dispositivo 1', '').upper() or \
           'ZIPPER' in transformed_data.get('Dispositivo 2', '').upper() or \
           'ZIPPER' in transformed_data.get('Dispositivo 3', '').upper():
            transformed_data['UsoZipper'] = 'SI'
        else:
            transformed_data['UsoZipper'] = 'NO'

    # Crear columna GradoTroquel (si no es ROLLO)
    if transformed_data['TipoProducto'].upper() != "ROLLO":
        troquel_columns = ['Dispositivo 1', 'Dispositivo 2', 'Dispositivo 3', 'Dispositivo 4', 'Troquel 1', 'Troquel 2']
        grado = 'NO TROQUEL'
    
        for col in troquel_columns:
            troquel_value = transformed_data.get(col, '')
            if troquel_value in troqueles_forma:
                grado = 'TROQUEL FORMA'
                break
            elif troquel_value in troqueles_alto and grado != 'TROQUEL FORMA':
                grado = 'GRADO ALTO'
            elif troquel_value in troqueles_bajo and grado not in ['TROQUEL FORMA', 'GRADO ALTO']:
                grado = 'GRADO BAJO'
    
        transformed_data['GradoTroquel'] = grado

    return transformed_data

# Función de redondeo especial
def round_half_up(n):
    return math.floor(n + 0.5)

# Configurar Flask
app = Flask(__name__)

# Registrar el filtro en Jinja
app.jinja_env.filters['round_half_up'] = round_half_up

# Configurar el logger
logging.basicConfig(filename='./logs/predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo entrenado
model = joblib.load('./models/optimized_XGB_model2.pkl')

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    part_data = None
    incremento = None

    if request.method == "POST":
        parte = request.form.get("Parte")
        
        # Obtener y transformar los datos
        api_data = get_api_data(parte)
        if api_data:
            first_record = api_data['value'][0] if 'value' in api_data and api_data['value'] else None
            if first_record:
                part_data = transform_data(first_record)
        
        if part_data:
            # Usar los datos transformados para llenar el formulario
            input_data = {
                "Estructura": [part_data.get("Estructura")],
                "TipoProducto": [part_data.get("TipoProducto")],
                "TipoBolsa": [part_data.get("TipoBolsa")],
                "TipoSellado": [part_data.get("TipoSellado")],
                "GrupoFormato": [part_data.get("GrupoFormato")],
                "Impreso": [part_data.get("Impreso")],
                "Calibre": [part_data.get("Calibre")],
                "Valvula": [part_data.get("Valvula")],
                "GradoTroquel": [part_data.get("GradoTroquel")],
                "UsoZipper": [part_data.get("UsoZipper")],
            }

            input_df = pd.DataFrame(input_data)
            input_df = pd.get_dummies(input_df)

            # Encontrar las columnas faltantes
            missing_cols = list(set(model.feature_names_in_) - set(input_df.columns))

            # Crear un DataFrame con las columnas faltantes y valores de 0
            missing_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)

            # Concatenar el DataFrame original con las columnas faltantes
            input_df = pd.concat([input_df, missing_df], axis=1)

            # Asegurarse de que las columnas estén en el mismo orden que en el modelo entrenado
            input_df = input_df[model.feature_names_in_]

            # Hacer la predicción
            prediction = model.predict(input_df)[0]

            # Calcular el incremento basado en las reglas de prioridad
            if part_data["TipoProducto"].upper() == "ROLLO" and "FOIL" in part_data["Estructura"].upper():
                incremento = 108 - (prediction * 100)
            elif part_data["TipoProducto"].upper() == "ROLLO":
                incremento = 105 - (prediction * 100)
            elif part_data["TipoProducto"].upper() == "BOLSA" and part_data["TipoBolsa"].upper() == "FLOW PACK":
                incremento = 108 - (prediction * 100)
            elif part_data["TipoProducto"].upper() == "BOLSA" and part_data["UsoZipper"].upper() == "SI":
                incremento = 108 - (prediction * 100)
            elif part_data["TipoProducto"].upper() == "BOLSA" and part_data["GradoTroquel"].upper() in ["GRADO ALTO", "TROQUEL FORMA"]:
                incremento = 107 - (prediction * 100)
            elif part_data["TipoProducto"].upper() == "BOLSA":
                incremento = 106 - (prediction * 100)
            else:
                incremento = 0

            # Si el incremento es menor o igual que cero, indicar que no hay que hacer incremento
            if incremento <= 0:
                incremento = "No se requiere incremento"
            else:
                incremento = f"{round_half_up(incremento)}%"  # Incluye el símbolo de porcentaje


            # Registrar la predicción en el archivo de log
            logging.info(f"Predicción realizada: {prediction:.2f} | Incremento sugerido: {incremento}% | Datos de entrada: {input_data}")

    return render_template("index.html", prediction=prediction, part_data=part_data, incremento=incremento)

if __name__ == "__main__":
    app.run(debug=True, port="9999")
