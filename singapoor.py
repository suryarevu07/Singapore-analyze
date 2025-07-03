from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import logging


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend


model_path = r"D:\resale_price_model (1).pkl"
scaler_path = r"D:\scaler (8).pkl"
csv_path = r"D:\newhdfselectedcolumns (3).csv"


model = None
scaler = None
data = pd.DataFrame()


town_options = [
    'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH', 'CENTRAL AREA',
    'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA',
    'MARINE PARADE', 'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
    'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL'
]
town_mapping = {town: i for i, town in enumerate(town_options)}

street_options = [
    'ANG MO KIO AVE 1', 'ANG MO KIO AVE 3', 'ANG MO KIO AVE 4', 'ANG MO KIO AVE 10', 'ANG MO KIO AVE 5',
    'ANG MO KIO AVE 8', 'ANG MO KIO AVE 6', 'ANG MO KIO AVE 9', 'ANG MO KIO AVE 2', 'BEDOK RESERVOIR RD',
    'BEDOK NTH ST 3', 'BEDOK STH RD', 'NEW UPP CHANGI RD', 'BEDOK NTH RD', 'BEDOK STH AVE 1', 'CHAI CHEE RD',
    'CHAI CHEE DR', 'BEDOK NTH AVE 4', 'BEDOK STH AVE 3', 'BEDOK STH AVE 2', 'BEDOK NTH ST 2', 'BEDOK NTH ST 4',
    'BEDOK NTH AVE 2', 'BEDOK NTH AVE 3', 'BEDOK NTH AVE 1', 'BEDOK NTH ST 1', 'CHAI CHEE ST', 'SIN MING RD',
    'SHUNFU RD', 'BT BATOK ST 11', 'BT BATOK WEST AVE 8', 'BT BATOK WEST AVE 6', 'BT BATOK ST 21',
    'BT BATOK EAST AVE 5', 'BT BATOK EAST AVE 4', 'HILLVIEW AVE', 'BT BATOK CTRL', 'BT BATOK ST 31',
    'BT BATOK EAST AVE 3', 'TAMAN HO SWEE', 'TELOK BLANGAH CRES', 'BEO CRES', 'TELOK BLANGAH DR', 'DEPOT RD',
    'TELOK BLANGAH RISE', 'JLN BT MERAH', 'HENDERSON RD', 'INDUS RD', 'BT MERAH VIEW', 'HENDERSON CRES',
    'BT PURMEI RD', 'TELOK BLANGAH HTS', 'EVERTON PK', 'KG BAHRU HILL', 'REDHILL CL', 'HOY FATT RD',
    'HAVELOCK RD', 'JLN KLINIK', 'JLN RUMAH TINGGI', 'JLN BT HO SWEE', 'KIM CHENG ST', 'MOH GUAN TER',
    'TELOK BLANGAH WAY', 'KIM TIAN RD', 'KIM TIAN PL', 'EMPRESS RD', "QUEEN'S RD", 'FARRER RD', 'JLN KUKOH',
    'OUTRAM PK', 'SHORT ST', 'SELEGIE RD', 'UPP CROSS ST', 'WATERLOO ST', 'QUEEN ST', 'BUFFALO RD',
    'ROWELL RD', 'ROCHOR RD', 'BAIN ST', 'SMITH ST', 'VEERASAMY RD', 'TECK WHYE AVE', 'TECK WHYE LANE',
    'CLEMENTI AVE 3', 'WEST COAST DR', 'CLEMENTI AVE 2', 'CLEMENTI AVE 5', 'CLEMENTI AVE 4', 'CLEMENTI AVE 1',
    'WEST COAST RD', 'CLEMENTI WEST ST 1', 'CLEMENTI WEST ST 2', 'CLEMENTI ST 13', "C'WEALTH AVE WEST",
    'CLEMENTI AVE 6', 'CLEMENTI ST 14', 'CIRCUIT RD', 'MACPHERSON LANE', 'JLN PASAR BARU', 'GEYLANG SERAI',
    'EUNOS CRES', 'SIMS DR', 'ALJUNIED CRES', 'GEYLANG EAST AVE 1', 'DAKOTA CRES', 'PINE CL', 'HAIG RD',
    'BALAM RD', 'JLN DUA', 'GEYLANG EAST CTRL', 'EUNOS RD 5', 'HOUGANG AVE 3', 'HOUGANG AVE 5', 'HOUGANG AVE 1',
    'HOUGANG ST 22', 'HOUGANG AVE 10', 'LOR AH SOO', 'HOUGANG ST 11', 'HOUGANG AVE 7', 'HOUGANG ST 21',
    'TEBAN GDNS RD', 'JURONG EAST AVE 1', 'JURONG EAST ST 32', 'JURONG EAST ST 13', 'JURONG EAST ST 21',
    'JURONG EAST ST 24', 'JURONG EAST ST 31', 'PANDAN GDNS', 'YUNG KUANG RD', 'HO CHING RD', 'HU CHING RD',
    'BOON LAY DR', 'BOON LAY AVE', 'BOON LAY PL', 'JURONG WEST ST 52', 'JURONG WEST ST 41',
    'JURONG WEST AVE 1', 'JURONG WEST ST 42', 'JLN BATU', "ST. GEORGE'S RD", 'NTH BRIDGE RD', 'FRENCH RD',
    'BEACH RD', 'WHAMPOA DR', 'UPP BOON KENG RD', 'BENDEMEER RD', 'WHAMPOA WEST', 'LOR LIMAU',
    'KALLANG BAHRU', 'GEYLANG BAHRU', 'DORSET RD', 'OWEN RD', 'KG ARANG RD', 'JLN BAHAGIA', 'MOULMEIN RD',
    'TOWNER RD', 'JLN RAJAH', 'KENT RD', 'AH HOOD RD', "KING GEORGE'S AVE", 'CRAWFORD LANE', 'MARINE CRES',
    'MARINE DR', 'MARINE TER', "C'WEALTH CL", "C'WEALTH DR", 'TANGLIN HALT RD', "C'WEALTH CRES", 'DOVER RD',
    'MARGARET DR', 'GHIM MOH RD', 'DOVER CRES', 'STIRLING RD', 'MEI LING ST', 'HOLLAND CL', 'HOLLAND AVE',
    'HOLLAND DR', 'DOVER CL EAST', 'SELETAR WEST FARMWAY 6', 'LOR LEW LIAN', 'SERANGOON NTH AVE 1',
    'SERANGOON AVE 2', 'SERANGOON AVE 4', 'SERANGOON CTRL', 'TAMPINES ST 11', 'TAMPINES ST 21',
    'TAMPINES ST 91', 'TAMPINES ST 81', 'TAMPINES AVE 4', 'TAMPINES ST 22', 'TAMPINES ST 12',
    'TAMPINES ST 23', 'TAMPINES ST 24', 'TAMPINES ST 41', 'TAMPINES ST 82', 'TAMPINES ST 83',
    'TAMPINES AVE 5', 'LOR 2 TOA PAYOH', 'LOR 8 TOA PAYOH', 'LOR 1 TOA PAYOH', 'LOR 5 TOA PAYOH',
    'LOR 3 TOA PAYOH', 'LOR 7 TOA PAYOH', 'TOA PAYOH EAST', 'LOR 4 TOA PAYOH', 'TOA PAYOH CTRL',
    'TOA PAYOH NTH', 'POTONG PASIR AVE 3', 'POTONG PASIR AVE 1', 'UPP ALJUNIED LANE', 'JOO SENG RD',
    'MARSILING LANE', 'MARSILING DR', 'MARSILING RISE', 'MARSILING CRES', 'WOODLANDS CTR RD',
    'WOODLANDS ST 13', 'WOODLANDS ST 11', 'YISHUN RING RD', 'YISHUN AVE 5', 'YISHUN ST 72', 'YISHUN ST 11',
    'YISHUN ST 21', 'YISHUN ST 22', 'YISHUN AVE 3', 'CHAI CHEE AVE', 'ZION RD', 'LENGKOK BAHRU',
    'SPOTTISWOODE PK RD', 'NEW MKT RD', 'TG PAGAR PLAZA', 'KELANTAN RD', 'PAYA LEBAR WAY', 'UBI AVE 1',
    'SIMS AVE', 'YUNG PING RD', 'TAO CHING RD', 'GLOUCESTER RD', 'BOON KENG RD', 'WHAMPOA STH',
    'CAMBRIDGE RD', 'TAMPINES ST 42', 'LOR 6 TOA PAYOH', 'KIM KEAT AVE', 'YISHUN AVE 6', 'YISHUN AVE 9',
    'YISHUN ST 71', 'BT BATOK ST 32', 'SILAT AVE', 'TIONG BAHRU RD', 'SAGO LANE', "ST. GEORGE'S LANE",
    'LIM CHU KANG RD', "C'WEALTH AVE", "QUEEN'S CL", 'SERANGOON AVE 3', 'POTONG PASIR AVE 2',
    'WOODLANDS AVE 1', 'YISHUN AVE 4', 'LOWER DELTA RD', 'NILE RD', 'JLN MEMBINA BARAT', 'JLN BERSEH',
    'CHANDER RD', 'CASSIA CRES', 'OLD AIRPORT RD', 'ALJUNIED RD', 'BUANGKOK STH FARMWAY 1',
    'BT BATOK ST 33', 'ALEXANDRA RD', 'CHIN SWEE RD', 'SIMS PL', 'HOUGANG AVE 2', 'HOUGANG AVE 8',
    'SEMBAWANG RD', 'SIMEI ST 1', 'BT BATOK ST 34', 'BT MERAH CTRL', 'LIM LIAK ST', 'JLN TENTERAM',
    'WOODLANDS ST 32', 'SIN MING AVE', 'BT BATOK ST 52', 'DELTA AVE', 'PIPIT RD', 'HOUGANG AVE 4',
    'QUEENSWAY', 'YISHUN ST 61', 'BISHAN ST 12', "JLN MA'MOR", 'TAMPINES ST 44', 'TAMPINES ST 43',
    'BISHAN ST 13', 'JLN DUSUN', 'YISHUN AVE 2', 'JOO CHIAT RD', 'EAST COAST RD', 'REDHILL RD',
    'KIM PONG RD', 'RACE COURSE RD', 'KRETA AYER RD', 'HOUGANG ST 61', 'TESSENSOHN RD', 'MARSILING RD',
    'YISHUN ST 81', 'BT BATOK ST 51', 'BT BATOK WEST AVE 4', 'BT BATOK WEST AVE 2', 'JURONG WEST ST 91',
    'JURONG WEST ST 81', 'GANGSA RD', 'MCNAIR RD', 'SIMEI ST 4', 'YISHUN AVE 7', 'SERANGOON NTH AVE 2',
    'YISHUN AVE 11', 'BANGKIT RD', 'JURONG WEST ST 73', 'OUTRAM HILL', 'HOUGANG AVE 6', 'PASIR RIS ST 12',
    'PENDING RD', 'PETIR RD', 'LOR 3 GEYLANG', 'BISHAN ST 11', 'PASIR RIS DR 6', 'BISHAN ST 23',
    'JURONG WEST ST 92', 'PASIR RIS ST 11', 'YISHUN CTRL', 'BISHAN ST 22', 'SIMEI RD', 'TAMPINES ST 84',
    'BT PANJANG RING RD', 'JURONG WEST ST 93', 'FAJAR RD', 'WOODLANDS ST 81', 'CHOA CHU KANG CTRL',
    'PASIR RIS ST 51', 'HOUGANG ST 52', 'CASHEW RD', 'TOH YI DR', 'HOUGANG CTRL', 'KG KAYU RD',
    'TAMPINES AVE 8', 'TAMPINES ST 45', 'SIMEI ST 2', 'WOODLANDS AVE 3', 'LENGKONG TIGA',
    'WOODLANDS ST 82', 'SERANGOON NTH AVE 4', 'SERANGOON CTRL DR', 'BRIGHT HILL DR', 'SAUJANA RD',
    'CHOA CHU KANG AVE 3', 'TAMPINES AVE 9', 'JURONG WEST ST 51', 'YUNG HO RD', 'SERANGOON AVE 1',
    'PASIR RIS ST 41', 'GEYLANG EAST AVE 2', 'CHOA CHU KANG AVE 2', 'KIM KEAT LINK', 'PASIR RIS DR 4',
    'PASIR RIS ST 21', 'SENG POH RD', 'HOUGANG ST 51', 'JURONG WEST ST 72', 'JURONG WEST ST 71',
    'PASIR RIS ST 52', 'TAMPINES ST 32', 'CHOA CHU KANG AVE 4', 'CHOA CHU KANG LOOP', 'JLN TENAGA',
    'TAMPINES CTRL 1', 'TAMPINES ST 33', 'BT BATOK WEST AVE 7', 'JURONG WEST AVE 5', 'TAMPINES AVE 7',
    'WOODLANDS ST 83', 'CHOA CHU KANG ST 51', 'PASIR RIS DR 3', 'YISHUN CTRL 1', 'CHOA CHU KANG AVE 1',
    'WOODLANDS ST 31', 'BT MERAH LANE 1', 'PASIR RIS ST 13', 'ELIAS RD', 'BISHAN ST 24', 'WHAMPOA RD',
    'WOODLANDS ST 41', 'PASIR RIS ST 71', 'JURONG WEST ST 74', 'PASIR RIS DR 1', 'PASIR RIS ST 72',
    'PASIR RIS DR 10', 'CHOA CHU KANG ST 52', 'CLARENCE LANE', 'CHOA CHU KANG NTH 6', 'PASIR RIS ST 53',
    'CHOA CHU KANG NTH 5', 'ANG MO KIO ST 21', 'JLN DAMAI', 'CHOA CHU KANG ST 62', 'WOODLANDS AVE 5',
    'WOODLANDS DR 50', 'CHOA CHU KANG ST 53', 'TAMPINES ST 72', 'UPP SERANGOON RD', 'JURONG WEST ST 75',
    'STRATHMORE AVE', 'ANG MO KIO ST 31', 'TAMPINES ST 34', 'YUNG AN RD', 'WOODLANDS AVE 4',
    'CHOA CHU KANG NTH 7', 'ANG MO KIO ST 11', 'WOODLANDS AVE 9', 'YUNG LOH RD', 'CHOA CHU KANG DR',
    'CHOA CHU KANG ST 54', 'REDHILL LANE', 'KANG CHING RD', 'TAH CHING RD', 'SIMEI ST 5',
    'WOODLANDS DR 40', 'WOODLANDS DR 70', 'TAMPINES ST 71', 'WOODLANDS DR 42', 'SERANGOON NTH AVE 3',
    'JELAPANG RD', 'BT BATOK ST 22', 'HOUGANG ST 91', 'WOODLANDS AVE 6', 'WOODLANDS CIRCLE',
    'CORPORATION DR', 'LOMPANG RD', 'WOODLANDS DR 72', 'CHOA CHU KANG ST 64', 'BT BATOK ST 24',
    'JLN TECK WHYE', 'WOODLANDS CRES', 'WOODLANDS DR 60', 'CHANGI VILLAGE RD', 'BT BATOK ST 25',
    'HOUGANG AVE 9', 'JURONG WEST CTRL 1', 'WOODLANDS RING RD', 'CHOA CHU KANG AVE 5', 'TOH GUAN RD',
    'JURONG WEST ST 61', 'WOODLANDS DR 14', 'HOUGANG ST 92', 'CHOA CHU KANG CRES', 'SEMBAWANG CL',
    'CANBERRA RD', 'SEMBAWANG CRES', 'SEMBAWANG VISTA', 'COMPASSVALE WALK', 'RIVERVALE ST',
    'WOODLANDS DR 62', 'SEMBAWANG DR', 'WOODLANDS DR 53', 'WOODLANDS DR 52', 'RIVERVALE WALK',
    'COMPASSVALE LANE', 'RIVERVALE DR', 'SENJA RD', 'JURONG WEST ST 65', 'RIVERVALE CRES',
    'WOODLANDS DR 44', 'COMPASSVALE DR', 'WOODLANDS DR 16', 'COMPASSVALE RD', 'WOODLANDS DR 73',
    'HOUGANG ST 31', 'JURONG WEST ST 64', 'WOODLANDS DR 71', 'YISHUN ST 20', 'ADMIRALTY DR',
    'COMPASSVALE ST', 'BEDOK RESERVOIR VIEW', 'YUNG SHENG RD', 'ADMIRALTY LINK', 'SENGKANG EAST WAY',
    'ANG MO KIO ST 32', 'ANG MO KIO ST 52', 'BOON TIONG RD', 'JURONG WEST ST 62', 'ANCHORVALE LINK',
    'CANBERRA LINK', 'COMPASSVALE CRES', 'CLEMENTI ST 12', 'MONTREAL DR', 'WELLINGTON CIRCLE',
    'SENGKANG EAST RD', 'JURONG WEST AVE 3', 'ANCHORVALE LANE', 'SENJA LINK', 'EDGEFIELD PLAINS',
    'ANCHORVALE DR', 'SEGAR RD', 'FARRER PK RD', 'PUNGGOL FIELD', 'EDGEDALE PLAINS', 'ANCHORVALE RD',
    'CANTONMENT CL', 'JLN MEMBINA', 'FERNVALE LANE', 'JURONG WEST ST 25', 'CLEMENTI ST 11',
    'PUNGGOL FIELD WALK', 'KLANG LANE', 'PUNGGOL CTRL', 'JELEBU RD', 'BUANGKOK CRES',
    'WOODLANDS DR 75', 'BT BATOK WEST AVE 5', 'JELLICOE RD', 'PUNGGOL DR', 'JURONG WEST ST 24',
    'SEMBAWANG WAY', 'FERNVALE RD', 'BUANGKOK LINK', 'FERNVALE LINK', 'JLN TIGA', 'YUAN CHING RD',
    'COMPASSVALE LINK', 'MARINE PARADE CTRL', 'COMPASSVALE BOW', 'PUNGGOL RD', 'BEDOK CTRL',
    'PUNGGOL EAST', 'SENGKANG CTRL', 'CANTONMENT RD', 'PUNGGOL PL', 'SENGKANG WEST AVE',
    'TAMPINES CTRL 7', 'GHIM MOH LINK', 'SIMEI LANE', 'YISHUN ST 41', 'TELOK BLANGAH ST 31',
    'JLN KAYU', 'LOR 1A TOA PAYOH', 'PUNGGOL WALK', 'SENGKANG WEST WAY', 'BUANGKOK GREEN',
    'PUNGGOL WAY', 'YISHUN ST 31', 'TECK WHYE CRES', 'MONTREAL LINK', 'UPP SERANGOON CRES',
    'SUMANG LINK', 'SENGKANG EAST AVE', 'YISHUN AVE 1', 'ANCHORVALE CRES', 'ANCHORVALE ST',
    'TAMPINES CTRL 8', 'YISHUN ST 51', 'UPP SERANGOON VIEW', 'TAMPINES AVE 1', 'BEDOK RESERVOIR CRES',
    'ANG MO KIO ST 61', 'DAWSON RD', 'FERNVALE ST', 'HOUGANG ST 32', 'TAMPINES ST 86', 'SUMANG WALK',
    'CHOA CHU KANG AVE 7', 'KEAT HONG CL', 'JURONG WEST CTRL 3', 'KEAT HONG LINK', 'ALJUNIED AVE 2',
    'CANBERRA CRES', 'SUMANG LANE', 'CANBERRA ST', 'ANG MO KIO ST 44', 'ANG MO KIO ST 51',
    'BT BATOK EAST AVE 6', 'BT BATOK WEST AVE 9', 'CANBERRA WALK', 'WOODLANDS RISE', 'TAMPINES ST 61',
    'YISHUN ST 43', 'CANBERRA VIEW', 'SENGKANG WEST RD', 'TAMPINES NTH DR 1', 'ALKAFF CRES',
    'BIDADARI PK DR', 'BT BATOK ST 41', 'NORTHSHORE DR', 'YISHUN ST 42', 'YISHUN ST 44'
]
street_mapping = {street: i for i, street in enumerate(street_options)}

flat_type_mapping = {}
flat_model_mapping = {}


try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.debug(f"Model loaded from {model_path}, type: {type(model).__name__}")
    else:
        logger.error(f"Model file not found at {model_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.debug(f"Scaler loaded from {scaler_path}, features: {getattr(scaler, 'feature_names_in_', 'Not available')}")
    else:
        logger.error(f"Scaler file not found at {scaler_path}")

    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)
        required_columns = ['town', 'flat_type', 'street_name', 'floor_area_sqm', 'flat_model', 'lease_remaining', 'Year', 'Month']
        if not data.empty and all(col in data.columns for col in required_columns):
            logger.debug(f"Loaded CSV data sample: {data.head().to_dict() or 'Empty DataFrame'}")
            flat_type_mapping = {v: i for i, v in enumerate(sorted(data['flat_type'].unique()))}
            flat_model_mapping = {v: i for i, v in enumerate(sorted(data['flat_model'].unique()))}
            # Generate town-street mapping from CSV
            town_street_mapping = {town: sorted(data[data['town'] == town]['street_name'].unique().tolist()) for town in town_options if town in data['town'].unique()}
        else:
            logger.error(f"CSV is empty or missing required columns. Expected: {required_columns}, Found: {data.columns.tolist()}")
    else:
        logger.error(f"CSV file not found at {csv_path}")

    logger.debug(f"Generated flat_type_mapping: {flat_type_mapping}")
    logger.debug(f"Generated flat_model_mapping: {flat_model_mapping}")
    logger.debug(f"Generated town_street_mapping: {town_street_mapping}")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy' if model and scaler else 'unhealthy', 'message': 'Backend is running'}), 200

@app.route('/get_town_streets', methods=['POST'])
def get_town_streets():
    try:
        data = request.get_json()
        town_idx = data.get('town_idx')
        if town_idx is None or not 0 <= town_idx < len(town_options):
            return jsonify({'error': f'Invalid town index. Must be 0-{len(town_options)-1}.'}), 400
        selected_town = town_options[town_idx]
        streets = town_street_mapping.get(selected_town, [])
        street_indices = [street_options.index(street) for street in streets if street in street_options]
        return jsonify({'streets': street_indices})
    except Exception as e:
        logger.error(f"Error fetching town streets: {str(e)}")
        return jsonify({'error': f'Failed to fetch streets: {str(e)}'}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model or scaler not loaded. Please provide valid files at specified paths.'}), 500

    try:
        data = request.get_json()
        features = data.get('features')

        if not features or len(features) != 8:
            return jsonify({'error': 'Invalid or missing feature array. Expected 8 values.'}), 400

        flat_type_idx, floor_area, year, town_idx, street_idx, flat_model_idx, lease_remaining, month = features

        if not town_options or not 0 <= town_idx < len(town_options):
            return jsonify({'error': f'Invalid town index. Must be 0-{len(town_options)-1}.'}), 400
        if not street_options or not 0 <= street_idx < len(street_options):
            return jsonify({'error': f'Invalid street index. Must be 0-{len(street_options)-1}.'}), 400

        selected_town = town_options[town_idx]
        selected_street = street_options[street_idx]
        flat_type_encoded = flat_type_mapping.get(data['flat_type_options'][flat_type_idx] if 'flat_type_options' in data else flat_type_idx, 0)
        town_encoded = town_mapping.get(selected_town, 0)
        street_name_encode = street_mapping.get(selected_street, 0)
        flat_model_encoded = flat_model_mapping.get(data['flat_model_options'][flat_model_idx] if 'flat_model_options' in data else flat_model_idx, 0)

     
        feature_names = ['flat_type_encoded', 'floor_area_sqm', 'Year', 'town_encoded', 'street_name_encode', 'flat_model_encoded', 'lease_remaining', 'Month']
        input_features = np.array([
            flat_type_encoded,
            floor_area,
            year,
            town_encoded,
            street_name_encode,
            flat_model_encoded,
            lease_remaining,
            month
        ]).reshape(1, -1)
        input_df = pd.DataFrame(input_features, columns=feature_names)

        scaled_features = scaler.transform(input_df)
        log_pred = model.predict(scaled_features)[0]
        price = np.expm1(log_pred)

        return jsonify({'price': round(price, 2)})

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}. Ensure backend is running and data is valid.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
