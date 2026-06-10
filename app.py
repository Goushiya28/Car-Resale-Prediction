"""
AutoValue AI — Flask Backend
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib, json, numpy as np, datetime, os

app = Flask(__name__)

# ── Load model & metadata once at startup ────────────────────────────────
BASE = os.path.dirname(__file__)
model    = joblib.load(os.path.join(BASE, 'models', 'xgboost_car_price_model.pkl'))
features = json.load(open(os.path.join(BASE, 'models', 'feature_columns.json')))

# ── Label encoding maps (must match training exactly) ────────────────────
BRAND_ENC = {
    'Audi':0,'BMW':1,'Ford':2,'Honda':3,'Hyundai':4,'Jeep':5,
    'Kia':6,'Lexus':7,'MG':8,'Mahindra':9,'Maruti':10,'Nissan':11,
    'Renault':12,'Skoda':13,'Tata':14,'Volkswagen':15
}
COUNTRY_ENC = {
    'China/UK':0,'Czech Republic':1,'France':2,'Germany':3,
    'India':4,'Japan':5,'South Korea':6,'USA':7
}
FUEL_ENC = {'Cng':0,'Diesel':1,'Electric':2,'Hybrid':3,'Petrol':4}
TRANS_ENC = {'Automatic':0,'Manual':1}

# ── Brand → country mapping ───────────────────────────────────────────────
BRAND_COUNTRY = {
    'Audi':'Germany','BMW':'Germany','Ford':'USA','Honda':'Japan',
    'Hyundai':'South Korea','Jeep':'USA','Kia':'South Korea','Lexus':'Japan',
    'MG':'China/UK','Mahindra':'India','Maruti':'India','Nissan':'Japan',
    'Renault':'France','Skoda':'Czech Republic','Tata':'India','Volkswagen':'Germany'
}

# ── Typical engine_cc by (brand, fuel) ───────────────────────────────────
ENGINE_CC = {
    ('Audi','Petrol'):1984,('BMW','Diesel'):1995,
    ('Ford','Diesel'):1498,('Ford','Petrol'):1497,
    ('Honda','Cng'):1199,('Honda','Diesel'):1498,('Honda','Petrol'):1349,
    ('Hyundai','Diesel'):1493,('Hyundai','Petrol'):1497,
    ('Jeep','Diesel'):1956,('Jeep','Petrol'):1368,
    ('Kia','Diesel'):1493,('Kia','Petrol'):1353,
    ('Lexus','Petrol'):2494,('MG','Diesel'):1956,
    ('MG','Hybrid'):1451,('MG','Petrol'):1451,
    ('Mahindra','Diesel'):2184,('Mahindra','Petrol'):1997,
    ('Maruti','Diesel'):1248,('Maruti','Hybrid'):1490,('Maruti','Petrol'):1462,
    ('Nissan','Petrol'):999,('Renault','Diesel'):1461,('Renault','Petrol'):999,
    ('Skoda','Petrol'):1498,('Tata','Diesel'):1497,
    ('Tata','Electric'):0,('Tata','Petrol'):1199,('Volkswagen','Petrol'):999,
}

# ── Typical mileage by (brand, fuel) ─────────────────────────────────────
MILEAGE = {
    ('Audi','Petrol'):14.9,('BMW','Diesel'):20.3,
    ('Ford','Diesel'):22.7,('Ford','Petrol'):16.4,
    ('Honda','Cng'):16.5,('Honda','Diesel'):23.7,('Honda','Petrol'):15.9,
    ('Hyundai','Diesel'):20.4,('Hyundai','Petrol'):17.0,
    ('Jeep','Diesel'):16.7,('Jeep','Petrol'):14.1,
    ('Kia','Diesel'):19.5,('Kia','Petrol'):16.5,
    ('Lexus','Petrol'):18.3,('MG','Diesel'):17.4,
    ('MG','Hybrid'):15.8,('MG','Petrol'):13.9,
    ('Mahindra','Diesel'):15.1,('Mahindra','Petrol'):17.0,
    ('Maruti','Diesel'):23.8,('Maruti','Hybrid'):18.0,('Maruti','Petrol'):19.8,
    ('Nissan','Petrol'):17.7,('Renault','Diesel'):20.0,('Renault','Petrol'):19.1,
    ('Skoda','Petrol'):17.2,('Tata','Diesel'):23.2,
    ('Tata','Electric'):0,('Tata','Petrol'):17.5,('Volkswagen','Petrol'):20.1,
}

PREMIUM_BRANDS = {'Mercedes-Benz','BMW','Audi','Volvo','Lexus'}

def build_feature_vector(data):
    """Convert raw form input → 28-feature vector matching training order."""
    brand    = data.get('brand', 'Hyundai')
    fuel     = data.get('fuel', 'Petrol')
    trans    = data.get('trans', 'Manual')
    owner    = data.get('owner', '1st Owner')
    year     = int(data.get('year', 2021))
    km       = int(data.get('km', 45000))
    n_images = int(data.get('number_of_images', 10))

    # Condition ratings — use form values if sent, else dataset means
    core_r    = float(data.get('core_systems_rating',    9.71))
    supp_r    = float(data.get('supporting_systems_rating', 9.45))
    int_r     = float(data.get('interiors_ac_rating',    8.82))
    ext_r     = float(data.get('exteriors_lights_rating', 8.64))
    wear_r    = float(data.get('wear_tear_rating',       8.39))
    cond_score = float(data.get('overall_condition_score', 9.0))

    # Insurance: default to 12 months remaining if not provided
    insurance_months = int(data.get('insurance_months_remaining', 12))

    country   = BRAND_COUNTRY.get(brand, 'India')
    engine_cc = ENGINE_CC.get((brand, fuel), 1497)
    mileage   = MILEAGE.get((brand, fuel), 17.0)
    car_age   = max(2025 - year, 0)
    owner_num = {'1st Owner':1,'2nd Owner':2,'3rd Owner':3}.get(owner, 1)

    # Engineered features (must match Cell 6 exactly)
    log_km             = np.log1p(km)
    km_per_year        = km / (car_age + 1)
    age_sq             = car_age ** 2
    cc_per_km          = engine_cc / (km + 1)
    condition_age_ratio= cond_score / (car_age + 1)
    is_premium         = int(brand in PREMIUM_BRANDS)
    is_electric        = int(fuel == 'Electric')
    is_diesel          = int(fuel == 'Diesel')
    is_automatic       = int(trans == 'Automatic')
    insurance_active   = int(insurance_months > 0)

    row = {
        'brand':                    BRAND_ENC.get(brand, 4),
        'country':                  COUNTRY_ENC.get(country, 4),
        'year':                     year,
        'car_age':                  car_age,
        'km_driven_clean':          km,
        'fuel_type':                FUEL_ENC.get(fuel, 4),
        'transmission':             TRANS_ENC.get(trans, 1),
        'number_of_images':         n_images,
        'core_systems_rating':      core_r,
        'supporting_systems_rating':supp_r,
        'interiors_ac_rating':      int_r,
        'exteriors_lights_rating':  ext_r,
        'wear_tear_rating':         wear_r,
        'overall_condition_score':  cond_score,
        'engine_cc_clean':          engine_cc,
        'mileage_kmpl_clean':       mileage,
        'insurance_months_remaining': insurance_months,
        'owner_num':                owner_num,
        'log_km':                   log_km,
        'km_per_year':              km_per_year,
        'age_sq':                   age_sq,
        'cc_per_km':                cc_per_km,
        'condition_age_ratio':      condition_age_ratio,
        'is_premium':               is_premium,
        'is_electric':              is_electric,
        'is_diesel':                is_diesel,
        'is_automatic':             is_automatic,
        'insurance_active':         insurance_active,
    }

    # Return in exact feature order used during training
    return np.array([row[f] for f in features]).reshape(1, -1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        X = build_feature_vector(data)

        # Predict (model was trained on log-price)
        log_pred = model.predict(X)[0]
        price    = float(np.expm1(log_pred))

        # Confidence: fixed 87% for XGBoost on this dataset (Test R²=0.85)
        conf = 87

        # Price range ±8%
        low  = price * 0.92
        high = price * 1.08

        # EV ROI %: typical new EV ~₹15L
        ev_pct = round((price / 1_500_000) * 100, 1)

        def fmt(v):
            if v >= 1_00_000:
                return f"₹{v/1_00_000:.2f}L"
            return f"₹{int(v):,}"

        return jsonify({
            'price':   fmt(price),
            'low':     fmt(low),
            'high':    fmt(high),
            'conf':    conf,
            'ev_pct':  ev_pct,
            'raw':     int(price)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("  AutoValue AI — Flask Server")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
