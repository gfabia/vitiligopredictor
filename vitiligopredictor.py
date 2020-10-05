# Vitiligo Predictor 
# Glenn G. Fabia (c) 2020

import json
import numpy as np
import pickle
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

class VitiligoPredictor:
    def __init__(self, model="adaboost", show_messages=False):
        
        self.debug = show_messages

        if self.debug:
            print("Initializing model...")  

        self.features_transformer = ColumnTransformer(
            [('binaries', OrdinalEncoder(dtype="int8"), ['sex', 'history']),
            ('categoricals', OneHotEncoder(dtype="int8", categories=(("I","II","III","IV","V"),)), ['skin_type',]),
            ('numericals', StandardScaler(), ['age', 'reading'])
            ], 
            remainder="drop"
        )

        self.label_transformer = OrdinalEncoder(dtype="int8")

    def load_feature_transformer(self, filename="feature-transformer.sav"):
        if self.debug:
            print("Loading trained model '{}'...".format(filename))
        self.features_transformer = pickle.load(open(filename, 'rb'))
    
    def load_label_transformer(self, filename="label-transformer.sav"):
        if self.debug:
            print("Loading trained model '{}'...".format(filename))
        self.label_transformer = pickle.load(open(filename, 'rb'))
    
    def load_trained_model(self, filename="model.sav"):
        if self.debug:
            print("Loading trained model '{}'...".format(filename))
        self.model = pickle.load(open(filename, 'rb'))

    def predict(self, X):
        pred = self.model.predict(X)
        return pred        

    def is_valid(self, df):
        required_columns = [
            {"column_name": "id", "type" : str},
            {"column_name": "age", "type" : int, "min_value" : 0},
            {"column_name": "sex", "type" : str, "choices" : ["F", "M"]},
            {"column_name": "history", "type" : str, "choices" : ["Yes", "No"]},
            {"column_name": "skin_type", "type" : str, "choices" : ["I", "II", "III", "IV", "V"]},
            {"column_name": "reading", "type" : int, "min_value" : 0 },
        ]
        for feature in required_columns:
            if feature["column_name"] not in df.columns:
                raise Exception("Column '{}' is required.".format(feature["column_name"]))
                return False

        for feature in required_columns:
            for index, row in df.iterrows():
                if type(row[feature["column_name"]]) != feature["type"]:
                    raise Exception("In '{}', expected '{}' but got '{}'.".format(row["id"], feature["type"], type(row[feature["column_name"]])))
                    return False
                if feature["column_name"] != "id" and type(row[feature["column_name"]]) is str and row[feature["column_name"]] not in feature["choices"]:
                    raise Exception("In '{}', got an invalid datum '{}'.".format(row["id"], row[feature["column_name"]]))
                    return False
                if type(row[feature["column_name"]]) is int and row[feature["column_name"]] < feature["min_value"]:
                    raise Exception("In '{}', got an invalid datum '{}'.".format(row["id"], row[feature["column_name"]]))
                    return False
        return True

    def json_to_features(self, data):
        df = pd.read_json(data, orient="records")
        return (df, self.features_transformer.fit_transform(df)) if self.is_valid(df) else (None, None)

    def predict_json(self, data):    
        df, features = self.json_to_features(data)
        categories = self.label_transformer.categories_[0]
        predictions = pd.Series( [ list(categories)[x] for x in self.predict(features)] )
        df["prediction"] = predictions
        result = df.sort_values(by=["id"]).to_dict(orient="records")
        return result

def main(args):

    if len(args) != 2:
        print("Usage: {} data".format(args[0]))

        print("\nWhere 'data' must be in JSON format. For example:",
        "'[{\"id\": \"R89\", \"age\": 19, \"sex\": \"M\", \"history\": \"No\", \"skin_type\": \"IV\", \"reading\": 75}]'",        
        "\nRequired parameters are:",
        "  * 'id' (string)", 
        "  * 'age' (integer)",
        "  * 'sex' (string; 'M' or 'F')",
        "  * 'history' (string; 'Yes' or 'No'),",
        "  * 'skin_type' (string; must be 'I', 'II', 'III', 'IV', or 'V'), and", 
        "  * 'reading' (integer)", sep="\n")
        return

    try:
        V = VitiligoPredictor()
        V.load_trained_model(filename="trained-models/adaboost-model.sav")
        V.load_feature_transformer(filename="trained-models/features-transformer.sav")
        V.load_label_transformer(filename="trained-models/label-transformer.sav")

        json_str = args[1]
        result = V.predict_json(json_str)
        print(json.dumps({"success" : True, "data" : result }))

    except Exception as e:
        print(json.dumps({"success" : False, "error" : str(e) }))




