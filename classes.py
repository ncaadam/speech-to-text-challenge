import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import spacy

class Preprocessor(object):
    input_path = None
    raw_data = None
    clean_data = None
    processed_df = None
    metadata_df = None

    def __init__(self,input_path):
        self.input_path = input_path
    
    def _load_data(self):
        with open(self.input_path) as f:
            raw_data = f.readlines()
        self.raw_data = raw_data
        return self.raw_data

    def _clean_up(self):
        clean_data = list()
        for line in self.raw_data:
            sentence, labels = line.split("\t")
            sentence = sentence[3:-3].strip()
            labels = labels.split(" ")[1:]
            target = labels[-1].replace("\n","")
            labels = labels[:-1]
            clean_data.append([sentence, labels, target])
        self.clean_data = clean_data
        return self.clean_data

    def _create_dataframe(self):
        self.processed_df = pd.DataFrame(self.clean_data,columns=["sentence","labels","target"]).reset_index().rename(columns={"index":"sentence_num"})
        return self.processed_df

    def _create_metadata_features(self):
        metadata_df = self.processed_df.copy()
        metadata_df["words"] = metadata_df.sentence.str.split(" ")
        metadata_df["num_labels"] = metadata_df.labels.map(len)
        metadata_df["num_words"] = metadata_df.words.map(len)
        metadata_df["target_atis_flight_flag"] = (metadata_df.target == "atis_flight").astype(int)
        metadata_df["num_targets"] = metadata_df.target.map(lambda x: len(x.split("#")))
        metadata_df["multi_target_flag"] = (metadata_df.num_targets > 1).astype(int)
        self.metadata_df = metadata_df.copy()
        return self.metadata_df
    
    def preprocess_data(self):
        self.raw_data = self._load_data()
        self.clean_data = self._clean_up()
        self.processed_df = self._create_dataframe()
        self.metadata_df = self._create_metadata_features()
        return self.metadata_df
    
    def write_to_disk(self,output_path,fmt='binary'):
        if self.metadata_df is None:
            raise ValueError("Metadata dataframe not found. Execute preprocess function.")
        else:
            if fmt == "binary":
                self.metadata_df.to_pickle(output_path)
            elif fmt == "csv":
                self.metadata_df.to_csv(output_path,index=False)
            else:
                raise NotImplementedError("Save format must be 'binary' or 'csv'")


class Modeler(object):
    input_data_path = None
    input_df = None
    nlp = None
    initial_model = None
    initial_scoring_df = None
    final_model = None
    final_scoring_df = None
    final_predictions = None


    def __init__(self,input_data_path):
        self.inital_model = joblib.load("./models/primary_target_model.pkl")
        self.final_model = joblib.load("./models/final_target_model.pkl")
        self.input_data_path = input_data_path
        self.nlp = spacy.load("en")

    def _load_data(self,fmt='binary'):
        if self.input_df is not None:
            return self.input_df
        else:
            if fmt == "binary":
                input_df = pd.read_pickle(self.input_data_path)
                return input_df
            elif fmt == "csv":
                input_df = pd.read_csv(self.input_data_path)
                return input_df
            else:
                raise NotImplementedError("Load format must be 'binary' or 'csv'")

    def _create_features(self):
        self.input_df["nlp_data"] = self.input_df.sentence.map(self.nlp)
        self.input_df["nlp_vector"] = self.input_df.nlp_data.map(lambda x: x.vector)
        self.input_df["nlp_norm_vector"] = self.input_df.nlp_data.map(lambda x: x.vector_norm)
        temp_df = pd.DataFrame([x for x in self.input_df["nlp_vector"]], columns = ["array_value_{}".format(i) for i,_ in enumerate(self.input_df.nlp_vector[0])])
        initial_scoring_df = pd.merge(temp_df,self.input_df[["nlp_norm_vector","target_atis_flight_flag"]],left_index=True,right_index=True)
        return initial_scoring_df

    def _make_initial_predictions(self):
        proba_predictions = pd.DataFrame(self.inital_model.predict_proba(self.initial_scoring_df.loc[:,~self.initial_scoring_df.columns.isin(["target_atis_flight_flag"])]),columns=["prob_0","prob_1"])
        merged_predictions = pd.merge(proba_predictions,self.initial_scoring_df.loc[:,["target_atis_flight_flag"]],left_index=True,right_index=True)
        final_scoring_df = pd.merge(self.input_df[["target"]],merged_predictions[["prob_1"]],left_index=True,right_index=True).rename(columns={"prob_1":"prob_primary_target"})
        final_scoring_df = pd.merge(final_scoring_df,self.initial_scoring_df.loc[:,~self.initial_scoring_df.columns.isin(["target_atis_flight_flag"])],left_index=True,right_index=True)
        return final_scoring_df

    def _make_final_predictions(self):
        final_predictions = pd.DataFrame(self.final_model.predict(self.final_scoring_df.loc[:,~self.final_scoring_df.columns.isin(["target"])]),columns=["prediction"])
        merged_predictions = pd.merge(final_predictions,self.final_scoring_df.loc[:,"target"],left_index=True,right_index=True)
        return merged_predictions

    def score_dataset(self):
        self.input_df = self._load_data()
        self.initial_scoring_df = self._create_features()
        self.final_scoring_df = self._make_initial_predictions()
        self.final_predictions = self._make_final_predictions()

    def get_classification_report(self):
        return classification_report(self.final_predictions["target"],self.final_predictions["prediction"])