from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import numpy as np
import pdmlabs as OnlineADEngine
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences


class CoxPH(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, seq_length=10,save_model=False,normalize=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model
        self.saved_model_path=None

        self.seq_length = seq_length
        self.normalize = normalize

    def normalize_columns(self,A):
        """
        Normalize a 2D numpy array column-wise.
        If a column is constant (std = 0), the output is zeros for that column.
        """
        A = np.asarray(A, dtype=float)

        mean = A.mean(axis=0)
        std = A.std(axis=0)

        # Identify constant columns
        constant = (std == 0)

        # Avoid division by zero
        std_safe = std.copy()
        std_safe[constant] = 1.0  # dummy value (won't be used)

        # Normalize
        A_norm = (A - mean) / std_safe

        # Set constant columns to zero
        A_norm[:, constant] = 0.0

        return A_norm

    def create_windowed_data(self,df: pd.DataFrame, seq_len: int, normalize: bool = True):
        feature_cols = [c for c in df.columns if c not in ["vehicle_id", "label", "event"]]

        all_X, all_y = [], []
        all_e = []
        for vid, vdf in df.groupby("vehicle_id"):
            vdf = vdf.reset_index(drop=True)
            vdf[feature_cols] = vdf[feature_cols].astype("float32")

            data = vdf[feature_cols].values
            labels = vdf["label"].values
            events = vdf["event"].values

            padded = np.pad(data, ((seq_len - 1, 0), (0, 0)), mode="edge")

            X = np.stack([padded[i:i + seq_len] for i in range(len(data))])
            # normalize

            if normalize:
                X = self.normalize_columns(X)
            all_X.append(X)
            all_y.append(labels)
            all_e.append(events)

        X_final = np.concatenate(all_X)
        y_final = np.concatenate(all_y)
        e_final = np.concatenate(all_e)

        # if np.isnan(y_final.numpy()).any():
        #     print("❌ Inf inside windowed X")
        # if np.isinf(y_final.numpy()).any():
        #     print("❌ Inf inside windowed X")
        return X_final, y_final, e_final


    def fit(self, historic_data: list[pd.DataFrame], historic_sources: list[str], event_data: pd.DataFrame,
            anomaly_ranges: list[list]) -> None:
        """
        This method is used to fit a anomaly detection model in supervised way (training), where the data are passed in form
        of Dataframes along with their respected source and labels.

        :param historic_data: a list of Dataframes (used to fit a semi-supervised model). The `historic_data` list parameter elements should be copied if a corresponding method needs to store them for future processing
        :param historic_sources: a list with strings (names) of the different sources
        :param event_data: event data that are produced from the different sources
        :param anomaly_ranges: labels regarding if the data are normal or not. It is a list of lists, where each inner list corresponds to a source and contains the labels for the data in that source.
        :return: None.
        """

        for current_historic_data, current_historic_source, labels in zip(historic_data, historic_sources,
                                                                          anomaly_ranges):
            from sksurv.util import Surv

            df = current_historic_data.copy()
            df["label"] = [lb[0] for lb in labels]
            df["event"] = [lb[1] for lb in labels]
            X_final, y_final, e_final = self.create_windowed_data(df, self.seq_length, self.normalize)
            X_final = X_final.reshape(X_final.shape[0], -1)

            print(X_final.shape)

            ydf = pd.DataFrame({'event': [lb for lb in e_final], 'RUL': [lb for lb in y_final]})
            y = Surv.from_dataframe("event", "RUL", ydf)

            # RandomSurvivalForest(n_estimators=100, min_samples_split=6, min_samples_leaf=5, verbose=1, n_jobs=4)
            self.model_per_source[current_historic_source] = CoxPHSurvivalAnalysis(*self.initial_args,**self.initial_kwargs)
            self.model_per_source[current_historic_source].fit(X_final, y)
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in y['RUL']])



    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        # TODO need to check if a model is available for the provided source
        target_data = target_data.copy()
        target_data["label"] = 0
        target_data["event"] = 0
        X_final, y_final, e_final = self.create_windowed_data(target_data, self.seq_length, self.normalize)
        X_final = X_final.reshape(X_final.shape[0], -1)

        target_data_t = X_final.astype('float32')

        predictions = self.model_per_source[source].predict_survival_function(target_data_t, True)
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result

    def predict_one(self, new_sample: pd.Series, source: str, is_event: bool) -> float:

        new_sample=self.normalize_columns(new_sample).reshape(1,-1) if self.normalize else new_sample.reshape(1,-1)

        predictions = self.model_per_source[source].predict_survival_function(new_sample, True)
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result[0]


    def get_params(self) -> dict:
        params = {}
        for i, arg in enumerate(self.initial_args):
            params[f"arg{i}"] = arg
        # include keyword args normally
        params["seq_length"] = self.seq_length
        params["normalize"] = self.normalize
        params["model_path"] = self.saved_model_path
        params.update(self.initial_kwargs)

        return params

    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "CoxPH"

    def get_all_models(self):
        pass