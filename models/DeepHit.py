import pandas as pd
import numpy as np
import pdmlabs as OnlineADEngine
from OnlineADEngine.method.supervised_method import SupervisedMethodInterface
from OnlineADEngine.pdm_evaluation_types.types import EventPreferences
from pycox.models import DeepHitSingle
import torchtuples as tt # Some useful functions
from sklearn.preprocessing import StandardScaler




class DeepHIT(SupervisedMethodInterface):
    def __init__(self, event_preferences: EventPreferences, seq_length=10,save_model=False,num_nodes=[32, 32],batch_norm=True,
                 dropout=0.1,batch_size=256,learning_rate=0.01,epochs=100,normalize=False, *args, **kwargs):
        super().__init__(event_preferences=event_preferences)
        self.model_per_source = {}
        self.avail_times_per_source = {}
        self.num_durations = {}
        self.initial_args = args
        self.initial_kwargs = kwargs
        self.save_model = save_model
        self.scalers = {}

        self.num_nodes= num_nodes
        self.normalize= normalize
        self.batch_norm=batch_norm
        self.dropout=dropout
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.seq_length=seq_length

        self.saved_model_path = None

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

            # windowing and normalization
            df = current_historic_data.copy()
            df["label"] = [lb[0] for lb in labels]
            df["event"] = [lb[1] for lb in labels]
            X_final, y_final, e_final =self.create_windowed_data(df, self.seq_length, self.normalize)
            X_final = X_final.reshape(X_final.shape[0], -1)


            print(X_final.shape)


            ydf=pd.DataFrame({'event': [lb for lb in e_final],'RUL': [lb for lb in y_final]})

            self.num_durations[current_historic_source] = len(np.unique(ydf['RUL']))
            labtrans = DeepHitSingle.label_transform(self.num_durations[current_historic_source])
            get_target = lambda ydf: (ydf['RUL'].values, ydf['event'].values)

            y_train = labtrans.fit_transform(*get_target(ydf))
            x_train = X_final.astype('float32')
            scalerr=StandardScaler()
            x_train=scalerr.fit_transform(x_train)
            self.scalers[current_historic_source]=scalerr
            # train = (x_train, y_train)


            num_nodes = self.num_nodes
            batch_norm = self.batch_norm
            dropout = self.dropout
            batch_size = self.batch_size
            learning_rate=self.learning_rate
            epochs = self.epochs

            in_features = x_train.shape[1]
            out_features = labtrans.out_features
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
            model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts,device = None)
            model.optimizer.set_lr(learning_rate)
            # callbacks = [tt.callbacks.EarlyStopping()]
            log = model.fit(x_train, y_train, batch_size, epochs)

            self.model_per_source[current_historic_source]=model
            self.avail_times_per_source[current_historic_source]=np.unique([ty for ty in ydf['RUL']])

    def predict(self, target_data: pd.DataFrame, source: str, event_data: pd.DataFrame):
        # TODO need to check if a model is available for the provided source
        # windowing and normalization
        target_data = target_data.copy()
        target_data["label"] = 0
        target_data["event"] = 0
        X_final, y_final, e_final = self.create_windowed_data(target_data, self.seq_length, self.normalize)
        X_final = X_final.reshape(X_final.shape[0], -1)


        target_data_t = self.scalers[source].transform(X_final.astype('float32'))
        surv = self.model_per_source[source].predict_surv_df(target_data_t)


        predictions = surv.values.T
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result

    def predict_one(self, new_sample, source: str, is_event: bool) -> float:
        new_sample=self.normalize_columns(new_sample).reshape(1,-1) if self.normalize else new_sample.reshape(1,-1)

        target_data_t = self.scalers[source].transform(new_sample.astype('float32'))
        surv = self.model_per_source[source].predict_surv_df(target_data_t)
        predictions = surv.values.T
        n, T = predictions.shape

        # Repeat the time array for every curve → shape (n, T)
        times_tiled = np.tile(self.avail_times_per_source[source], (n, 1))

        # Stack into (n, 2, T)
        result = np.stack([predictions, times_tiled], axis=1)
        return result[0]

    def get_params(self) -> dict:
        params = {
            "num_nodes" : self.num_nodes,
            "batch_norm" : self.batch_norm,
            "dropout" : self.dropout,
            "batch_size" : self.batch_size,
            "learning_rate" : self.learning_rate,
            "epochs" : self.epochs,
            "seq_length" : self.seq_length,
        }
        return params

    def get_library(self) -> str:
        # TODO we could also try to return a reference to the corresponding subpackage if it works
        return 'no_save'

    def __str__(self) -> str:
        """
            Returns a string representation of the corresponding method
        """
        return "DeepHIT"

    def get_all_models(self):
        pass

