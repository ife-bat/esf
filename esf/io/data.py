import datetime
import logging
import pathlib
import pickle
import uuid
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from esf.settings.parameters import (
    CYCLE_UNIT,
    DOD_UNIT,
    Q_,
    RATE_UNIT,
    SOC_UNIT,
    SOH_UNIT,
    TEMPERATURE_UNIT,
    TIME_UNIT,
    Columns,
    DataType,
    ESFParams,
    Regime,
)

# Note - currently, various methods are using hard-coded strings when accessing
# columns. Meaning that these can not be changed without rewriting many methods.
# But you can add more columns if needed without getting into trouble.
SAMPLE_DATA_COLUMNS = ["uid", "t", "N", "T", "SoH", "SoC", "DoD", "L", "y_err"]


MINIMUM_COLUMNS = {
    DataType.CALENDAR_VS_TEMPERATURE: ["uid", "t", "SoH", "SoC", "T", "L", "y_err"],
    DataType.CYCLE_VS_TEMPERATURE: ["uid", "t", "SoH", "SoC", "T", "L", "y_err"],
    DataType.CYCLE_VS_DOD: ["uid", "t", "SoH", "DoD", "L", "y_err"],
    DataType.CALENDAR_VS_SOC: ["uid", "t", "SoH", "SoC", "L", "y_err"],
}

DATA_TYPES_TO_CHECK = MINIMUM_COLUMNS.keys()


SAMPLE_DATA_REQUIRED_INPUT_COLUMNS = [
    "t",
    "N",
    "T",
    "SoH",
    "SoC",
]
SAMPLE_DATA_NORMALIZABLE_COLUMNS = ["DoD"]

OP_DATA_COLUMNS = ["uid", "t", "N", "T", "SoH", "SoC", "DoD"]
OP_DATA_NORMALIZABLE_COLUMNS = ["DoD"]

# The internal drive-cycle columns that OperationalData produces for the
# degradation predictor (see esf.simulations.degradation).
DRIVE_CYCLE_COLUMNS = ["time", "soc", "temperature", "c-rate"]

# Maps the internal field names to the column names of a raw operational
# (field-data) frame. Either "time" (numeric, with a unit) or "datetime"
# (timestamps) supplies the time axis; "soh" is optional (measured health for
# validating a prediction). Override per call via OperationalData's column_map.
OP_INPUT_DEFAULT_COLUMN_MAP = {
    "time": "t",
    "datetime": "DateTime",
    "soc": "SoC",
    "c_rate": "c-rate",
    "temperature": "T",
    "soh": "SoH",
}

DEFAULT_Y_ERR = 1.0

DELIMITER = ","
USE_PD_INDEX = False

logger = logging.getLogger(__name__)


def percentage_to_float(x):
    """
    Converts a percentage string to a float.
    Args:
        x (str): The percentage string to be converted.
    Returns:
        float: The converted float value.
    Example:
        >>> percentage_to_float("50%")
        50.0
    """

    return float(x.replace("%", ""))


def extract_values_from_column_name(
    name,
    splitter="_",
    r_stripper=None,
    l_stripper=None,
    index=-1,
    convert_to=None,
    add_norm=0.0,
    mult_norm=1.0,
    verbose=False,
):

    selected_part = name.split(splitter)[index]
    txt = f"Name: {name} -> {selected_part}"
    if r_stripper is not None:
        selected_part = selected_part.rstrip(r_stripper)
        txt += f" -> {selected_part}"
    if l_stripper is not None:
        selected_part = selected_part.lstrip(l_stripper)
        txt += f" -> {selected_part}"
    if convert_to is not None:
        try:
            selected_part = convert_to(selected_part)
            txt += f" -> {selected_part}"
            selected_part = selected_part * mult_norm + add_norm
            txt += f" -> {selected_part}"
        except ValueError:
            txt += f" -> Could not convert to {convert_to}"
    if verbose:
        print(txt)
    return selected_part


@dataclass
class MetaDataItem:
    """Contains metadata for a data-set.

    Attributes:
        cell_id (str): A unique identifier for the data-set
        data_type (DataType): The type of data-set
        comment (str): A comment for the data-set
        date (str): The date when the data was added
        time_unit (str): The unit of time
        cycle_unit (str): The unit of cycles
        temperature_unit (str): The unit of temperature
        cols (list): The original columns in the data-set
        is_valid (bool): If the data is valid or not

    """

    cell_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data_type: DataType = DataType.CALENDAR_VS_TEMPERATURE
    comment: str = None
    date: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    time_unit: str = TIME_UNIT
    cycle_unit: str = CYCLE_UNIT
    temperature_unit: str = TEMPERATURE_UNIT
    rate_unit: str = RATE_UNIT
    soc_unit: str = SOC_UNIT
    dod_unit: str = DOD_UNIT
    soh_unit: str = SOH_UNIT
    cols: list = field(default_factory=list)
    is_valid: bool = True

    def _as_frame(self):
        return pd.DataFrame(asdict(self))

    def _as_dict(self):
        return asdict(self)

    def keys(self):
        return asdict(self).keys()

    def unit_of(self, key):
        if key == Columns.TIME:
            return self.time_unit
        elif key == Columns.TEMPERATURE:
            return self.temperature_unit
        elif key == Columns.N:
            return self.cycle_unit
        elif key == Columns.SOC:
            return self.soc_unit
        elif key == Columns.DOD:
            return self.dod_unit
        elif key == Columns.SOH:
            return self.soh_unit
        else:
            return None


@dataclass
class MetaData:
    """
    Collection of data-sets metadata.

    Attributes:
        data (dict): A dictionary with MetaDataItem objects

    Methods:
        add_metadata: Add metadata to the data
        remove_metadata: Remove metadata from the data
        update_metadata: Update metadata in the data
        get_metadata: Get metadata from the data
        get_valid: Get the keys (uid) for valid data-sets
        as_frame: Return the metadata as a pandas DataFrame
    """

    data: dict = field(default_factory=dict)

    def __str__(self):
        txt = "<MetaData object>\n"
        for key, value in self.data.items():
            txt += f"  {key}: {value}\n"
        txt += f"  Valid: {self.number_of_valid}/{len(self.data)} items\n"
        return txt

    def __len__(self):
        return len(self.data)

    def as_frame(self):
        return pd.DataFrame([v.__dict__ for v in self.data.values()])

    @staticmethod
    def _guess_data_type(cols):
        warnings.warn(
            "Guessing data type from columns is an experimental feature. Proceed with caution."
        )
        dtype = DataType.CALENDAR_VS_TEMPERATURE
        if "N" in cols and "DoD" in cols:
            return DataType.CYCLE_VS_DOD
        elif "N" in cols and "T" in cols:
            return DataType.CYCLE_VS_TEMPERATURE
        elif "SoC" in cols:
            return DataType.CALENDAR_VS_SOC
        return dtype

    def add_metadata(self, key=None, data_type=None, subset=None, **kwargs):
        if data_type is None:
            data_type = self._guess_data_type(kwargs.get("cols", []))
        if key is None:
            key = str(uuid.uuid4())
        if subset is not None:
            key = f"{subset}_{key}"
        m = MetaDataItem(cell_id=key, data_type=data_type, **kwargs)
        self.data[key] = m
        return key

    def has_type(self, data_type):
        return any(item.data_type == data_type for item in self.data.values())

    def types(self):
        t = set()
        for key in self.data:
            t.add(self.data[key].data_type)
        return list(t)

    def remove_metadata(self, key):
        self.data.pop(key, None)

    def update_metadata(self, key, **kwargs):
        m = self.data[key]
        for key, value in kwargs.items():
            setattr(m, key, value)

    def get_metadata(self, key):
        return self.data[key]

    @property
    def number_of_valid(self):
        if self.data is None:
            return 0

        i = 0
        for key in self.data:
            if self.data[key].is_valid:
                i += 1
        return i

    def get_valid(self, data_type=DataType.CALENDAR_VS_TEMPERATURE):
        if self.data is None:
            return []

        v = []
        for key in self.data:
            if self.data[key].is_valid and self.data[key].data_type == data_type:
                v.append(key)
        return v

    def append(self, metadata):
        self.data.update(metadata.data)


class Data:
    def __init__(self, data=None, metadata=None):
        self.columns = SAMPLE_DATA_COLUMNS
        self.required_input_columns = SAMPLE_DATA_REQUIRED_INPUT_COLUMNS
        self.normalizable_columns = SAMPLE_DATA_NORMALIZABLE_COLUMNS
        # `data if None else` (not `data or`): a passed DataFrame is truth-value
        # ambiguous, so `data or default` would raise
        self.data = data if data is not None else pd.DataFrame(columns=self.columns)
        self.chemistry = None
        self.metadata = metadata or MetaData()

    def __str__(self):
        txt = "<Data object>\n"
        txt += f"Data: {type(self.data)}\n"
        txt += f"Chemistry: {self.chemistry}\n"
        txt += f"  rows: {self.data.shape[0]}\n"
        txt += f"  cols: {self.data.columns}\n"
        txt += f"Meta: {self.metadata}\n"
        return txt

    def __iter__(self):
        """Yield one {"uid", "data", "metadata"} dict per data-set (uid).

        Implemented as a generator so that iterations are independent:
        nested or concurrent loops over the same object each get their own
        state (the previous implementation kept the position on the instance,
        which silently broke nested iteration).
        """
        if self.data.empty:
            return
        for uid in self.data["uid"].unique().tolist():
            yield {
                "uid": uid,
                "data": self.data.loc[self.data["uid"] == uid].copy(),
                "metadata": self.metadata.get_metadata(uid),
            }

    def add_data_from_csv(self, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for this child class")

    def add_data(
        self,
        data: pd.DataFrame,
        data_type=None,
        comment=None,
        time_unit=TIME_UNIT,
        cycle_unit=CYCLE_UNIT,
        temperature_unit=TEMPERATURE_UNIT,
        **kwargs,
    ):
        """Add data to the data attribute.

        Args:
            data (pd.DataFrame): The data to be added.
            data_type (DataType): The type of data-set
            comment (str): A comment for the data-set
            time_unit (str): The unit of time
            cycle_unit (str): The unit of cycles
            temperature_unit (str): The unit of temperature
        """

        # make sure the data is in a readable format:
        if not self._validate_data(data):
            raise ValueError("Data is not of correct format")

        # not allowed to inject cols and cell_id through this method to the metadata:
        _ = kwargs.pop("cols", None)
        _ = kwargs.pop("cell_id", None)
        cols = list(data.columns)
        # make sure the data has the correct columns etc.:
        data = self._transform_data(data)

        if "subset" not in cols:
            data["subset"] = "A"

        g = data.groupby("subset")

        for subset, d in g:
            d = d.drop(columns=["subset"])

            key = self.metadata.add_metadata(
                cols=cols,
                data_type=data_type,
                subset=subset,
                comment=comment,
                time_unit=time_unit,
                cycle_unit=cycle_unit,
                temperature_unit=temperature_unit,
                **kwargs,
            )

            # create a unique identifier for the data and the subsets
            d["uid"] = key

            if self.data.empty:
                self.data = d
            else:
                self.data = pd.concat([self.data, d], ignore_index=True)

    def normalize(self, cols=None):
        if cols is None:
            self.normalize_all()
        else:
            for col in cols:
                if col in self.normalizable_columns:
                    self.data[col] = self.data[col] / self.data[col].max()

    def _normalizable_and_available_cols(self):
        cols = []
        for c in self.data.columns:
            if c in self.normalizable_columns:
                cols.append(c)
        return cols

    def _transform_data(self, data):
        print("Transforming data is not implemented for this child class")
        return data

    def _validate_data(self, data):
        # This is "hard" validation
        if not isinstance(data, pd.DataFrame):
            return False

        # This is "soft" validation (warnings)
        for col in self.required_input_columns:
            if col not in data.columns:
                logger.debug(f"Missing column '{col}' in data")
        return True

    def _convert_time_data(
        self, data: pd.DataFrame, time_unit=TIME_UNIT
    ) -> pd.DataFrame:
        new_data = []
        for uid in data["uid"].unique():
            md = self.metadata.get_metadata(uid)
            g = data.loc[data["uid"] == uid, :].copy()
            if "t" in g.columns:
                try:
                    v = g["t"].values
                    in_original_units = Q_(v, md.time_unit)
                    in_target_units = in_original_units.to(time_unit)
                    g.loc[:, "t"] = in_target_units.magnitude
                except Exception as e:
                    warnings.warn(f"Could not convert time data for uid={uid}: {e}")

            new_data.append(g)
        data = pd.concat(new_data, axis=0, ignore_index=True)
        return data

    def _convert_unit(self, value: float | pd.Series, unit_from: str, unit_to: str):
        # Check if time is a pandas Series and handle accordingly
        if isinstance(value, pd.Series):
            # Apply the conversion to each element in the Series
            return value.apply(lambda x: Q_(x, unit_from).to(unit_to).magnitude)
        else:
            # Handle scalar
            return Q_(value, unit_from).to(unit_to).magnitude

    def _convert_temperature_data(
        self, data: pd.DataFrame, temperature_unit=TEMPERATURE_UNIT
    ) -> pd.DataFrame:
        new_data = []
        for uid, g in data.groupby("uid"):
            md = self.metadata.get_metadata(uid)
            if "T" in g.columns:
                try:
                    v = g["T"].values
                    in_original_units = Q_(v, md.temperature_unit)
                    in_target_units = in_original_units.to(temperature_unit)
                    g.loc[:, "T"] = in_target_units.magnitude
                except Exception as e:
                    warnings.warn(
                        f"Could not convert temperature data for uid={uid}: {e}"
                    )
            new_data.append(g)
        data = pd.concat(new_data, axis=0, ignore_index=True)
        return data

    def normalize_cols(self, cols):
        if not cols:
            warnings.warn("No columns to normalize")
            return
        g = self.data.groupby("uid")[cols]
        if not len(g):
            warnings.warn("No data to normalize")
        try:
            d = g.transform(lambda x: x / x.max())
            self.data[cols] = d[cols]
        except ValueError:
            logger.error(f"normalization failed for columns {cols}")
            raise

    def normalize_all(self):
        cols = self._normalizable_and_available_cols()
        self.normalize_cols(cols)

    def _load_data(self, data_path: pathlib.Path, loader_func=None, **kwargs):
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        if loader_func is None:
            loader_func = pd.read_csv
        data = loader_func(data_path, **kwargs)
        return data

    def _to_frame(self):
        data = self.data.copy()
        frames = []
        for uid in data["uid"].unique():
            d = data.loc[data["uid"] == uid, :].copy()
            md = self.metadata.get_metadata(uid)
            md = md._as_dict()
            for k in md:
                if k == "cols":
                    d[k] = str(md[k])
                else:
                    d[k] = md[k]
            frames.append(d)
        return pd.concat(frames)

    def _save_data_to_csv(self, save_path: pathlib.Path, **kwargs):
        sep = kwargs.pop("sep", DELIMITER)
        index = kwargs.pop("index", USE_PD_INDEX)
        data = self._to_frame()
        data.to_csv(save_path, sep=sep, index=index, **kwargs)
        return save_path

    def _save_data_to_pickle(self, save_path: pathlib.Path, **kwargs):
        db = dict(
            data=self.data,
            metadata=self.metadata,
        )
        with open(save_path, "wb") as db_file:
            pickle.dump(db, db_file, **kwargs)

        return save_path

    def _load_data_from_pickle(self, data_path: pathlib.Path, **kwargs):
        with open(data_path, "rb") as db_file:
            db = pickle.load(db_file)
        self.data = db["data"]
        self.metadata = db["metadata"]

    def _load_data_from_csv(self, data_path: pathlib.Path, **kwargs):

        dev_mode = kwargs.pop("dev_mode", False)
        include_meta = kwargs.pop("include_meta", True)

        import ast

        def convert_to_list(ini_list: str) -> list | str:
            try:
                res = ast.literal_eval(ini_list)
            except ValueError as e:
                print(e)
                return ini_list
            return res

        sep = kwargs.pop("sep", DELIMITER)
        index_col = kwargs.pop("index_col", USE_PD_INDEX)
        data = pd.read_csv(data_path, sep=sep, index_col=index_col, **kwargs)

        if not include_meta:
            self.data = data
            return

        dummy_meta = MetaDataItem()
        meta_keys = dummy_meta.keys()

        for key, d in data.groupby("uid"):
            try:
                m = d.loc[:, d.columns.isin(meta_keys)].iloc[0, :].to_dict()
                if dev_mode and "cols" in m:
                    m["cols"] = convert_to_list(m["cols"])
                if "data_type" in m:
                    m["data_type"] = DataType(m["data_type"])

                _ = m.pop("cols", None)
                _ = m.pop("cell_id", None)

                d = d.loc[:, ~d.columns.isin(meta_keys)].copy()
                self.add_data(d, **m)
                print(f"...added data for uid={key}, size={d.shape}")
            except Exception as e:
                print(f"Could not load data for uid={key}: {e}")

    @staticmethod
    def _generate_filename(extension: str = "pkl"):
        datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        filename = f"{datetime_str}-esf-data.{extension}"
        return filename

    def save(self, save_path: pathlib.Path = None, extension: str = "pkl", **kwargs):
        if save_path is None:
            save_path = pathlib.Path.cwd() / self._generate_filename(
                extension=extension
            )
        else:
            save_path = pathlib.Path(save_path)
            if save_path.is_dir():
                save_path = save_path / self._generate_filename(extension=extension)
        if save_path.suffix == ".csv":
            return self._save_data_to_csv(save_path, **kwargs)
        if save_path.suffix == ".pkl":
            return self._save_data_to_pickle(save_path, **kwargs)
        else:
            raise ValueError("Unknown file format")


def _load_data_items_from_pickle(data_path: pathlib.Path, **kwargs) -> dict:
    with open(data_path, "rb") as db_file:
        db = pickle.load(db_file)
    return db


class SampleData(Data):
    """
    This class is a placeholder for the data that is used for parametrizing (fitting) the model.

    Attributes:
        data (pd.DataFrame): The data that is used in the model
        metadata (dict): Metadata for the data

    Methods:
        add_data: Add data to the data attribute
        normalize: Normalize the data
        calendar_life_vs_temperature: Get the calendar life vs temperature
        calendar_life_vs_soc: Get the calendar life vs SoC
        cycle_life_vs_dod: Get the cycle life vs DoD
        export_data: Save the data to a csv file
        import_data: Load the data from a csv file
        save: Save the data to a pickle file
        load: Load the data from a pickle file


    The proposed degradation model requires at least two sets
    of data to obtain the model parameters: the battery calendar
    aging test data with specified SoC and temperature, and the
    battery cycle life data with specified DoD and temperature.
    The cycle life data are normalized to the reference condition.

    Table of model coefficients tuned using LMO battery degradation test data from the same manufacturer:

    Nonlinear Degradation Model
    calendar: L = 1 − alpha_sei·exp(-t·beta_sei·ftd) − (1 − alpha_sei)·exp(−t·ftd)
    cycling: L = 1 − alpha_sei·exp(-N·beta_sei·ft1) − (1 − alpha_sei)·exp(−N·ft1)
    ------------------------------------------------------------------------------
    alpha_sei = 5.75e-2
    beta_sei = 121.0

    DoD Stress Model Sδ(δ) = (k_d1·δ^k_d2 + k_d3)^−1:
    -------------------------------------------------
    Sδ(δA) = [fd,1[δ=δA,σ=σA,T=T_A,t_p=t_p,A] / (S_T(T_A)·Sσ(σA))] − St(t_p,A)

    k_d1 = 1.0e4
    k_d2 = -5.01e-1
    k_d3 = -2.3e5

    SoC Stress Model S_alpha = exp(k_alpha · (alpha - alpha_ref)):
    --------------------------------------------------------------
    k_alpha = 1.04
    alpha_ref = 0.5

    Temperature Stress Model S_T(T) = exp(k_T·(T−T_ref)·(T_ref/T):
    --------------------------------------------------------------

    S_T(T_A) = fd,t[T=T_A] / fd,t[T=T_ref]

    fd,t[T=T_A] = degradation rate at T=T_A

    k_T = 6.93e-2
    T_ref = 25.0 C

    Calendar Aging Model S_t(t) = k_t·t:
    ------------------------------------
    k_t = 4.14e-10/s


    References:
        B.Xu et al. "Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment",
        IEEE Transactions on Smart Grid, vol.9, 2018, 1131-1140.

    """

    def __init__(self, data=None, metadata=None):
        super().__init__(data, metadata)
        self.columns = SAMPLE_DATA_COLUMNS
        self.normalizable_columns = SAMPLE_DATA_NORMALIZABLE_COLUMNS
        # `data if None else` (not `data or`): a passed DataFrame is truth-value
        # ambiguous, so `data or default` would raise
        self.data = data if data is not None else pd.DataFrame(columns=self.columns)
        self.chemistry = None
        self.metadata = metadata or MetaData()

    def __str__(self):
        txt = "<SampleData object>\n"
        txt += f"Data: {type(self.data)}\n"
        txt += f"Chemistry: {self.chemistry}\n"
        txt += f"  rows: {self.data.shape[0]}\n"
        txt += f"  cols: {self.data.columns}\n"
        txt += f"Meta: {self.metadata}\n"
        return txt

    def add_data_from_csv(
        self,
        data_path: pathlib.Path,
        data_type: DataType,
        column_names: dict,
        reader_kwargs: dict,
        z_val=None,
        **kwargs,
    ):
        data = self._load_data(data_path, pd.read_csv, **reader_kwargs)
        data = data.rename(columns=column_names)
        if z_val is not None:
            if data_type in [
                DataType.CALENDAR_VS_TEMPERATURE,
                DataType.CYCLE_VS_TEMPERATURE,
            ]:
                data["T"] = z_val
            elif data_type in [DataType.CALENDAR_VS_SOC]:
                data["SoC"] = z_val
            elif data_type in [DataType.CYCLE_VS_DOD]:
                data["DoD"] = z_val
        self.add_data(data, data_type=data_type, **kwargs)

    @classmethod
    def load_sample_data(cls, data_path: pathlib.Path, **kwargs):
        data = cls()
        if data_path.suffix == ".csv":
            print("Loading data from csv")
            print(" - this feature is considered less robust than loading from pickle")
            data._load_data_from_csv(data_path, **kwargs)
        elif data_path.suffix == ".pkl":
            data._load_data_from_pickle(data_path, **kwargs)
        else:
            raise ValueError("Unknown file format")
        return data

    def append_sample_data(self, data_path: pathlib.Path, **kwargs):
        if data_path.suffix != ".pkl":
            raise ValueError(f"Appending data from {data_path.suffix} is not allowed!")

        db = _load_data_items_from_pickle(data_path, **kwargs)
        data = db["data"]
        metadata = db["metadata"]

        self.metadata.append(metadata)
        new_data = pd.concat([self.data, data], ignore_index=True)
        new_data = new_data.drop_duplicates()
        self.data = new_data.reset_index(drop=True)

    def normalize_life(self):
        col = "L"
        if col not in self.data.columns:
            raise ValueError(
                f"Column {col} not found in data. Either add it or calculate (calculate_life_fraction) it first."
            )
        self.normalize_cols([col])

    def calculate_life_fraction(self):
        if "SoH" not in self.data.columns:
            warnings.warn("SoH column not found in data. Skipping.")
            return
        self.data["L"] = 1.0 - self.data["SoH"]

    def _transform_data(self, data):
        if "y_err" in data.columns:
            data["y_err"] = data["y_err"].astype(float)
        else:
            data["y_err"] = DEFAULT_Y_ERR

        for col in self.columns:
            if col not in data.columns:
                data[col] = np.nan
        return data

    def _convert_unit(self, value: float | pd.Series, unit_from: str, unit_to: str):
        # Check if time is a pandas Series and handle accordingly
        if isinstance(value, pd.Series):
            # Apply the conversion to each element in the Series
            return value.apply(lambda x: Q_(x, unit_from).to(unit_to).magnitude)
        else:
            # Handle scalar
            return Q_(value, unit_from).to(unit_to).magnitude

    def _filter_data(
        self,
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        filter_key=None,
        filter_val=None,
        filter_unit=None,
        delta=0.01,
    ):
        if filter_key is None:
            filter_key = data_type.z

        # Filter out the data that is invalid (wrong type or marked as invalid) or not of the correct data type
        valid = self.metadata.get_valid(data_type)
        d = self.data.loc[self.data["uid"].isin(valid)]

        if filter_val is not None:
            if filter_unit is not None:
                # Make sure the filter value is in the correct unit
                uids = d["uid"].unique()
                for uid in uids:
                    unit_to = self.metadata.get_metadata(uid).unit_of(filter_key)
                    filter_val = self._convert_unit(filter_val, filter_unit, unit_to)
            if isinstance(filter_val, str):
                query = f"{filter_key}{filter_val}"
                try:
                    d = d.query(query)
                except Exception as e:
                    print(f"FILTERING ERROR IN {query}")
                    print(e)
            else:
                d = d.loc[
                    (d[filter_key] >= filter_val - filter_val * delta)
                    & (d[filter_key] <= filter_val + filter_val * delta)
                ]
        return d

    @staticmethod
    def _select_columns(d: pd.DataFrame, cols: list) -> pd.DataFrame:
        try:
            d = d.loc[:, cols]
        except KeyError as e:
            print("KEY ERROR")
            print(f"Columns needed: {sorted(cols)}")
            print(f"Columns found: {sorted(d.columns)}")
            if "L" not in d.columns:
                print(" - have you forgotten to run data.calculate_life_fraction()?")
            raise e
        return d

    def _data_picker(
        self,
        data_type,
        filter_key,
        filter_value,
        time_unit,
        temperature_unit,
        cols,
        strict_cols=None,
        convert_time_unit=True,
        strict_mode=True,
        filter_unit=None,
    ):
        strict_cols = strict_cols or cols
        time_unit = time_unit or TIME_UNIT
        temperature_unit = temperature_unit or TEMPERATURE_UNIT

        d = self._filter_data(
            data_type,
            filter_key=filter_key,
            filter_val=filter_value,
            filter_unit=filter_unit,
        )
        if d.empty:
            warnings.warn(
                f"No data found for {data_type} with filter {filter_key}={filter_value}"
            )
            return d
        if convert_time_unit:
            d = self._convert_time_data(d, time_unit)
        d = self._convert_temperature_data(d, temperature_unit)
        d = self._select_columns(d, cols)

        bad_cols = {}
        for col in strict_cols:
            if col not in d.columns:
                warnings.warn(f"Column {col} not found in data ({data_type})")
                bad_cols[data_type] = "missing column"
            elif d[col].isna().values.any():
                warnings.warn(f"Column {col} has missing values ({data_type})")
                bad_cols[data_type] = d
        if bad_cols and strict_mode:
            raise ValueError(
                f"Data is not in the correct format (strict mode): {bad_cols}"
            )
        return d

    def get(
        self,
        data_type: DataType,
        filter_value=None,
        time_unit=None,
        temperature_unit=None,
        strict_mode=True,
    ) -> pd.DataFrame:
        selectors = {
            (Regime.CALENDAR, Columns.TEMPERATURE): self.calendar_life_vs_temperature,
            (Regime.CALENDAR, Columns.SOC): self.calendar_life_vs_soc,
            (Regime.CYCLING, Columns.TEMPERATURE): self.cycle_life_vs_temperature,
            (Regime.CYCLING, Columns.DOD): self.cycle_life_vs_dod,
        }
        selector = selectors.get((data_type.regime, data_type.z))
        if selector is None:
            # previously some combinations silently returned None (and one
            # branch called a selector method that never existed)
            raise ValueError(
                f"Unsupported data type: {data_type} "
                f"(regime={data_type.regime}, z={data_type.z})"
            )
        return selector(filter_value, time_unit, temperature_unit, strict_mode)

    def calendar_life_vs_temperature(
        self,
        filter_value=None,
        time_unit=None,
        temperature_unit=None,
        strict_mode=True,
    ) -> pd.DataFrame:

        cols = ["uid", "t", "SoH", "SoC", "T", "L", "y_err"]
        strict_cols = ["uid", "t", "SoH", "SoC", "T", "L", "y_err"]
        data_type = DataType.CALENDAR_VS_TEMPERATURE
        filter_key = "T"
        d = self._data_picker(
            data_type,
            filter_key,
            filter_value,
            time_unit,
            temperature_unit,
            cols,
            strict_cols=strict_cols,
            convert_time_unit=True,
            strict_mode=strict_mode,
            filter_unit=temperature_unit,
        )
        return d

    def calendar_life_vs_soc(
        self,
        filter_value=None,
        time_unit=None,
        temperature_unit=None,
        strict_mode=True,
    ) -> pd.DataFrame:

        cols = ["uid", "t", "SoH", "SoC", "T", "L", "y_err"]
        data_type = DataType.CALENDAR_VS_SOC
        filter_key = "SoC"
        d = self._data_picker(
            data_type,
            filter_key,
            filter_value,
            time_unit,
            temperature_unit,
            cols,
            convert_time_unit=True,
            strict_mode=strict_mode,
        )
        return d

    def cycle_life_vs_temperature(
        self,
        filter_value=None,
        time_unit=None,
        temperature_unit=None,
        strict_mode=True,
    ) -> pd.DataFrame:

        cols = ["uid", "N", "SoH", "SoC", "T", "L", "y_err"]
        strict_cols = ["uid", "N", "SoH", "T", "L", "y_err"]
        data_type = DataType.CYCLE_VS_TEMPERATURE
        filter_key = "T"
        d = self._data_picker(
            data_type,
            filter_key,
            filter_value,
            time_unit,
            temperature_unit,
            cols,
            strict_cols=strict_cols,
            convert_time_unit=True,
            strict_mode=strict_mode,
        )
        return d

    def cycle_life_vs_dod(
        self,
        filter_value=None,
        time_unit=None,
        temperature_unit=None,
        strict_mode=True,
    ) -> pd.DataFrame:

        cols = ["uid", "N", "t", "SoH", "DoD", "L", "T", "y_err"]
        data_type = DataType.CYCLE_VS_DOD
        filter_key = "DoD"
        d = self._data_picker(
            data_type,
            filter_key,
            filter_value,
            time_unit,
            temperature_unit,
            cols,
            convert_time_unit=True,
            strict_mode=strict_mode,
        )
        return d

    def check_data(
        self,
        filter_value=None,
        verbose=False,
    ) -> pd.DataFrame:
        """Check the data for consistency."""

        bad_cols = {}
        for data_type in DATA_TYPES_TO_CHECK:
            print()
            print("-" * 80)
            print(f"Checking {data_type}")
            print(f" - filter_key: {data_type.z}")
            print(f" - cols: {MINIMUM_COLUMNS[data_type]}")
            print("-" * 80)
            filter_key = data_type.z
            cols = MINIMUM_COLUMNS[data_type]

            d = self._filter_data(
                data_type, filter_key=filter_key, filter_val=filter_value
            )
            if d.empty:
                print(
                    f"No data found for {data_type} with filter {filter_key}={filter_value}"
                )
                bad_cols[data_type] = ("no data", None)
                print("-" * 80)
                continue
            try:
                d = d.loc[:, cols]
            except KeyError:
                print("KEY ERROR")
                print(f"Columns needed: {sorted(cols)}")
                print(f"Columns found: {sorted(d.columns)}")
            if "L" not in d.columns:
                print(" - have you forgotten to run data.calculate_life_fraction()?")

            for col in cols:
                if col not in d.columns:
                    print(f"WARNING: Column {col} not found in data")
                    bad_cols[data_type] = ("missing column", col)
                if d[col].isna().values.any():
                    print(f"WARNING: Column {col} has missing values")
                    bad_cols[data_type] = ("missing values", d)
            print("-" * 80)

        if verbose:
            if bad_cols:
                print("SUMMARY:")
                for k, v in bad_cols.items():
                    print(f"Data type: {k}, error: {v[0]}")
                    print("Details:")
                    print(v[1])
            else:
                print("No bad columns found")

        return bad_cols

    def plot_data(self, *args, **kwargs):
        """Overview plot of all data-sets. Delegates to esf.io.data_plotting."""
        from esf.io.data_plotting import plot_sample_data

        return plot_sample_data(self, *args, **kwargs)


class OperationalData(Data):
    """Operating time series for predicting degradation from real field data.

    Where ``SampleData`` holds *aging-test* data for **fitting**,
    ``OperationalData`` holds a single operating trace -- state of charge,
    C-rate, temperature and (optionally) measured state of health sampled over
    time -- for **prediction**. It converts a raw field-data frame to the
    package's internal units and hands off the drive-cycle frame that
    :func:`esf.drive_cycle_degradation_calculator` consumes.

    A typical raw field-data frame (IFE format)::

        DateTime,            SOC (%), Crate (C), Temp (DegC), SOH (%)
        2020.10.08 15:29:47, 100.15,  0.75,      26.11,       100
        2020.10.08 15:29:50, 100.08,  0.75,      26.11,       100
        ...

    Build one with :meth:`from_field_dataframe`, which maps those columns and
    converts to the internal convention (time in seconds, temperature in
    kelvin, SoC/SoH as fractions, C-rate unchanged -- see the units convention
    in the README). ``to_drive_cycle`` then yields the predictor's input, and
    ``predict`` runs it::

        op = OperationalData.from_field_dataframe(frame)
        result = op.predict(prms)             # loss over the trace
        measured = op.measured_soh()          # for validation, if SoH present

    The ingestion API is intentionally minimal; it may grow when the field-data
    apps are built (round-4 B3 / A5).
    """

    DRIVE_CYCLE_COLUMNS = DRIVE_CYCLE_COLUMNS
    REQUIRED_FIELDS = ("soc", "c_rate", "temperature")

    def __init__(self, data=None, metadata=None):
        super().__init__(data=data, metadata=metadata)
        self.columns = OP_DATA_COLUMNS
        self.normalizable_columns = OP_DATA_NORMALIZABLE_COLUMNS

    @classmethod
    def from_field_dataframe(
        cls,
        frame: pd.DataFrame,
        *,
        column_map: dict | None = None,
        soc_in_percent: bool = True,
        soh_in_percent: bool = True,
        temperature_unit: str = "degC",
        time_unit: str = "seconds",
        uid: str = "operational",
    ) -> "OperationalData":
        """Build an ``OperationalData`` from a raw field-data frame.

        Args:
            frame: the raw operational trace.
            column_map: overrides for :data:`OP_INPUT_DEFAULT_COLUMN_MAP`,
                mapping the internal field names (``"time"``/``"datetime"``,
                ``"soc"``, ``"c_rate"``, ``"temperature"``, ``"soh"``) to the
                frame's column names.
            soc_in_percent / soh_in_percent: divide those columns by 100 to get
                a 0-1 fraction (default: yes).
            temperature_unit: the unit of the temperature column (converted to
                kelvin). Common: ``"degC"`` or ``"K"``.
            time_unit: the unit of a numeric time column (converted to
                seconds). Ignored when a ``datetime`` column is used instead.
            uid: identifier stored on the trace.

        Time comes from a ``datetime`` column (parsed, elapsed seconds from the
        first timestamp) if present, otherwise from the numeric ``time`` column
        interpreted in ``time_unit``. ``soc``, ``c_rate`` and ``temperature``
        are required; ``soh`` is optional (measured health, for validation).
        """
        cmap = {**OP_INPUT_DEFAULT_COLUMN_MAP, **(column_map or {})}

        missing = [f for f in cls.REQUIRED_FIELDS if cmap[f] not in frame.columns]
        if missing:
            raise ValueError(
                "operational data is missing required column(s) "
                f"{[cmap[f] for f in missing]} (fields {missing}); "
                f"columns present: {list(frame.columns)}"
            )

        out = pd.DataFrame(index=range(len(frame)))
        out["time"] = cls._field_time(frame, cmap, time_unit)
        out["soc"] = frame[cmap["soc"]].to_numpy(dtype=float) / (
            100.0 if soc_in_percent else 1.0
        )
        out["c-rate"] = frame[cmap["c_rate"]].to_numpy(dtype=float)
        out["temperature"] = (
            Q_(frame[cmap["temperature"]].to_numpy(dtype=float), temperature_unit)
            .to(TEMPERATURE_UNIT)
            .magnitude
        )
        if cmap["soh"] in frame.columns:
            out["SoH"] = frame[cmap["soh"]].to_numpy(dtype=float) / (
                100.0 if soh_in_percent else 1.0
            )
        out["uid"] = uid

        return cls(data=out)

    @staticmethod
    def _field_time(frame, cmap, time_unit) -> np.ndarray:
        """Elapsed seconds from a datetime column, or a converted time column."""
        if cmap["datetime"] in frame.columns:
            timestamps = pd.to_datetime(frame[cmap["datetime"]])
            return (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
        if cmap["time"] in frame.columns:
            values = frame[cmap["time"]].to_numpy(dtype=float)
            return Q_(values, time_unit).to(TIME_UNIT).magnitude
        raise ValueError(
            "operational data needs a time axis: either a "
            f"'{cmap['datetime']}' (datetime) or '{cmap['time']}' (numeric) "
            "column"
        )

    def to_drive_cycle(self) -> pd.DataFrame:
        """The drive-cycle frame the degradation predictor consumes.

        Columns ``time`` (s), ``soc`` (0-1), ``temperature`` (K), ``c-rate``
        (C); rows with missing values in any of those are dropped.
        """
        missing = [c for c in self.DRIVE_CYCLE_COLUMNS if c not in self.data.columns]
        if missing:
            raise ValueError(f"operational data has no {missing} column(s)")
        drive_cycle = self.data[self.DRIVE_CYCLE_COLUMNS].dropna()
        if drive_cycle.empty:
            raise ValueError("no complete drive-cycle rows in the operational data")
        return drive_cycle.reset_index(drop=True)

    def measured_soh(self) -> pd.DataFrame | None:
        """Measured ``time``/``SoH`` for validating a prediction, or None."""
        if "SoH" not in self.data.columns:
            return None
        return self.data[["time", "SoH"]].dropna().reset_index(drop=True)

    def predict(
        self,
        prms: ESFParams,
        cycle_numbers=None,
        **kwargs,
    ) -> pd.DataFrame:
        """Predict degradation over the trace with the given parameters.

        Delegates to :func:`esf.drive_cycle_degradation_calculator`. With
        ``cycle_numbers=None`` the trace is treated as a single operating pass
        (``cycle_numbers=[1]``); pass an array to simulate repeated passes.
        """
        from esf.simulations.degradation import drive_cycle_degradation_calculator

        if cycle_numbers is None:
            cycle_numbers = np.array([1.0])
        return drive_cycle_degradation_calculator(
            self.to_drive_cycle(), prms, cycle_numbers=cycle_numbers, **kwargs
        )


def example_sample_data():
    from esf.settings.parameters import DATA_FOLDER

    datadir = DATA_FOLDER
    assert datadir.exists()

    cal_vs_soc_deg25_path = (
        datadir / "Ageing_Data_Org/calendar_degradation/calend_deg_at_25_deg.csv"
    )
    cal_vs_soc_deg25 = pd.read_csv(cal_vs_soc_deg25_path)

    column_names = list(cal_vs_soc_deg25.columns)

    n = [
        extract_values_from_column_name(
            name, "=", r_stripper="%", convert_to=float, verbose=False, mult_norm=0.01
        )
        for name in column_names
    ]

    n[0] = "t"
    cal_vs_soc_deg25.columns = n

    cal_vs_soc_deg25 = pd.melt(
        cal_vs_soc_deg25, id_vars=n[0], value_name="SoH", var_name="SoC"
    )
    cal_vs_soc_deg25["T"] = 25 + 273.15

    # ---- 2 ----
    cycle_soh_path = datadir / "Ageing_Data_Org/cycling_degradation/cyc_test_data.csv"
    sep = ","
    cols = {"N_Cycle": "N", "SOH": "SoH"}
    cycle_soh_path = pd.read_csv(cycle_soh_path, sep=sep)
    cycle_soh_path = cycle_soh_path.rename(columns=cols)
    cycle_soh_path["T"] = 25 + 273.15

    # ---- 3 ----
    cal_vs_temp_50_soc_path = (
        datadir / "Ageing_Data_Org/calendar_degradation/calend_deg_at_50_SoC.csv"
    )
    cal_vs_temp_50_soc = pd.read_csv(cal_vs_temp_50_soc_path, sep=sep)

    column_names = list(cal_vs_temp_50_soc.columns)

    n = [
        extract_values_from_column_name(
            name, "=", convert_to=int, verbose=False, add_norm=273.15
        )
        for name in column_names
    ]
    n[0] = "t"
    cal_vs_temp_50_soc.columns = n

    cal_vs_temp_50_soc = pd.melt(
        cal_vs_temp_50_soc,
        id_vars=n[0],
        value_name="SoH",
        var_name="T",
    )
    cal_vs_temp_50_soc["SoC"] = 0.5

    # ---- 4 ----
    cycle_vs_dod_soh80_lmn_path = (
        datadir / "Ageing_Data_Org/cycling_degradation/cycle_nb_at_80_SoH_LMO.csv"
    )
    cycle_vs_dod_soh80_lmn = pd.read_csv(cycle_vs_dod_soh80_lmn_path, sep=sep)
    cycle_vs_dod_soh80_lmn.columns = ["DoD", "N"]
    cycle_vs_dod_soh80_lmn["SoH"] = 0.8
    cycle_vs_dod_soh80_lmn["T"] = 25 + 273.15

    data = SampleData()
    data.add_data(
        cal_vs_soc_deg25,
        comment="This is a test calendar data vs SoC for SEI fitting",
        time_unit="years",
        temperature_unit="K",
        data_type=DataType.CALENDAR_VS_SOC,
    )
    data.add_data(
        cycle_soh_path,
        comment="This is a test cycling data vs temperature for SEI fitting",
        time_unit="days",
        temperature_unit="K",
        data_type=DataType.CYCLE_VS_TEMPERATURE,
    )
    data.add_data(
        cal_vs_temp_50_soc,
        comment="This is a test calendar data vs temperature for Temperature Stress Factor fitting",
        data_type=DataType.CALENDAR_VS_TEMPERATURE,
        time_unit="years",
        temperature_unit="K",
    )

    data.add_data(
        cycle_vs_dod_soh80_lmn,
        comment="This is a test cycling data vs DoD for LMO for DoD Stress Factor fitting",
        data_type=DataType.CYCLE_VS_DOD,
        cycle_unit="EFC",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    data.normalize_all()
    data.calculate_life_fraction()
    return data


def get_hardcoded_soc_deg_rates():
    """Create a DataFrame with hard-coded SoC degradation rates for testing.

    Returns:
        pd.DataFrame: DataFrame with columns 'SoC' and 'deg_rate'
    """
    import pandas as pd

    data = {
        "SoC": [0.5, 0.6, 0.8, 1.0],
        "deg_rate": [0.000036, 0.000039, 0.000049, 0.000060],
    }

    return pd.DataFrame(data)


def get_hardcoded_temperature_deg_rates():
    """Create a DataFrame with hard-coded temperature degradation rates for testing.

    Temperatures are in kelvin (15/25/35/45/55 degC), following the internal
    units convention.

    Returns:
        pd.DataFrame: DataFrame with columns 'T' and 'deg_rate'
    """
    import pandas as pd

    data = {
        "T": [288.15, 298.15, 308.15, 318.15, 328.15],
        "deg_rate": [0.000017, 0.000035, 0.000069, 0.000130, 0.000236],
    }

    return pd.DataFrame(data)
