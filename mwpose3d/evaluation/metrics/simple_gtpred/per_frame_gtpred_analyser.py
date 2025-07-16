import pandas as pd
import orjson as json
import numpy as np

from mwpose3d.registry import METRICS
from .simple_gtpred_analyzer import SimpleGTPredAnalyzer


@METRICS.register_module()
class PerFrameGTPredAnalyzer(SimpleGTPredAnalyzer):
    """
    Implement the behaviour of the SimpleGTPredAnalyzer, but transfrom the data to be workable
    er-frame. Moreover, we store the data in a faster file format than json.

    Right now we process the json after it's been created, however, it would be faster to create
    the dataframe as `process_sample` is being called.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fast_out_file = self.out_file.replace(".json", ".parquet")
        self.make_out_file(self.fast_out_file)

    def _load_report(self, load_json: bool = False):
        """
        Normally, we would load the json, parse it and write back the parquet file. However,
        we now have access to the original data, so we should prefer using that. Loading the json
        should not be done.
        """
        if not load_json:
            return self.report

        with open(self.out_file, "r", encoding="UTF-8") as f:
            json_data: list[dict] = json.loads(f.read())
        return json_data

    def _load_data(self, load_json: bool) -> pd.DataFrame:
        """
        Load the generated report and transform it into a faster and more usable format.
        Currently, we drop the per-joint errors and agr

        This code is copied from mmwave-generalization.
        """

        report_data = self._load_report(load_json)
        df_temp = pd.json_normalize(report_data)

        # Ensure correct column naming
        df_temp.index.name = "frame_nr"
        df_temp.index = df_temp.index.astype(np.int32)
        df_temp = df_temp.reset_index()

        # Create a column of the columns names, such that we can split these later
        df_temp = df_temp.melt(
            id_vars="frame_nr", var_name="column_names", value_name="value"
        )

        # Split 'column_names' into joint number and value name
        # (e.g., '0.gt_joint' → '0', 'gt_joint')
        df_temp[["joint", "value_name"]] = df_temp["column_names"].str.split(
            ".", expand=True
        )
        df_temp["joint"] = df_temp["joint"].astype(np.int8)

        # Pivot so that each value_name becomes a column again
        df_temp = df_temp.pivot_table(
            index=["frame_nr", "joint"],
            columns="value_name",
            values="value",
            aggfunc="first",
            sort=False,
        ).reset_index()

        # Remove index name 'value_name'
        df_temp.columns.names = [None]

        # We need to re-do this because the values where in the same column as the gt_joint values,
        # which are _object_ (list)
        df_temp.abs_error = df_temp.abs_error.astype(np.float32)
        df_temp.square_error = df_temp.square_error.astype(np.float32)

        # Calculate the Mean Average Error
        df_temp["MAE"] = df_temp.groupby("frame_nr", sort=False)["abs_error"].transform(
            "mean"
        )

        # Re-order columns, set the index
        df_temp = df_temp[
            [
                "frame_nr",
                "MAE",
                "joint",
                "gt_joint",
                "pred_joint",
                "abs_error",
                "square_error",
            ]
        ]
        df_temp = df_temp.set_index(["frame_nr", "MAE", "joint"])
        return df_temp

    def evaluate(self, **kwargs):
        """
        The process_sample function takes the processed data and ground truth and stores it as a
        report in `self.report`, together with the ground truth in `self.gt` and prediction in
        `self.pred_data`.

        We take these results, transform them and save them as a parquet file.
        """
        super().evaluate(**kwargs)

        # Instead of loading the dumped json, we could also directly read `self.report`, however,
        # does this produce the same result?
        print(f"Finished evaluation, loading data and writing to {self.fast_out_file}")
        dataframe = self._load_data(load_json=False)
        dataframe.to_parquet(self.fast_out_file)
