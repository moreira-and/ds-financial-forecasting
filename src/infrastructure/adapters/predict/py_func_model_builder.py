import pickle
import mlflow.pyfunc
import pandas as pd
import numpy as np


class PyFuncModelTemplate:
    def __call__(self, preprocessor, model, postprocessor):
        class _Model(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                self.preprocessor = preprocessor
                self.model = model
                self.postprocessor = postprocessor

            def predict(self, context, model_input: pd.DataFrame):
                X_proc = self.preprocessor.transform(model_input)
                y_pred_raw = self.model.predict(X_proc)
                return self.postprocessor.transform(y_pred_raw)

        return _Model()

