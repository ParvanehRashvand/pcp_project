import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
import inspect

def accepts_param(func, param_name):
    sig = inspect.signature(func)
    params = sig.parameters
    # Prüfen ob Parameter explizit da ist ODER **kwargs existiert
    return (
        param_name in params
        or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    )

class SubjectPipeline(Pipeline):
    def __init__(self, steps, mask=None):
        super().__init__(steps)
        self.mask = mask

    def fit(self, X, y=None, **fit_params):
        if len(self.steps) == 0:
            self.is_fitted_ = True
            return self

        Xt, yt, final_fit_params = self._fit(X, y, **fit_params)

        if self._final_estimator not in (None, "passthrough"):
            self._final_estimator.fit(Xt, yt, **final_fit_params)

        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, "is_fitted_")

        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if transform in (None, "passthrough"):
                continue
            Xt = transform.transform(Xt, y=y, groups=self.mask)

        final = self._final_estimator
        if final not in (None, "passthrough") and hasattr(final, "transform"):
            Xt = final.transform(Xt, y=y, groups=self.mask)

        return Xt

    def predict(self, X, **predict_params):
        check_is_fitted(self, "is_fitted_")

        if len(self.steps) == 0 or self._final_estimator in (None, "passthrough"):
            raise AttributeError("The final step does not implement predict().")

        Xt = X
        for _, _, transform in self._iter(with_final=False):
            if transform in (None, "passthrough"):
                continue
            Xt = transform.transform(Xt)

        return self._final_estimator.predict(Xt, **predict_params)

    def score(self, X, y=None, sample_weight=None, groups=None):
        """
        Default:
            Accuracy between predict(X) and y.

        Optional extension:
            If groups is provided and predict(X) returns window/sample-level predictions
            while y is also window/sample-level, aggregate both to subject/group level
            by majority vote before scoring.
        """
        check_is_fitted(self, "is_fitted_")

        if y is None:
            raise ValueError("Score requires y for supervised evaluation.")

        y_pred = np.asarray(self.predict(X))
        y_true = np.asarray(y)

        if y_pred.shape[0] == y_true.shape[0]:
            return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

        if groups is None:
            raise ValueError(
                "Prediction and target are on different levels. "
                "Provide groups for aggregation or align predict(X) with y."
            )

        groups = np.asarray(groups)
        if not (len(groups) == len(y_pred) == len(y_true)):
            raise ValueError(
                "When using grouped scoring, groups, y, and predictions must have equal length."
            )

        y_pred_grouped = self._majority_vote_by_group(y_pred, groups)
        y_true_grouped = self._majority_vote_by_group(y_true, groups)

        return accuracy_score(y_true_grouped, y_pred_grouped)

    def _majority_vote_by_group(self, values, groups):
        values = np.asarray(values)
        groups = np.asarray(groups)

        unique_groups = []
        grouped_values = []

        for g in groups:
            if g not in unique_groups:
                unique_groups.append(g)

        for g in unique_groups:
            vals = values[groups == g]
            vals = vals[~self._is_nan_label_array(vals)]
            if len(vals) == 0:
                raise ValueError(f"Group {g!r} contains no valid labels after filtering.")
            grouped_values.append(self._majority_vote(vals))

        return np.asarray(grouped_values)

    @staticmethod
    def _majority_vote(values):
        values = np.asarray(values)
        uniq, counts = np.unique(values, return_counts=True)
        return uniq[np.argmax(counts)]

    @staticmethod
    def _is_nan_label_array(values):
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.floating):
            return np.isnan(values)
        return np.zeros(values.shape, dtype=bool)

    def _fit(self, X, y=None, **fit_params):
        """Fit the pipeline except the last step.

        Difference to the sci-kit learn Pipeline class method _fit is that the mask attribute is potentially
        manipulated by the transformer and therefore updated.
        """
        self.steps = list(self.steps)

        if len(self.steps) == 0:
            return X, y, {}

        self._validate_steps()

        Xt, yt = X, y

        fit_params_steps = {
            name: {}
            for name, step in self.steps
            if step not in (None, "passthrough")
        }

        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(
                    f"Fit parameters must use the step__param format, got {pname!r}."
                )
            step, param = pname.split("__", 1)
            if step not in fit_params_steps:
                raise ValueError(f"Unknown step name in fit parameters: {step!r}")
            fit_params_steps[step][param] = pval

        for _, name, transformer in self._iter(with_final=False):
            if transformer in (None, "passthrough"):
                continue

            if accepts_param(transformer.fit_transform, "groups"):
                result = transformer.fit_transform(Xt, yt, groups=self.mask)
            else:
                result = transformer.fit_transform(Xt, yt)

            if isinstance(result, tuple):
                Xt, self.mask = result[:2]
            else:
                Xt = result

        final_name = self.steps[-1][0]
        final_fit_params = fit_params_steps.get(final_name, {})

    return Xt, yt, final_fit_params