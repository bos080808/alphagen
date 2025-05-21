from typing import Optional
from torch import Tensor
from alphagen.data.alpha_calculator_base import TensorAlphaCalculator
from alphagen.data.expression import Expression
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.gold_data import GoldData


class QLibGoldDataCalculator(TensorAlphaCalculator):
    def __init__(self, data: GoldData, target: Optional[Expression] = None):
        super().__init__(normalize_by_day(target.evaluate(data)) if target is not None else None)
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))
    
    @property
    def n_days(self) -> int:
        return self.data.n_days 