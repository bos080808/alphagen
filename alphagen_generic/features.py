from alphagen.data.expression import Feature, Ref
from alphagen_qlib.gold_data import FeatureType

# Basic price features
high = High = HIGH = Feature(FeatureType.HIGH)
low = Low = LOW = Feature(FeatureType.LOW)
volume = Volume = VOLUME = Feature(FeatureType.VOLUME)
open_ = Open = OPEN = Feature(FeatureType.OPEN)
close = Close = CLOSE = Feature(FeatureType.CLOSE)
vwap = Vwap = VWAP = Feature(FeatureType.VWAP)

# Gold specific features
interest_rate = Interest_Rate = INTEREST_RATE = Feature(FeatureType.INTEREST_RATE)
usd_index = USD_Index = USD_INDEX = Feature(FeatureType.USD_INDEX)
inflation = Inflation = INFLATION = Feature(FeatureType.INFLATION)

# Target calculation (for example, 20-day return)
target = Ref(close, -20) / close - 1
