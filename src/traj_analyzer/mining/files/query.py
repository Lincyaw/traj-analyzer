import sys

import duckdb
import pandas as pd

pd.set_option("display.width", 240, "display.max_colwidth", 320, "display.max_columns", 40)
print(duckdb.sql(sys.argv[1]).limit(200).df().to_string(index=False))
