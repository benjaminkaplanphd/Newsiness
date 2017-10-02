import pandas as pd
import numpy as np


def get_vector(con, word):
	sql = "SELECT vector FROM w2v WHERE word = '%s'" % word
	result =  pd.read_sql_query(sql, con)['vector']
	if len(result)==0:
		return None
	res = result[0]
	res[:] = [float(a) for a in res]
	return np.array(res)
