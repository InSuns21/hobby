#coding:utf-8
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def road_json(path = 'JSON/data.json'):
	'''
	pathからJSON形式のデータを読み取って返します。
	'''
	f = open(path, 'r')
	jsonData = json.load(f)
	print jsonData
	f.close()
	return jsonData


def road_csv_cls(csv_path = 'res/iris.csv' , n_cls = 5 ,rand_state = 20):
	'''
	CSVファイルをロードして指定された属性でクラスタリングして、クラスタの番号を追加します。
	'''
	bol_row = road_json('JSON/data.json')
	names = road_json('JSON/row.json')
	iris = pd.read_csv(csv_path, header=None, names=names)
	features = iris.loc[:,bol_row].values
	# クラスタリング
	kmeans_model = KMeans(n_clusters = n_cls , random_state = rand_state).fit(np.array(features))
	labels = kmeans_model.labels_
	df = pd.DataFrame(labels)
	iris["cls"] = df
	print iris
	iris.to_csv('res/outout.csv', index=False)
	
road_csv_cls()
