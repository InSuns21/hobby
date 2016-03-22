#coding:utf-8
import json
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cluster import KMeans

def road_json(path = 'json/analyzeTarget.json'):
	'''
	pathからJSON形式のデータを読み取って返します。
	'''
	f = open(path, 'r')
	jsonData = json.load(f)
	f.close()
	return jsonData

def clsify_data(input_path = 'res/input.csv',n_cls = 5 ,rand_state = 20):
	'''
	CSVファイルをロードしてデータ数が多い場合はランダムサンプリングしたデータフレームでK-MEANメソッドを適用し、そのラベル結果を元に
	inputデータをラベリングします。
	'''

	#検証データ読み込み処理
	bln_row_test = road_json('json/analyzeTarget_test.json')
	df_test_ori = pd.read_csv(input_path, encoding="SHIFT-JIS")
	df_test  = df_test_ori.loc[:,bln_row_test]

	if(len(df_test) > 500):
		# K-MEAN + SVM
		#ランダムサンプリング
		sampler = np.random.permutation(len(df_test))
		df_test_sampling = df_test.take(sampler[:300])
		#ランダムサンプリング　に対して　K-MEAN
		kmeans_model = KMeans(n_clusters = n_cls , random_state = rand_state).fit(np.array(df_test_sampling.values))

		# 学習データ
		labels = kmeans_model.labels_
		features = df_test_sampling.values

		# ガウシアンカーネルによるナイーブベイズ分類器
		clf = GaussianNB()
		# データをもとに学習
		clf.fit(np.array(features), np.array(labels))

		# 分類器で分類する
		results = clf.predict(df_test.values)
		df = pd.DataFrame(results)
	else:
		#simple_K-mean
		kmeans_model = KMeans(n_clusters = n_cls , random_state = rand_state).fit(np.array(df_test.values))
		labels = kmeans_model.labels_
		df = pd.DataFrame(labels)

	# 出力処理
	df_test_ori["cls"] = df
	df_test_ori.to_csv('res/output_k-mean2.csv', encoding="SHIFT-JIS",index=False)

if __name__ == '__main__':
	dict = road_json('json/config_kmean.json')
	clsify_data(dict['test_path'],dict['n_clusters'],dict['random_state'])