#coding:utf-8
import json
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


def road_json(path = 'json/analyzeTarget.json'):
	'''
	pathからJSON形式のデータを読み取って返します。
	'''
	f = open(path, 'r')
	jsonData = json.load(f)
	f.close()
	return jsonData


def clsify_data(learn_path = 'res/learn.csv' ,
	input_path = 'res/input.csv'):
	'''
	CSVファイルをロードして学習データを元にナイーブベイズで分類します。
	'''
	# 学習データ読み込み
	df_learn = pd.read_csv(learn_path, encoding="SHIFT-JIS")
	bln_row_learn = road_json('json/analyzeTarget_learn.json')
	features = df_learn.loc[:,bln_row_learn].values
	# ラベル読み込み
	labels = df_learn.ix[:,"cls"].values
	
	clf = GaussianNB() # ガウシアンカーネルによるナイーブベイズ分類器
	# データをもとに学習
	clf.fit(np.array(features), np.array(labels))

	#検証データ読み込み処理
	bln_row_test = road_json('json/analyzeTarget_test.json')
	df_test_ori = pd.read_csv(input_path, encoding="SHIFT-JIS")
	df_test  = df_test_ori.loc[:,bln_row_test]

	# 分類器で分類する
	results = clf.predict(df_test.values)

	# 出力処理
	df = pd.DataFrame(results)
	df_test_ori["cls"] = df
	df_test_ori.to_csv('res/output_svm.csv', encoding="SHIFT-JIS",index=False)


if __name__ == '__main__':
	dict = road_json('json/config_svm.json')
	clsify_data(dict['learn_path'],dict['test_path'])