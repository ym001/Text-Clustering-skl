#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  classifTextSKL.JeuxSmear4Smear.py
#  
#  Copyright 2017 yves <yves.mercadier@ac-montpellier.fr>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

from sklearn.pipeline import Pipeline

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans,AgglomerativeClustering,AffinityPropagation
from sklearn.mixture import GaussianMixture

from keras.datasets import imdb
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np

import unicodedata

def lecture_du_jeu_de_imdbkeras():
		max_words=20000
		x_train_imdb=[]
		y_train_imdb=[]
		(x_imdb, y_imdb), (x_test_imdb, y_test_imdb) = imdb.load_data(num_words=max_words,maxlen=300,seed=113)
		
		#reconstruction
		wordDict = {y:x for x,y in imdb.get_word_index().items()}
		for doc in x_imdb:
			sequence=""
			for index in doc:
				sequence+=" "+wordDict.get(index)
			x_train_imdb.append(sequence)
		for i in y_imdb:
			y_train_imdb.append(str(i))
					
		return x_train_imdb,y_train_imdb
		

def list_label(label_jeu):
	label=[]
	for lab in label_jeu:
		for l in lab:
			if l not in label:
				label.append(l)
	return label	
		
def main(args):

	#importation des données
	doc_train,label_train=lecture_du_jeu_de_imdbkeras()
	label=list_label(label_train)
	
	n_features=10000
	n_components=10
	verbose=1

	print("Extracting features from the training dataset using a sparse vectorizer")
	hasher = HashingVectorizer(n_features=n_features,
									stop_words='english', alternate_sign=False,
									norm=None, binary=False)
	vectorizer = make_pipeline(hasher, TfidfTransformer())
                                
	X = vectorizer.fit_transform(doc_train)

	print("n_samples: %d, n_features: %d" % X.shape)
	
	print("Performing dimensionality reduction using LSA")
	# Vectorizer results are normalized, which makes KMeans behave as
	# spherical k-means for better results. Since LSA/SVD results are
	# not normalized, we have to redo the normalization.
	svd = TruncatedSVD(n_components)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)

	X = lsa.fit_transform(X)

	explained_variance = svd.explained_variance_ratio_.sum()
	print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))
	clustering={}
	clustering['KMeans']= KMeans(n_clusters=len(label), init='k-means++', max_iter=100, n_init=1,verbose=verbose)
	clustering['GaussianMixture']= GaussianMixture(n_components=len(label), covariance_type='full')
	clustering['AgglomerativeClustering']=AgglomerativeClustering(n_clusters=len(label),linkage='ward')

	for clu in clustering:
		algo=clustering[clu]
		print("Clustering sparse data with %s" % clu)
		algo.fit(X)
		if clu=="GaussianMixture":
			labels_prediction=algo.predict(X)
		else:
			labels_prediction=algo.labels_


		print("Homogeneity: %0.3f" % metrics.homogeneity_score(label_train, labels_prediction))
		print("Completeness: %0.3f" % metrics.completeness_score(label_train, labels_prediction))
		print("V-measure: %0.3f" % metrics.v_measure_score(label_train, labels_prediction))
		print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(label_train, labels_prediction))
		print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels_prediction, sample_size=1000))

	print('Fin du script...')
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
