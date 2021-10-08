import tensorflow.keras.backend as K
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import csv
import seaborn as sns

def balanced_dataset(dt_y, size=None):
    dt = []
    if size is None:
        n_smp = dt_y.value_counts().min()
    else:
        n_smp = int(size / len(dt_y.value_counts().index))

    for label in dt_y.value_counts().index:
        samples = dt_y[dt_y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        dt += samples[indexes].tolist()
    return dt


def load_all_datasets():
    datasets_x = []
    datasets_y = []
    for idx_dt in range(16):
        datasets_x.append(np.load('x'+str(idx_dt)+'.npy'))
        datasets_y.append(np.load('y'+str(idx_dt)+'.npy'))
    return datasets_x, datasets_y


def get_average_vector(embeddings):
    return np.average(embeddings, axis=0)


def get_embeddings_bow(dt):
    cv = CountVectorizer(dt)
    count_vector = cv.fit_transform(dt)
    vocab = cv.get_feature_names()
    embeddings = embed(vocab)
    avg = get_average_vector(embeddings)
    return avg


def compute_dataset_proximity(name_datasets, name_target_datasets, embeddings, fg_metric, hit_n):
    if fg_metric == 0:
        similarities = cosine_similarity(embeddings)
    else:
        similarities = euclidean_distances(embeddings)

    df = pd.DataFrame(similarities,
                      index=name_datasets,
                      columns=name_datasets)

    most_similars = []
    if fg_metric == 0:
        for el in name_target_datasets:
            most_similars.append(df[el].nlargest(n=hit_n + 1))
    else:
        for el in name_target_datasets:
            most_similars.append(df[el].nsmallest(n=hit_n + 1))

    return df.filter(name_target_datasets) \
               .style \
               .background_gradient(axis=None).set_properties(subset=name_target_datasets,
                                                              **{'width': '70px'}), most_similars

def build_unified_similarity_ranking(rankings_list, name_datasets, name_target_datasets):
    dicts = []
    count = 0
    for i, j, k, l in zip(rankings_list[0], rankings_list[1], rankings_list[2], rankings_list[3]):
      data = {}
      for e in name_datasets:
        data[e] = 0
      #print('Dataset <'+ name_target_datasets[count] +'>:')
      for index, value in i.items():
        if index != name_target_datasets[count]:
            data[index] = data.get(index) + 1
      for index, value in j.items():
        if index != name_target_datasets[count]:
            data[index] = data.get(index) + 1
      for index, value in k.items():
        if index != name_target_datasets[count]:
            data[index] = data.get(index) + 1
      for index, value in l.items():
        if index != name_target_datasets[count]:
            data[index] = data.get(index) + 1
      dicts.append(data)
      count = count + 1
    return dicts

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
l1, l2 = load_all_datasets()

list_embeddings_avg = []
list_embeddings_vocab_avg = []

for i in l1:
    embeddings = get_average_vector(embed(i))
    list_embeddings_avg.append(embeddings)
    bow = get_embeddings_bow(i)
    list_embeddings_vocab_avg.append(bow)


name_datasets = ['2018 BR Election PT', 'Restaurants PT', '2016 US Election EN', 'GOP Debate EN',
                 '2012 US Election EN', 'TV PT', 'Music Festival EN', 'Urban Problems PT', 'Airlines EN',
                 'Movies 1 EN', 'Movies PT', 'Movies 2 EN', 'Apple EN', 'Airlines ES', '2018 CO Election ES',
                 'Sports ES']
name_target_datasets = ['2018 BR Election PT', '2016 US Election EN', '2012 US Election EN', '2018 CO Election ES']

a, most_similar_a = compute_dataset_proximity(name_datasets, name_target_datasets, list_embeddings_avg, 0, 5)
b, most_similar_b = compute_dataset_proximity(name_datasets, name_target_datasets, list_embeddings_avg, 1, 5)
cc, most_similar_c = compute_dataset_proximity(name_datasets, name_target_datasets, list_embeddings_vocab_avg, 0, 5)
d, most_similar_d = compute_dataset_proximity(name_datasets, name_target_datasets, list_embeddings_vocab_avg, 1, 5)

display(pd.DataFrame(most_similar_a))

display(pd.DataFrame(most_similar_b))

display(pd.DataFrame(most_similar_c))

display(pd.DataFrame(most_similar_d))

rankings_list = [most_similar_a, most_similar_b, most_similar_c, most_similar_d]

dicts = build_unified_similarity_ranking(rankings_list, name_datasets, name_target_datasets)

sim_df = pd.DataFrame(dicts).transpose()
sim_df.columns = name_target_datasets
cm = sns.color_palette("blend:white,green", as_cmap=True)
sim_df.style.background_gradient(cmap=cm, axis=None).set_properties(subset=name_target_datasets,
                                                                    **{'width': '60px'}, **{'text-align': 'center'})

