import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from dataloader import gestureBlobDataset, surgeonStudyDataset, size_collate_fn
from neural_networks import encoderDecoder
import os
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from multipledispatch import dispatch
from typing import List, Tuple
from tqdm import tqdm
from barbar import Bar
from joblib import dump, load
import re

def store_embeddings_in_dict(blobs_folder_path: str, model: encoderDecoder) -> dict:
    blobs_folder = os.listdir(blobs_folder_path)
    blobs_folder = list(filter(lambda x: '.DS_Store' not in x, blobs_folder))
    blobs_folder.sort(key = lambda x: int(x.split('_')[1]))

    embeddings_list = []
    gestures_list = []
    user_list = []
    skill_dict = {'B': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 1, 'G': 0, 'H': 0, 'I': 0}
    skill_list = []
    file_list = []
    
    model.eval()

    for file in blobs_folder:
        print('Processing file {}'.format(file))

        curr_path = os.path.join(blobs_folder_path, file)
        curr_blob, _ = pickle.load(open(curr_path, 'rb'))
        try:
            curr_blob =  curr_blob.view(1, 50, 240, 320)

            out = model.conv_net_stream(curr_blob)
            out = out.cpu().detach().data.numpy()
            embeddings_list.append(out)

            file_list.append(file)
            file = file.split('_')
            gestures_list.append(file[-1].split('.')[0])
            user_list.append(file[3][0])
            skill_list.append(skill_dict[file[3][0]])
        except:
            pass

    final_dict = {'gesture': gestures_list, 'user': user_list, 'skill': skill_list, 'embeddings': embeddings_list, 'file_list': file_list}
    
    return(final_dict)

def cluster_statistics(blobs_folder_path: str, model: encoderDecoder, num_clusters: int) -> pd.DataFrame:
    results_dict = store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    k_means = KMeans(n_clusters = num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict['embeddings']).reshape(-1, 2048))
    results_dict['cluster_indices'] = cluster_indices
    df = pd.DataFrame(results_dict)
    return(df)

def cluster_statistics_multidata(blobs_folder_paths_list: List[str], model: encoderDecoder, num_clusters: int) -> pd.DataFrame:
    results_dict = {'gesture': [], 'user': [], 'skill': [], 'embeddings': [], 'task': []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_embeddings_in_dict(blobs_folder_path = path, model = model)
        # import pdb; pdb.set_trace()
        temp_results_dict['task'] = [idx]*len(temp_results_dict['skill'])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)
    k_means = KMeans(n_clusters = num_clusters)
    cluster_indices = k_means.fit_predict(np.array(results_dict['embeddings']).reshape(-1, 2048))
    results_dict['cluster_indices'] = cluster_indices
    df = pd.DataFrame(results_dict)
    return(df)

def evaluate_model(blobs_folder_path: str, model: encoderDecoder, num_clusters: int, save_embeddings: bool) -> None:
    df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = num_clusters)
    if save_embeddings:
        print('Saving dataframe.')
        df.to_pickle('./df.p')
    y = df['gesture'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 2048)
    classifier = XGBClassifier(n_estimators = 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 8765)
    
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)
    
    print('Training set classification report.')
    print(classification_report(y_train, y_hat))
    
    print('Test set classification report.')
    print(classification_report(y_test, y_hat_test))

def evaluate_model_multidata(blobs_folder_paths_list: str, model: encoderDecoder, num_clusters: int, save_embeddings: bool, classifier_save_path: str = './xgboost_save/multidata_xgboost.joblib') -> None:
    df = cluster_statistics_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = num_clusters)
    if save_embeddings:
        print('Saving dataframe.')
        df.to_pickle('./df.p')
    y = df['task'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 2048)
    classifier = XGBClassifier(n_estimators = 1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 5113)
    
    print('Fitting classifier.')
    classifier.fit(X_train, y_train)
    y_hat = classifier.predict(X_train)
    y_hat_test = classifier.predict(X_test)
    
    print('Training set classification report.')
    print(classification_report(y_train, y_hat))
    
    print('Test set classification report.')
    print(classification_report(y_test, y_hat_test))

    print('Saving classifier.')
    dump(classifier, classifier_save_path)
    print('Classifier saved.')

def plot_umap_clusters(blobs_folder_path: str, model: encoderDecoder, plot_store_path: str) -> None:
    results_dict = store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    embeddings = np.array(results_dict['embeddings']).squeeze()
    
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['skill']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()

    le_gest = LabelEncoder()
    le_gest.fit(results_dict['gesture'])
    print('Generating gesture plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()

    le_user = LabelEncoder()
    le_user.fit(results_dict['user'])
    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_user.transform(results_dict['user'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_user.png')
    plt.savefig(save_path)
    plt.clf()

def plot_umap_clusters_multidata(blobs_folder_paths_list: str, model: encoderDecoder, plot_store_path: str) -> None:
    if not os.path.exists(plot_store_path):
        os.mkdir(plot_store_path)

    results_dict = {'gesture': [], 'user': [], 'skill': [], 'embeddings': [], 'task': []}
    for idx, path in enumerate(blobs_folder_paths_list):
        temp_results_dict = store_embeddings_in_dict(blobs_folder_path = path, model = model)
        temp_results_dict['task'] = [idx]*len(temp_results_dict['skill'])
        for key, value in temp_results_dict.items():
            results_dict[key].extend(value)

    # import pdb; pdb.set_trace()
    embeddings = np.array(results_dict['embeddings']).squeeze()
    
    print('Training umap reducer.')    
    umap_reducer = umap.UMAP()
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    print('Generating skill plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['skill']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Skill clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_skill.png')
    plt.savefig(save_path)
    plt.clf()

    le_gest = LabelEncoder()
    le_gest.fit(results_dict['gesture'])
    print('Generating gesture plots.')
    # import pdb; pdb.set_trace()
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])#, c=[sns.color_palette()[x] for x in le_gest.transform(results_dict['gesture'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the Gesture clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_gesture.png')
    plt.savefig(save_path)
    plt.clf()

    le_user = LabelEncoder()
    le_user.fit(results_dict['user'])
    print('Generating user plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in le_user.transform(results_dict['user'])])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_user.png')
    plt.savefig(save_path)
    plt.clf()

    le_task = LabelEncoder()
    le_task.fit(results_dict['task'])
    print('Generating task plots.')
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=[sns.color_palette()[x] for x in results_dict['task']])
    plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection of the User clusters', fontsize=24);
    save_path = os.path.join(plot_store_path, 'umap_task.png')
    plt.savefig(save_path)
    plt.clf()

def label_surgical_study_video(optical_flow_path: str, model: encoderDecoder, labels_store_path: str, num_frames_per_blob: int, spacing: int, classifier_load_path: str = './xgboost_save/multidata_xgboost.joblib') -> None:
    print('Loading dataset')
    dataset = surgeonStudyDataset(optical_flow_path = optical_flow_path, num_frames_per_blob = num_frames_per_blob, spacing = spacing)
    dataloader = DataLoader(dataset = dataset, batch_size = 128, shuffle = False, collate_fn = size_collate_fn)

    embeddings = []
    start_seconds_list = []
    end_seconds_list = []
    print('Generating embeddings.')
    for data, start_second, end_second in Bar(dataloader): 
        output = model.conv_net_stream(data)
        output = output.cpu().detach().data.numpy()
        embeddings.append(output)
        start_seconds_list.append(start_second)
        end_seconds_list.append(end_second)
    
    embeddings = np.concatenate(embeddings)
    start_seconds_list = np.concatenate(start_seconds_list)
    end_seconds_list = np.concatenate(end_seconds_list)

    print('Loading XGBoost')
    classifier = load(classifier_load_path)
    
    labels = classifier.predict(embeddings)

    labels = labels.tolist()
    start_seconds_list = start_seconds_list.tolist()
    end_seconds_list = end_seconds_list.tolist()

    labels.insert(0, 'labels')
    start_seconds_list.insert(0, 'start_second')
    end_seconds_list.insert(0, 'end_second')

    print('Saving labels.')

    # Labels are as follows: Needle passing: 0, Knot Tying: 1, Suturing: 2
    
    with open(labels_store_path, 'w') as f:
        for i in range(len(labels)):
            line = str(start_seconds_list[i]) + '\t' +  str(end_seconds_list[i]) + '\t' + str(labels[i])
            f.write(line)
            f.write('\n')

    f.close()
    print('Labels saved.')

def evaluate_model_superuser(blobs_folder_path: str, model: encoderDecoder, transcriptions_path: str, experimental_setup_path: str) -> None:
    transcription_file_names = os.listdir(transcriptions_path)
    transcription_file_names = list(filter(lambda x: '.DS_Store' not in x, transcription_file_names))

    transcription_translation_dict = {}
    count = 0
    for file in transcription_file_names:
        curr_file_path = os.path.join(transcriptions_path, file)
        with open(curr_file_path, 'r') as f:
            for line in f:
                line = line.strip('\n').strip()
                line = line.split(' ')
                start = line[0]
                end = line[1]
                gesture = line[2]
                
                transcription_name = file.split('.')[0] + '_' + start.zfill(6) + '_' + end.zfill(6) + '.txt'
                new_name = 'blob_{}_video'.format(count) + '_'.join(file.split('.')[0].split('_')[0:3]) + '_gesture_' + gesture +'.p'
                new_name = re.sub('Knot_Tying', '', new_name)
                new_name = re.sub('Needle_Passing', '', new_name)
                new_name = re.sub('Suturing', '', new_name)
                transcription_translation_dict[transcription_name] = new_name
                count += 1

    df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = 5)
    
    file_to_index_dict = {}
    file_count = 0
    for file in df['file_list']:
        file_to_index_dict[file] = file_count
        file_count += 1
    
    y = df['skill'].values.ravel()
    X = [np.array(v) for v in df['embeddings']]
    X = np.array(X).reshape(-1, 2048)

    sampler_list = []
    iterations = os.listdir(experimental_setup_path)
    iterations = list(filter(lambda x: '.DS_Store' not in x, iterations))
    
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    itr = 0
    for iter_num in tqdm(iterations):
        directory_path = os.path.join(experimental_setup_path, iter_num)
        
        train_indices = []
        test_indices = []
        
        with open(os.path.join(directory_path, 'Train.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    train_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except:
                    pass
            f.close()
        
        with open(os.path.join(directory_path, 'Test.txt')) as f:
            for line in f:
                items = line.strip('\n').split('           ')
                try:
                    test_indices.append(file_to_index_dict[transcription_translation_dict[items[0]]])
                except:
                    pass
            f.close()

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]

        classifier = XGBClassifier(n_estimators = 1000)
        classifier.fit(X_train, y_train)

        # y_hat = classifier.predict(X_train)
        y_hat_test = classifier.predict(X_test)
        report_test = classification_report(y_test, y_hat_test, output_dict = True)
        # metrics['accuracy'] = (metrics['accuracy']*itr + report_test['accuracy'])/(itr + 1)
        # metrics['precision'] = (metrics['precision']*itr + report_test['weighted avg']['precision'])/(itr + 1)
        # metrics['recall'] = (metrics['recall']*itr + report_test['weighted avg']['recall'])/(itr + 1)
        # metrics['f1-score'] = (metrics['f1-score']*itr + report_test['weighted avg']['f1-score'])/(itr + 1)
        # metrics['support'] = (metrics['support']*itr + report_test['weighted avg']['support'])/(itr + 1)
        # itr += 1

        metrics['accuracy'].append(report_test['accuracy'])
        metrics['precision'].append(report_test['weighted avg']['precision'])
        metrics['recall'].append(report_test['weighted avg']['recall'])
        metrics['f1-score'].append(report_test['weighted avg']['f1-score'])
        metrics['support'].append(report_test['weighted avg']['support'])
        
    for key, val in metrics.items():
        print('Mean {} : {} \t \t Std {} : {}'.format(key, np.mean(val), key, np.std(val)))

def main():
    blobs_folder_path = '../jigsaw_dataset/Suturing/blobs'
    blobs_folder_paths_list = ['../jigsaw_dataset/Needle_Passing/blobs', '../jigsaw_dataset/Knot_Tying/blobs', '../jigsaw_dataset/Suturing/blobs']
    
    model = encoderDecoder(embedding_dim = 2048)
    model.load_state_dict(torch.load('./weights_save/suturing_weights/suturing_2048.pth'))

    # store_embeddings_in_dict(blobs_folder_path = blobs_folder_path, model = model)
    # df = cluster_statistics(blobs_folder_path = blobs_folder_path, model = model, num_clusters = num_clusters)
    # return(df)

    # evaluate_model(blobs_folder_path = blobs_folder_path, model = model, num_clusters = 10, save_embeddings = False)
    # evaluate_model_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, num_clusters = 10, save_embeddings = False)

    # plot_umap_clusters(blobs_folder_path = blobs_folder_path, model = model, plot_store_path = './umap_plots/Needle_Passing')
    # plot_umap_clusters_multidata(blobs_folder_paths_list = blobs_folder_paths_list, model = model, plot_store_path = './umap_plots/multidata')

    optical_flow_path = '../jigsaw_dataset/Surgeon_study_videos/optical_flow/left_suturing.p'
    labels_store_path = '../jigsaw_dataset/Surgeon_study_videos/multimodal_labels/multidata_2048_labels.txt'
    num_frames_per_blob = 25
    spacing = 2
    label_surgical_study_video(optical_flow_path = optical_flow_path, model = model, labels_store_path = labels_store_path, num_frames_per_blob = num_frames_per_blob, spacing = spacing)

    transcriptions_path = '../jigsaw_dataset/Suturing/transcriptions/'
    experimental_setup_path = '../jigsaw_dataset/Experimental_setup/Suturing/Balanced/GestureClassification/UserOut/1_Out/'
    # evaluate_model_superuser(blobs_folder_path = blobs_folder_path, model = model, transcriptions_path = transcriptions_path, experimental_setup_path = experimental_setup_path)

if __name__ == '__main__':
    main()
