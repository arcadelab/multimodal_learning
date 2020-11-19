import argparse
from ast import parse
from training import train_encoder_decoder_embeddings, train_encoder_decoder_multidata_embeddings
from video_preprocessing import computeOpticalFlow
from dataloader import create_data_blobs
from embeddings_cluster_explore import evaluate_model, evaluate_model_multidata, plot_umap_clusters, plot_umap_clusters_multidata, label_surgical_study_video
from neural_networks import encoderDecoder
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Argument parser to pre-process, train, evaluate and plot results from an encoder-decoder architecture.')
    parser.add_argument('--mode', metavar='--mode', type=str, required=True)

    # Training arguments
    parser.add_argument('--lr', metavar='--lr', type=str)
    parser.add_argument('--num_epochs', metavar='--num_epochs', type=str)
    parser.add_argument('--blobs_folder_path', metavar='--blobs_folder_path', type=list, nargs='+')
    parser.add_argument('--weights_save_path', metavar='--weights_save_path', type=str)

    # Preprocess arguments
    parser.add_argument('--source_directory', metavar='--source_directory', type=str)
    parser.add_argument('--resized_video_directory', metavar='--resized_video_directory', type=str)
    parser.add_argument('--destination_directory', metavar='--destination_directory', type=str)
    parser.add_argument('--resize_dim', metavar='--resize_dim', type=list, nargs=2)

    parser.add_argument('--optical_flow_path', metavar='--optical_flow_path', type=str)
    parser.add_argument('--transcriptions_path', metavar='--transcriptions_path', type=str)
    parser.add_argument('--kinematics_path', metavar='--kinematics_path', type=str)
    parser.add_argument('--frames_per_blob', metavar='--frames_per_blob', type=str)
    parser.add_argument('--blobs_path', metavar='--blobs_path', type=str)
    parser.add_argument('--spacing', metavar='--spacing', type=str)

    # Eval arguments
    parser.add_argument('--model_dim', metavar='--model_dim', type=str)
    parser.add_argument('--plot_save_path', metavar='--plot_save_path', type=str)
    parser.add_argument('--experimental_setup_path', metavar='--experimental_setup_path', type=str)
    parser.add_argument('--labels_store_path', metavar='--labels_store_path', type=str)

    args = parser.parse_args()

    if args.mode == 'train':
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_path = args.blobs_folder_path[0]
            if 'knot' or 'tying' in blobs_folder_path.lower():
                dataset_name = 'Knot_Tying'
            elif 'needle' or 'passing' in blobs_folder_path.lower():
                dataset_name = 'Needle_Passing'
            elif 'suturing' in blobs_folder_path.lower():
                dataset_name = 'Suturing'
            else:
                dataset_name = 'dataset'
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        train_encoder_decoder_embeddings(lr=lr, num_epochs=num_epochs, blobs_folder_path=blobs_folder_path,
                                         weights_save_path=weights_save_path, weight_decay=weight_decay,
                                         dataset_name=dataset_name)

    elif args.mode == 'multidata_train':
        weight_decay = 1e-8
        try:
            lr = float(args.lr)
        except:
            lr = 1e-4
        try:
            num_epochs = int(args.num_epochs)
        except:
            num_epochs = 1000
        try:
            blobs_folder_paths_list = args.blobs_folder_path
        except Exception as e:
            print(e)
        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)
        train_encoder_decoder_multidata_embeddings(lr=lr, num_epochs=num_epochs,
                                                   blobs_folder_paths_list=blobs_folder_paths_list,
                                                   weights_save_path=weights_save_path, weight_decay=weight_decay)

    elif args.mode == 'optical_flow':
        try:
            source_directory = args.source_directory
        except Exception as e:
            print(e)

        try:
            resized_video_directory = args.resized_video_directory
        except Exception as e:
            print(e)

        try:
            destination_directory = args.destination_directory
        except Exception as e:
            print(e)

        try:
            resize_dim = tuple(args.resize_dim)
        except:
            resize_dim = (320, 240)
        optical_flow_compute = computeOpticalFlow(source_directory=source_directory,
                                                  resized_video_directory=resized_video_directory,
                                                  destination_directory=destination_directory, resize_dim=resize_dim)
        optical_flow_compute.run()

    elif args.mode == 'data_blobs':
        try:
            optical_flow_folder_path = args.optical_flow_path
        except Exception as e:
            print(e)

        try:
            transcriptions_folder_path = args.transcriptions_path
        except Exception as e:
            print(e)

        try:
            kinematics_folder_path = args.kinematics_path
        except Exception as e:
            print(e)

        try:
            num_frames_per_blob = int(args.frames_per_blob)
        except:
            num_frames_per_blob = 25

        try:
            blobs_save_folder_path = args.blobs_path
        except Exception as e:
            print(e)

        try:
            spacing = int(args.spacing)
        except:
            spacing = 2

        create_data_blobs(optical_flow_folder_path=optical_flow_folder_path,
                          transcriptions_folder_path=transcriptions_folder_path,
                          kinematics_folder_path=kinematics_folder_path, num_frames_per_blob=num_frames_per_blob,
                          blobs_save_folder_path=blobs_save_folder_path, spacing=spacing)

    elif args.mode == 'eval':
        try:
            blobs_folder_path = args.blobs_path
        except Exception as e:
            print(e)

        try:
            weights_save_path = args.weights_save_path
        except Exception as e:
            print(e)

        try:
            model_dim = int(args.model_dim)
        except:
            model_dim = 2048

        model = encoderDecoder(embedding_dim=model_dim)
        model.load_state_dict(torch.load(weights_save_path))
        evaluate_model(blobs_folder_path=blobs_folder_path, model=model, num_clusters=10, save_embeddings=False)


if __name__ == '__main__':
    main()