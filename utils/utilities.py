import datetime
import os
import shutil
import csv
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm


def make_dir(dir_path):
    """
    Makes a directory if already doesn't exist
    :param dir_path: Directory path to be created
    :return: Directory path (str)
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def set_seed(seed):
    """
    Set model random seed. The model outputs are seed dependent.
    :param seed: An int.
    :return: No return
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_device(logger):
    """
    Get device model will be run on (GPU or CPU)
    :param logger: Logger object to note the device
    :return: device type, num_of_gpus
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))
    return device, n_gpu


def create_save_path(args, execution_file_path):
    """
    1) Constructs a model save path: "master_dir/(optional folder)/train_data_name/model_size__date)"
    2) Creates a copy of the main code.
    :param args: Model arguments object
    :param execution_file_path: file path to the main code
    :return: Training specific directory where everything will be saved to.
    """
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    
    # Extract dataset file name from the full path
    dataset_name = os.path.basename(os.path.normpath(args.train_data_path)).split(".")[0]

    if args.store_in_folder:
        
        log_path = "{}/{}/{}_{}".format(args.store_in_folder, dataset_name, args.model_size, now)
    else:
        log_path = "{}/{}/{}_{}".format(dataset_name, args.model_size, now)
    make_dir(log_path)

    # COPY OF THE MAIN CODE
    shutil.copy2(execution_file_path, "{}/copy_of_code_that_run_this_experiment.py".format(log_path))
    return log_path


def log_arguments(run_details_file_path, args, special_tokens):
    """
    Saves training information to a file, like arguments and special tokens.
    :param run_details_file_path: File to be written to
    :param args: Model arguments object
    :param special_tokens: Special tokens used in this training
    :return: No return
    """
    now = datetime.datetime.now().strftime("%d-%m-%Y@%H'%M")
    # Open a file and appends to a file. If doesn't exists (+) means to create it.
    d_file = open(run_details_file_path, "a+")
    d_file.write("@" * 30 + " RUN INFO " + "@" * 30)
    d_file.write("\n\nDATE: {}".format(now))
    d_file.write("\n\nUSING THE FOLLOWING ARGS:\n{}".format(args))
    d_file.write("\n\nSPECIAL TOKENS: {}".format(special_tokens))
    d_file.close()


def save_dataset(path, input, append=True):
    """
    Saves data to a file path.
    :param path: Save file path
    :param input: Data to be saved
    :param append: Whether we should append data or write to clean file
    :return: No return
    """
    if append:
        with open(path, 'a+', encoding='utf_8') as f:
            writer = csv.writer(f)
            writer.writerows(input)
    else:
        with open(path, 'w+', encoding='utf_8') as f:
            writer = csv.writer(f)
            writer.writerows(input)
    f.close()


def load_dataset(dataset_path):
    """
    Loads lyrics dataset of the following format (genre, artist, year, album, song_name, lyrics)
    :param dataset_path: Dataset file path (type csv)
    :return: List of tuples where each entry contains a song and its metadata
    """
    df = pd.read_csv(dataset_path).dropna()
    output = []
    for idx, row in tqdm(df.iterrows()):
        emotion = row['genre']
        lyric = row['lyric']
        
        output.append((emotion, lyric))
    return output


def format_n_tokenize_data(raw_dataset, enc, max_input_len=1024):
    """
    Seperates metadata with respective special tokens and then tokenizes the formated text
    :param raw_dataset: Text to format and style
    :param enc: Tokenizer object
    :return: Formated data in the form of tuples
    """
    ### TODO: make training examples where lyrics are as a condition to predict features

    # Get the dict: special token -> token id
    spe = enc.added_tokens_encoder

    formated_data = []
    for emo, lyrics in raw_dataset:
        emo = [spe["[s:emo]"]] + enc.encode(emo) + [spe["[e:emo]"]]
        emo_len = len(emo) # 3
        lyric_cut_len = max_input_len - emo_len - 2 # 2 = [s:lyrics] & [e:lyrics]
        ly = [spe["[s:lyrics]"]] + enc.encode(lyrics[:lyric_cut_len]) + [spe["[e:lyrics]"]]
        formated_data.append((emo, ly))
    return formated_data


def construct_input(formated_data, device, enc, max_input_len=1024):
    """
    Given a tokenized dataset, this method constructs inputs required for the GPT2 model fine-tuning.
    In particular, it creates token_type_ids & positional_ids, randomly drops lyrics' features, applies padding and
    creates language modelling labels, as well as the attention masks.
    Refer to - https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel - for an indication of the inputs
    :param formated_data: Tokenised dataset with special tokens inplace provided in the form of tuple(all song features)
    :param device: Device that will run this code (GPU, CPU)
    :param max_input_len: Max input length allowed by the model
    :return: Tuple of tensors: (token_ids, token_type_ids, position_ids, attention_mask, lm_labels)
             where each is of shape: (num_of_inputs * batch_size * sequence_length) -> (N, 1, 1024)
    """
    sucessfull_candidates = []
    for emo, lyrics in formated_data:
        # 1) Prepare input partitions, i.e., token type ids & position ids
        # Token type ids, alternatively called segment ids
        emo_seg = list([1] * len(emo))
        lyr_seg = list([2] * len(lyrics))

        # 2) Randomly drop features for model to learn to handle subset of conditions
        # 10% to drop all metadata but lyrics
        if np.random.rand() <= 0.1:
            # An integer sequence (0 -> input_len)
            position_ids = list(np.arange(0, len(lyrics)))
            curr_input = {
                "tok_ids": lyrics,
                "tok_type_ids": lyr_seg,
                "pos_ids": position_ids
            }
        else:
            tokens_subset = []
            segment_subset = []

            tokens_subset += emo
            segment_subset += emo_seg

            # Add lyrics in all cases -> add lyrics
            tokens_subset += lyrics
            segment_subset += lyr_seg
            position_ids = list(np.arange(0, len(tokens_subset)))

            curr_input = {
                "tok_ids": tokens_subset,
                "tok_type_ids": segment_subset,
                "pos_ids": position_ids
            }
        
        
        # Get rid of songs longer than allowed size, alternatively we could cut off the excess
        if len(curr_input["tok_ids"]) > max_input_len:
            print(len(curr_input["tok_ids"]))
            continue
        
        # 3) Add padding to make the input max_input_len
        len_before_padding = len(curr_input["tok_ids"])
        padding = max_input_len - len_before_padding

        curr_input["tok_ids"] += list([enc.pad_token_id] * padding)
        curr_input["tok_type_ids"] += list([0] * padding)
        curr_input["pos_ids"] += list([0] * padding)
     
        # 4) Attention Mask, 1 = unmasked, 0 = masked
        attention_mask = list([1] * len_before_padding) + list([0] * padding)
        sucessfull_candidates.append((
            curr_input["tok_ids"], 
            curr_input["tok_type_ids"], 
            curr_input["pos_ids"], 
            attention_mask
        ))

    # We need the model inputs separate for the DataLoader
    # From tuples of (N, 5, 1024) -> (N, 1024) x 5
    # Note: inputs contains 5 lists
    inputs = map(list, zip(*sucessfull_candidates))

    # Transform each input into a tensor of shape:
    # (num_inputs, batch_size, sequence_len) -> (N, 1, 1024)
    dataset = [torch.tensor(t, device=torch.device(device)).unsqueeze(1) for t in inputs]

    return dataset


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    :param logits: Logits distribution shape (batch size x vocabulary size)
    :param top_k: Keep only top k tokens with highest probability (top-k filtering).
    :param top_p: Keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    :param filter_value: Value that will be ignored by in the softmax
    :return: Filtered logits
    """

    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits