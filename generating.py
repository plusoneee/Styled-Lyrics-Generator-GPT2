import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import utilities as U
import config as cnf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_token_types(input, enc):
    """
    This method generates toke_type_ids that correspond to the given input_ids.
    :param input: Input_ids (tokenised input)
    :param enc: Model tokenizer object
    :return: A list of toke_type_ids corresponding to the input_ids
    """
    meta_dict = {
        "emo": {
            "st_token": "[s:emo]",
            "end_token": "[e:emo]",
            "tok_type_id": 1
        },
        "lyrics": {
            "st_token": "[s:lyrics]",
            "end_token": "[e:lyrics]",
            "tok_type_id": 2
        }
    }

    tok_type_ids = [0] * len(input)

    for feature in meta_dict.keys():
        start_tok_id = enc.added_tokens_encoder[meta_dict[feature]["st_token"]]
        end_tok_id = enc.added_tokens_encoder[meta_dict[feature]["end_token"]]
        tok_type_val = meta_dict[feature]["tok_type_id"]

        # If this feature exists in the input, find out its indexes
        if start_tok_id and end_tok_id in input:
            st_indx = input.index(start_tok_id)
            end_indx = input.index(end_tok_id)
            tok_type_ids[st_indx:end_indx+1] = [tok_type_val] * ((end_indx-st_indx) + 1)
        # This means that this is the token we are currently predicting
        elif start_tok_id in input:
            st_indx = input.index(start_tok_id)
            tok_type_ids[st_indx:] = [tok_type_val] * (len(input)-st_indx)

    return tok_type_ids


def generate_lyrics(model, enc, gen_batch, context, end_token, device):
    """
    Generates a sequence of words from the fine-tuned model, token by token. This method generates with the
    token_type_ids and position_ids -> since the fine-tuned model is trained with the former input partitions.
    Note: When generating with the 'past', it is required to pass the single generated token only, without
    the previous tokens (not the concatination of the previous input + the generated token).
    :param model: Loaded fine-tune model object
    :param enc: Loaded tokeniser object
    :param args: Arguments passed for the generation
    :param context: Tokenized input_ids on which the output generations will be based on.
    :param end_token: Signal to cut off further generation, e.g., [e:lyrics]
    :param device: Device on which the model will be run on
    :return: Generated lyrics along with the condition provided.
    """
    # Pack in tensor and correct shape
    input_ids = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch, 1)
    position_ids = torch.arange(0, len(context), device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch, 1)
    token_type_ids = torch.tensor(get_token_types(context, enc), device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch, 1)

    # 'Output' stores the concatination of word by word prediction
    output = input_ids.tolist()

    # Get the end of generation signal, token_id of, e.g., [e:lyrics]
    end_token_id = enc.added_tokens_encoder[end_token]
    max_len = enc.max_len

    with torch.no_grad():
        past = None
        keep_gen_4_these_batches = np.arange(0, gen_batch).tolist()
        for _ in trange(len(context), max_len):
            logits, past = model(input_ids=input_ids,
                                 position_ids=position_ids,
                                 token_type_ids=token_type_ids,
                                 past=past)

            next_token_logits = logits[:, -1, :]
            filtered_logits = U.top_k_top_p_filtering(next_token_logits, top_k=0, top_p=0.95)
            log_probs = F.softmax(filtered_logits, dim=-1)
            next_token_id = torch.multinomial(log_probs, num_samples=1)

            # Since we are using past, the model only requires the generated token as the next input
            input_ids = next_token_id
            position_ids = torch.tensor(len(output[0]), device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch, 1)
            # What ever was the last element we want the same value for the next toke_type_id
            token_type_ids = torch.tensor(token_type_ids[0][-1].item(), device=device, dtype=torch.long).unsqueeze(0).repeat(gen_batch, 1)

            # The gen should stop when the end tag reached; however, we can predict a few songs at a time (batch).
            # Solution: keep generating until model predicts the end signal for ALL batch indexes, but, only append
            # the predicted tokens to batch indexes that have not seen the end signal yet.
            if len(keep_gen_4_these_batches) > 0:
                for batch_indx in keep_gen_4_these_batches:
                    output[batch_indx].append(next_token_id[batch_indx].item())

                    if next_token_id[batch_indx].item() == end_token_id:
                        indx = keep_gen_4_these_batches.index(batch_indx)
                        keep_gen_4_these_batches.pop(indx)
            else:
                # Break out when predicted end signal for all batch indexes
                break

    return output


def main(lyric=None, emotion=None):
    device, n_gpu = U.get_device(logger)
    # Reload the model and the tokenizer
    model = GPT2LMHeadModel.from_pretrained(cnf.LOAD_MODEL_DIR)
    enc = GPT2Tokenizer.from_pretrained(cnf.LOAD_MODEL_DIR)
    model.to(device)
    model.eval()
    U.set_seed(np.random.randint(0, 100))

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @                    GENERATE FROM FINE-TUNED GPT2
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if lyric is None:
        emotion = "relaxed"
        lyric = "For all the times that you rained on my parade"
        context = "[s:emo]" + emotion + "[e:emo]" + "[s:lyrics]" + lyric 
        end_token = "[e:lyrics]"
    else:
        context = "[s:emo]" + emotion + "[e:emo]" + "[s:lyrics]" + lyric 
        end_token = "[e:lyrics]"
    
    context = enc.encode(context)
    gen_batch = cnf.GEN_BATCH

    sequence_batch = generate_lyrics(model, enc, gen_batch, context, end_token, device)

    for seq in sequence_batch:
        print(enc.decode(seq))
        print("\n---------------\n")


if __name__ == '__main__':
    main()
