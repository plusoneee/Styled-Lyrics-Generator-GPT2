import logging
import argparse
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import utils.utilities as U

import config as cnf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_train_data(enc, device):
    raw_dataset = U.load_dataset(cnf.TRAIN_DATA_PATH)
    formated_dataset = U.format_n_tokenize_data(raw_dataset, enc)
    train_tensor_data = U.construct_input(formated_dataset, device, enc, max_input_len=enc.max_len)

    # Load onto the Pytorch DataLoader
    # Note: the '*' extracts all elements from the list
    train_data = TensorDataset(*train_tensor_data)
    train_sampler = RandomSampler(train_data)
    train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=cnf.TRAIN_BATCH_SIZE)
    return train_data_loader


def main():
    
    device, n_gpu = U.get_device(logger)

    output_dir = U.create_save_path(cnf, __file__)
    run_details_file = os.path.join(output_dir, "run_details.txt")
    # tb_dir = os.path.join(output_dir, "all_scalars.json")
    tb_writer = SummaryWriter(output_dir)

    special_tokens_dict = {
        'additional_special_tokens':['[s:emo]', '[e:emo]', '[s:lyrics]', '[e:lyrics]']
    }
    U.log_arguments(run_details_file, special_tokens_dict["additional_special_tokens"], cnf)

    # Initialise model & tokenizer
    enc = GPT2Tokenizer.from_pretrained(cnf.MODEL_SIZE, pad_token='<|pad|>')
    enc.add_special_tokens(special_tokens_dict)

    model = GPT2LMHeadModel.from_pretrained(cnf.MODEL_SIZE)
    model.resize_token_embeddings(len(enc))

    # Prepare training data
    train_data_loader = prepare_train_data(enc, device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cnf.WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimization_steps = ((len(train_data_loader) * cnf.NUM_TRAIN_EPOCH) // \
                         (cnf.TRAIN_BATCH_SIZE * cnf.GRADIENT_ACCMULATION_STPES)) + 1000

    # TODO: Could use NVIDIA Apex for lower precision calculations.
    optimizer = AdamW(optimizer_grouped_parameters, lr=cnf.LEARNING_RATE, eps=cnf.ADAM_EPSILON)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cnf.WARMUP_STEPS, num_training_steps=optimization_steps)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # @                            FINE-TUNE GPT2
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    if cnf.TRAIN_MODEL:
        
        logger.info("\nFine-tuning GPT2")
        print("To visualise data using TensorBoardX -> type in console:\ntensorboard --logdir={}".format(output_dir))
        model.to(device)
        model.train()

        for epoch in trange(int(cnf.NUM_TRAIN_EPOCH), desc="Epoch"):
            past = None
            if epoch > 0:
                # Re-process dataset since the features dropout is random.
                train_data_loader = prepare_train_data(enc, device)

            for step, batch in enumerate(tqdm(train_data_loader, desc="Training")):
                tok_ids, tok_type_ids, pos_ids, att_mask = batch
                
                outputs = model(
                    input_ids=tok_ids, 
                    past=past, 
                    attention_mask=att_mask, 
                    token_type_ids=tok_type_ids,
                    position_ids=pos_ids, 
                    labels=tok_ids
                )
                
                loss = outputs[0]
                # predicted_scores = outputs[1]
                # past = outputs[2]

                # Log the loss to TensorBoardX
                global_step = (epoch * len(train_data_loader)) + (step + 1)
                tb_writer.add_scalar('loss', loss.item(), global_step)

                # Normalise the loss (Simulates average of a batch)
                loss = loss / cnf.GRADIENT_ACCMULATION_STPES
                loss.backward(retain_graph=True)

                if (step + 1) % cnf.GRADIENT_ACCMULATION_STPES == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cnf.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            if (epoch + 1) % cnf.SAVE_EVERY_N_EPOCH == 0:
                save_model_dir = U.make_dir(os.path.join(output_dir, "model_epoch_" + str(epoch + 1)))
                model.save_pretrained(save_model_dir)
                enc.save_pretrained(save_model_dir)

        tb_dir = os.path.join(output_dir, "all_scalars.json")
        tb_writer.export_scalars_to_json(tb_dir)
        tb_writer.close()

        # Save model and tokenizer to a directory
        save_model_dir = U.make_dir(os.path.join(output_dir, "model_epoch_" + str(epoch + 1)))
        model.save_pretrained(save_model_dir)
        enc.save_pretrained(save_model_dir)


if __name__ == '__main__':
    main()