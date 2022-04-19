import logging
import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
import utilities as U
import config as cnf

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT2Trainer:

    def __init__(self):
        self.device, self.n_gpu = U.get_device(logger)
        self.output_dir = U.create_save_path(cnf, __file__)
        run_details_file = os.path.join(self.output_dir, "run_details.txt")
        self.tb_writer = SummaryWriter(self.output_dir)

        self.special_tokens_dict = {
        'additional_special_tokens':['[s:emo]', '[e:emo]', '[s:lyrics]', '[e:lyrics]']
        }

        U.log_arguments(run_details_file, self.special_tokens_dict["additional_special_tokens"], cnf)

        # Initialise model & tokenizer
        self.tokenizer = self.init_tokenizer()
        self.model = self.init_model(self.tokenizer)

        # Prepare training data
        self.train_data_loader = self.prepared_train_data()

        # Prepare optimizer and schedule (linear warmup and decay)
        self.optimizer, self.scheduler = self.init_optimizer()

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cnf.WEIGHT_DECAY},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimization_steps = ((len(self.train_data_loader) * cnf.NUM_TRAIN_EPOCH) // \
                         (cnf.TRAIN_BATCH_SIZE * cnf.GRADIENT_ACCMULATION_STPES)) + 1000

        optimizer = AdamW(optimizer_grouped_parameters, lr=cnf.LEARNING_RATE, eps=cnf.ADAM_EPSILON)    
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=cnf.WARMUP_STEPS, 
            num_training_steps=optimization_steps
            )
        return optimizer, scheduler

    def init_tokenizer(self):
        tokenizer = GPT2Tokenizer.from_pretrained(cnf.MODEL_SIZE, pad_token='<|pad|>')
        tokenizer.add_special_tokens(self.special_tokens_dict)
        return tokenizer

    def init_model(self, tokenizer):
        model = GPT2LMHeadModel.from_pretrained(cnf.MODEL_SIZE)
        model.resize_token_embeddings(len(tokenizer))
        return model

    def train(self):
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # @                            FINE-TUNE GPT2
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
        logger.info("\nFine-tuning GPT2")
        logger.info("To visualise data using TensorBoardX -> type in console:\ntensorboard --logdir={}".format(self.output_dir))
        self.model.to(self.device)
        self.model.train()
        
        for epoch in trange(int(cnf.NUM_TRAIN_EPOCH), desc="Epoch"):
            past = None
            if epoch > 0:
                # Re-process dataset since the features dropout is random.
                self.train_data_loader = self.prepared_train_data()
            
            for step, batch in enumerate(tqdm(self.train_data_loader, desc="Training")):
                tok_ids, tok_type_ids, pos_ids, att_mask = batch
                outputs = self.model(
                    input_ids=tok_ids, 
                    past=past, 
                    attention_mask=att_mask, 
                    token_type_ids=tok_type_ids,
                    position_ids=pos_ids, 
                    labels=tok_ids
                )
                
                loss = outputs[0]
                # predicted_scores, past = outputs[1],  outputs[2]

                # Log the loss to TensorBoardX
                global_step = (epoch * len(self.train_data_loader)) + (step + 1)
                self.tb_writer.add_scalar('loss', loss.item(), global_step)

                # Normalise the loss (Simulates average of a batch)
                loss = loss / cnf.GRADIENT_ACCMULATION_STPES
                loss.backward(retain_graph=True)

                if (step + 1) % cnf.GRADIENT_ACCMULATION_STPES == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cnf.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            
            if (epoch + 1) % cnf.SAVE_EVERY_N_EPOCH == 0:
                save_model_dir = U.make_dir(os.path.join(self.output_dir, "model_epoch_" + str(epoch + 1)))
                self.save_trained_model(save_model_dir)
                self.load_trained_model(save_model_dir)
            
        tb_dir = os.path.join(self.output_dir, "all_scalars.json")
        self.tb_writer.export_scalars_to_json(tb_dir)
        self.tb_writer.close()

        # Save model and tokenizer to a directory
        save_model_dir = U.make_dir(os.path.join(self.output_dir, "model_epoch_" + str(epoch + 1)))
        self.save_trained_model(save_model_dir)

    def save_trained_model(self, checkpoint_dir):
        checkpoint_filepath = checkpoint_dir+os.sep+'checkpoint.bin'
        torch.save(self.model.state_dict(), checkpoint_filepath)
        self.tokenizer.save_pretrained(checkpoint_dir)

    def load_trained_model(self, checkpoint_dir):
        checkpoint_filepath = checkpoint_dir+os.sep+'checkpoint.bin'
        checkpoint = torch.load(checkpoint_filepath)
        self.model.load_state_dict(checkpoint)
        self.tokenizer.from_pretrained(checkpoint_dir)
        
    def prepared_train_data(self):
        raw_dataset = U.load_dataset(cnf.TRAIN_DATA_PATH)
        formated_dataset = U.format_n_tokenize_data(raw_dataset, self.tokenizer)
        train_tensor_data = U.construct_input(
            formated_dataset, 
            self.device, 
            self.tokenizer, 
            max_input_len=self.tokenizer.max_len
            )
        # Load onto the Pytorch DataLoader
        # Note: the '*' extracts all elements from the list
        train_data = TensorDataset(*train_tensor_data)
        train_sampler = RandomSampler(train_data)
        train_data_loader = DataLoader(train_data, sampler=train_sampler, batch_size=cnf.TRAIN_BATCH_SIZE)
        return train_data_loader


if __name__ == '__main__':
    trainer = GPT2Trainer()
    trainer.train()