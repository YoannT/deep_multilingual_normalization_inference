from nlstruct.text import huggingface_tokenize, regex_sentencize, partition_spans, encode_as_tag, split_into_spans, apply_substitutions, apply_deltas, reverse_deltas
# from nlstruct.dataloaders import load_from_brat
# from nlstruct.collections import Dataset, Batcher
from nlstruct.utils import merge_with_spans, normalize_vocabularies, factorize_rows, df_to_csr, factorize, torch_global as tg
# from nlstruct.modules.crf import BIODecoder, BIOULDecoder
# from nlstruct.environment import root, cached
# from nlstruct.train import seed_all
# from itertools import chain, repeat

# import numpy as np
# import torch
# import torch.nn.functional as F
# import math

# import pandas as pd
# import numpy as np
# import re
import string
from transformers import AutoModel, AutoTokenizer

# import os
# import traceback
from tqdm import tqdm

# from transformers import AdamW, BertModel

# from scipy.sparse import csr_matrix

# from nlstruct.environment import get_cache, load
# from nlstruct.utils import evaluating, torch_global as tg, freeze
# from nlstruct.scoring import compute_metrics, merge_pred_and_gold
# from nlstruct.train import make_optimizer_and_schedules, run_optimization, seed_all
# from nlstruct.train.schedule import ScaleOnPlateauSchedule, LinearSchedule, ConstantSchedule


# import networkx as nx
from nlstruct.utils import encode_ids

# def extract_mentions(batcher, all_nets, hyperparameters, max_depth=10, batch_size=32):
#     """
#     Parameters
#     ----------
#     batcher: Batcher 
#         The batcher containing the text from which we want to extract the mentions (and maybe the gold mentions)
#     ner_net: torch.nn.Module
#     max_depth: int
#         Max number of times we run the model per sample
        
#     Returns
#     -------
#     Batcher
#     """
#     pred_batches = []
#     n_mentions = 0
#     ner_net = all_nets["ner_net"]
#     tag_embeddings = all_nets["tag_embeddings"]
#     with evaluating(all_nets):
#         with torch.no_grad():
#             for batch_i, batch in enumerate(tqdm(batcher['sentence'].dataloader(batch_size=batch_size, shuffle=False, sparse_sort_on="token_mask", device=tg.device))):

#                 tag_embeds = torch.zeros(*batch["sentence", "token"].shape[:2], hyperparameters["tag_dim"], device=tg.device)
#                 current_sentences_idx = torch.arange(len(batch), device=tg.device)
#                 mask = batch["token_mask"]
#                 tokens = batch["token"]

#                 for i in range(max_depth):
#                     # Run the model argmax here
#                     ner_res = ner_net(
#                         tokens = tokens,
#                         mask = mask,
#                         tag_embeds = tag_embeds,
#                         return_embeddings=True
#                     )

#                     # Run the linear CRF Viterbi algorithm to compute the most likely sequence
#                     pred_tags = ner_net.crf.decode(ner_res["scores"], mask)
#                     spans = ner_net.crf.tags_to_spans(pred_tags, mask)

#                     # Save predicted mentions
#                     pred_batch = Batcher({
#                         "mention": {
#                             "mention_id": torch.arange(n_mentions, n_mentions+len(spans["span_doc_id"]), device=tg.device),
#                             "begin": spans["span_begin"],
#                             "end": spans["span_end"],
#                             "ner_label": spans["span_label"],
#                             "@sentence_id": current_sentences_idx[spans["span_doc_id"]],
#                             "depth": torch.full_like(spans["span_begin"], fill_value=i),
#                         },
#                         "sentence": dict(batch["sentence", ["sentence_id", "doc_id"]]),
#                         "doc": dict(batch["doc"])}, 
#                         check=False).sparsify()
#                     pred_batches.append(pred_batch)
#                     n_mentions += len(spans["span_doc_id"])

#                     non_empty_sentences = torch.unique(spans["span_doc_id"])

#                     if len(non_empty_sentences) == 0:
#                         break

#                     # Convert the predicted spans to tags using the same encoding scheme as the one used to decode predicted tags
#                     # (We could use a different one: BIODecoder/BIOULDecoder.spans_to_tags is a static function)
#                     feature_tags = ner_net.crf.spans_to_tags(
#                         torch.arange(len(spans["span_begin"]), device=spans["span_begin"].device),
#                         spans["span_begin"], 
#                         spans["span_end"],
#                         spans["span_label"], 
#                         n_tokens=batch["sentence", "token"].shape[1],
#                         n_samples=len(spans["span_begin"]),
#                     )
#                     tag_mention, tag_positions = feature_tags.nonzero(as_tuple=True)
#                     tag_sentence = spans["span_doc_id"][tag_mention]
#                     tag_values = feature_tags[tag_mention, tag_positions]

#                     tag_embeds = tag_embeds.view(-1, hyperparameters["tag_dim"]).index_add_(
#                         dim=0,
#                         index=tag_sentence * batch["sentence", "token"].shape[1] + tag_positions, 
#                         source=tag_embeddings.weight[tag_values-1]).view(len(current_sentences_idx), batch["sentence", "token"].shape[1], hyperparameters["tag_dim"])[non_empty_sentences]

#                     # Compute the tokens label tag embeddings of the observed (maybe overlapping) mentions
#                     tokens = tokens[non_empty_sentences]
#                     mask = mask[non_empty_sentences]
#                     current_sentences_idx = current_sentences_idx[non_empty_sentences]
#     return Batcher.concat(pred_batches)


# def make_batcher(docs, sentences, tokens):
#     """
#     Parameters:
#     ----------
#     docs: pd.DataFrame
#     sentences: pd.DataFrame
#     zones: pd.DataFrame
#     mentions: pd.DataFrame
#     conflicts: pd.DataFrame
#     tokens: pd.DataFrame
    
#     Returns
#     -------
#     Batcher
#     """
#     docs = docs.copy()
#     sentences = sentences.copy()
#     tokens = tokens.copy()
    
#     [tokens["token_id"]], unique_token_id = factorize_rows([tokens["token_id"]])
#     [sentences["sentence_id"], tokens["sentence_id"],], unique_sentence_ids = factorize_rows(
#         [sentences[["doc_id", "sentence_id"]], tokens[["doc_id", "sentence_id"]]])
#     [docs["doc_id"], sentences["doc_id"], tokens["doc_id"]], unique_doc_ids = factorize_rows(
#         [docs["doc_id"], sentences["doc_id"], tokens["doc_id"]])
    
#     batcher = Batcher({
#         "sentence": {
#             "sentence_id": sentences["sentence_id"],
#             "doc_id": sentences["doc_id"],
#             "token": df_to_csr(tokens["sentence_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(unique_sentence_ids)),
#             "token_mask": df_to_csr(tokens["sentence_id"], tokens["token_idx"], n_rows=len(unique_sentence_ids)),
#         },
#         "doc": {
#             "doc_id": np.arange(len(unique_doc_ids)),
#             "sentence_id": df_to_csr(sentences["doc_id"], sentences["sentence_idx"], sentences["sentence_id"], n_rows=len(unique_doc_ids)),
#             "sentence_mask": df_to_csr(sentences["doc_id"], sentences["sentence_idx"], n_rows=len(unique_doc_ids)),
#         }},
#         masks={"sentence": {"token": "token_mask"}, 
#                "doc": {"sentence_id": "sentence_mask"}}
#     )
#     return (
#         batcher, 
#         dict(docs=docs, sentences=sentences,tokens=tokens),
#         dict(token_id=unique_token_id, sentence_id=unique_sentence_ids, doc_id=unique_doc_ids)
#     )


# #@cached
# def preprocess(
#     dataset,
#     max_sentence_length,
#     bert_name,
#     vocabularies=None,
# ):
#     """
#     Parameters
#     ----------
#         dataset: Dataset
#         max_sentence_length: int
#             Max number of "words" as defined by the regex in regex_sentencize (so this is not the nb of wordpieces)
#         bert_name: str
#             bert path/name
#         ner_labels: list of str 
#             allowed ner labels (to be dropped or filtered)
#         unknown_labels: str
#             "drop" or "raise"
#         vocabularies: dict[str; np.ndarray or list]
        
#     Returns
#     -------
#     (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str; np.ndarray or list])
#         docs:      ('split', 'text', 'doc_id')
#         sentences: ('split', 'doc_id', 'sentence_idx', 'begin', 'end', 'text', 'sentence_id')
#         zones:     ('doc_id', 'sentence_id', 'zone_id', 'zone_idx')
#         mentions:  ('ner_label', 'doc_id', 'sentence_id', 'mention_id', 'depth', 'zone_id', 'text', 'mention_idx', 'begin', 'end', 'zone_mention_idx')
#         conflicts: ('doc_id', 'sentence_id', 'mention_id', 'mention_id_other', 'conflict_idx')
#         tokens:    ('split', 'token', 'sentence_id', 'token_id', 'token_idx', 'begin', 'end', 'doc_id', 'sentence_idx')
#         deltas:    ('doc_id', 'begin', 'end', 'delta')
#         vocs: vocabularies to be reused later for encoding more data or decoding predictions
#     """
#     # print("Dataset:", dataset)

#     # print("Transform texts...", end=" ")
#     transformed_docs, deltas = apply_substitutions(
#         dataset["docs"], *zip(
#             (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
#             (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
#             ("(?<=[a-zA-Z])(?=[0-9])", r" "),
#             ("(?<=[0-9])(?=[A-Za-z])", r" "),
#         ), apply_unidecode=True)
#     transformed_docs = transformed_docs.astype({"text": str})
#     # print("done")
    
#     # print("Splitting into sentences...", end=" ")
#     sentences = regex_sentencize(
#         transformed_docs, 
#         reg_split=r"((?:\s*\n\s*\n)+\s*|(?:(?<=[a-z0-9)]\n)|(?<=[a-z0-9)][ ](?:\.|\n))|(?<=[a-z0-9)][ ][ ](?:\.|\n)))\s*(?=[A-Z]))",
#         min_sentence_length=0, max_sentence_length=max_sentence_length,
#         # balance_parentheses=True, # default is True
#     )
    
#     # print("Tokenizing...", end=" ")
#     tokenizer = AutoTokenizer.from_pretrained(bert_name)
#     sentences["text"] = sentences["text"].str.lower()
#     tokens = huggingface_tokenize(sentences, tokenizer, doc_id_col="sentence_id")
#     # print("done")
    
#     # print("Computing vocabularies...")
#     [transformed_docs, sentences, tokens], vocs = normalize_vocabularies(
#         [transformed_docs, sentences, tokens], 
#         vocabularies={"split": ["train", "val", "test"]} if vocabularies is None else vocabularies,
#         train_vocabularies={"source": False, "text": False} if vocabularies is None else False,
#         verbose=False)
#     # print("done")
#     return transformed_docs, sentences, tokens, deltas, vocs

# from scipy.sparse import lil_matrix
# from nlstruct.exporters import brat

# export_to_brat = brat.export_to_brat

# def postprocess_batcher(batcher, dataset, prep, ids, vocabularies, recover_real_text=True, to_brat=False):
#     assert not to_brat or recover_real_text
# #     mentions_labels_df = pd.DataFrame({"labels": lil_matrix(batcher["mention", "label"]).rows, "mention_id": batcher["mention", "mention_id"]})
# #     mentions_labels_df["labels"] = mentions_labels_df["labels"].apply(lambda x: tuple(sorted(x)))
# #     mentions_labels_df = mentions_labels_df.explode("labels").reset_index(drop=True)
# #     mentions_labels_df['labels'] = np.asarray(vocabularies['label'])[mentions_labels_df['labels'].astype(int)].astype(str)
#     mentions_df = pd.DataFrame(dict(batcher["mention", ["sentence_id", "begin", "end", "ner_label", "mention_id"]]))
#     mentions_df[["doc_id", "sentence_id"]] = ids["sentence_id"].iloc[mentions_df["sentence_id"]].reset_index(drop=True)
#     mentions_df = mentions_df.merge(prep["sentences"][["sentence_id", "begin"] + (["text"] if not recover_real_text else [])], suffixes=('', '_sentence'), on="sentence_id")
#     mentions_df = mentions_df.merge(prep["tokens"][["sentence_id", "token_idx", "begin"]], left_on=("sentence_id", "begin"), right_on=("sentence_id", "token_idx"), suffixes=('', '_char'))
#     mentions_df = mentions_df.eval("end = end-1").merge(prep["tokens"][["sentence_id", "token_idx", "end"]], left_on=("sentence_id", "end"), right_on=("sentence_id", "token_idx"),
#                                                             suffixes=('', '_char'))
#     if recover_real_text:
#         mentions_df["begin"] = mentions_df["begin_char"] + mentions_df["begin_sentence"]
#         mentions_df["end"] = mentions_df["end_char"] + mentions_df["begin_sentence"]
#         mentions_df = mentions_df.merge(dataset["docs"][["doc_id", "text"]])
#         mentions_df = reverse_deltas(mentions_df, prep["deltas"], on="doc_id")
#     else:
#         mentions_df["begin"] = mentions_df["begin_char"]
#         mentions_df["end"] = mentions_df["end_char"]

#     mentions_df["text"] = mentions_df.apply(lambda x: x["text"][x["begin"]:x["end"]], axis=1)

#     try:
#         row_idx, col_idx = batcher["entity", "mention_mask"].nonzero()
#         entity_id = batcher['entity', 'entity_id'][row_idx]
#         mention_id = np.asarray(batcher['entity', 'mention_id'][row_idx, col_idx]).reshape(-1)
#         entities = pd.DataFrame({"entity_id": entity_id, "mention_id": mention_id})
#     except KeyError:
#         entities = pd.DataFrame({"entity_id": batcher["mention", "mention_id"], "mention_id": batcher["mention", "mention_id"]})
  
#     return mentions_df
# #     res = []
# #     if to_brat:
# #         res.append(export_to_brat(Dataset(docs=dataset["docs"], mentions=mentions_df)))
# #     return tuple(res)


# from os import path
# from pathlib import Path

# def preds_to_ann(
#     post_mentions, # mentions that are REVERSED (use postprocess_batcher first)
#     dataset,
#     vocs,
#     ann_path,
# ):
    
#     Path(ann_path).mkdir(parents=True, exist_ok=True)
    
#     for doc_id in dataset["docs"]["doc_id"].unique():
        
#         txt_str = dataset["docs"][dataset["docs"]["doc_id"].isin([doc_id])]["text"].iloc[0]
        
#         doc_mentions = post_mentions[post_mentions["doc_id"]==doc_id]
#         if len(doc_mentions):
            
#             ann_str = ''
            
#             i = 1
            
#             for begin, end, label_id, mention_text in doc_mentions[["begin", "end", "ner_label", "text"]].itertuples(index=False): 
#                 if "\n" in mention_text:
#                     line_begin = begin
#                     for line in mention_text.split("\n"):
#                         line_end = line_begin+len(line)
#                         ann_str += f"T{i}\t{vocs['ner_label'][label_id]} {line_begin} {line_end}\t{line}\n"
#                         line_begin += len(line) + 1
#                         i += 1
#                 else:
#                     ann_str += f"T{i}\t{vocs['ner_label'][label_id]} {begin} {end}\t{mention_text}\n"
#                     i += 1
            
#             with open(path.join(ann_path, f'{doc_id}.ann'), 'w') as f:
#                 f.write(ann_str)
                
#         with open(path.join(ann_path, f'{doc_id}.txt'), 'w') as f:
#             f.write(txt_str)

# def save_to_ann(texts, labels, ann_path):
#     import glob
#     files = glob.glob(ann_path + '/*')
#     for f in files:
#         os.remove(f)

#     # labels = [ll for l in labels for ll in l ]
#     # texts = [tt for t in texts for tt in t ]

#     for cpt_file, (text, label) in enumerate(zip(texts, labels)):
#         cpt_car = 0

#         mention_id = 0
#         mention_begin = 0
#         mention_end = 0
#         mention_label = ''
#         mention_text = ''

#         ann_str = ''
        
#         try:
#             text = [t.text for t in text]
#         except: pass

#         previous_label = ""

#         txt_str = ''

#         add_str = ""#"the concept is " 
#         add_str_len = len(add_str)

#         for i, (t, g) in enumerate(zip(text, label)):

#             current_label = g[2:].split(':')[-1]

#             start_new_mention = False

#             if g[0] == 'B':
#                 if mention_text != '':
#                     # add new mention
#                     mention_end = cpt_car - 1
#                     ann_str += f'T{mention_id}\t{mention_label} {mention_begin} {mention_end}\t{mention_text.strip(" ")}\n'
#                     mention_text = ''
#                     mention_id += 1

#                 mention_text = t + ' '
#                 txt_str += add_str
#                 cpt_car += add_str_len
#                 mention_begin = cpt_car
#                 mention_label = current_label

#             elif g[0] == 'I':
#                 mention_text += t + ' '
#                 if mention_label == '':
#                     mention_label = current_label
#             else:
#                 if mention_text != '':
#                     mention_end = cpt_car - 1
#                     ann_str += f'T{mention_id}\t{mention_label} {mention_begin} {mention_end}\t{mention_text.strip(" ")}\n'
#                     mention_text = ''
#                     mention_id += 1
#                 mention_text = ''

#             cpt_car += len(t) + 1
#             txt_str += t + ' '

#             if i==len(text)-1:
#                 if mention_text != '':
#                     mention_end = cpt_car - 1
#                     ann_str += f'T{mention_id}\t{mention_label} {mention_begin} {mention_end}\t{mention_text.strip(" ")}\n'
#                     mention_text = ''
#                     mention_id += 1

#         with open(ann_path + f'temp{cpt_file}.ann', 'w') as f:
#             f.write(ann_str)

#         with open(ann_path + f'temp{cpt_file}.txt', 'w') as f:
#             f.write(txt_str)

            
# class NERNet(torch.nn.Module):
#     def __init__(self,
#                  n_labels,
#                  hidden_dim,
#                  dropout,
#                  n_tokens=None,
#                  token_dim=None,
#                  embeddings=None,
#                  tag_scheme="bio",
#                  metric='linear',
#                  metric_fc_kwargs=None,
#                  ):
#         super().__init__()
#         if embeddings is not None:
#             self.embeddings = embeddings
#             if n_tokens is None or token_dim is None:
#                 if hasattr(embeddings, 'weight'):
#                     n_tokens, token_dim = embeddings.weight.shape
#                 else:
#                     n_tokens, token_dim = embeddings.embeddings.weight.shape
#         else:
#             self.embeddings = torch.nn.Embedding(n_tokens, token_dim) if n_tokens > 0 else None
#         assert token_dim is not None, "Provide token_dim or embeddings"
#         assert self.embeddings is not None

#         dim = (token_dim if n_tokens > 0 else 0)
#         self.dropout = torch.nn.Dropout(dropout)
#         if tag_scheme == "bio":
#             self.crf = BIODecoder(n_labels)
#         elif tag_scheme == "bioul":
#             self.crf = BIOULDecoder(n_labels)
#         else:
#             raise Exception()
#         if hidden_dim is None:
#             hidden_dim = dim
#         self.linear = torch.nn.Linear(dim, hidden_dim)
#         self.batch_norm = torch.nn.BatchNorm1d(dim)

#         n_tags = self.crf.num_tags
#         metric_fc_kwargs = metric_fc_kwargs if metric_fc_kwargs is not None else {}
#         if metric == "linear":
#             self.metric_fc = torch.nn.Linear(dim, n_tags)
#         elif metric == "cosine":
#             self.metric_fc = CosineSimilarity(dim, n_tags, rescale=rescale, **metric_fc_kwargs)
#         elif metric == "ema_cosine":
#             self.metric_fc = EMACosineSimilarity(dim, n_tags, rescale=rescale, **metric_fc_kwargs)
#         else:
#             raise Exception()
    
#     def extended_embeddings(self, tokens, mask, **kwargs):
#         # Default case here, size <= 512
#         # Small ugly check to see if self.embeddings is Bert-like, then we need to pass a mask
#         if hasattr(self.embeddings, 'encoder') or hasattr(self.embeddings, 'transformer'):
#             return self.embeddings(tokens, mask, **kwargs)[0]
#         else:
#             return self.embeddings(tokens)

#     def forward(self, tokens, mask, tag_embeds=None, return_embeddings=False):
#         # Embed the tokens
#         scores = None
#         # shape: n_batch * sequence * 768
#         embeds = self.extended_embeddings(tokens, mask, custom_embeds=tag_embeds)
#         state = embeds.masked_fill(~mask.unsqueeze(-1), 0)
#         state = torch.relu(self.linear(self.dropout(state)))# + state
#         state = self.batch_norm(state.view(-1, state.shape[-1])).view(state.shape)
#         scores = self.metric_fc(state)
#         return {
#             "scores": scores,
#             "embeddings": embeds if return_embeddings else None,
#         }           

def preprocess_train(
        dataset,
        bert_name,
        vocabularies,
        max_length=100,
        apply_unidecode=False,
        prepend_labels=[],
    ):

    mentions = dataset['mentions']
    mentions["text"] = mentions["text"].str.lower()
    mentions["mention_id"] = mentions["mention_id"].astype(str)

    subs = [
        (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
        (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
        ("(?<=[a-zA-Z])(?=[0-9])", r" "),
        ("(?<=[0-9])(?=[A-Za-z])", r" "),
        ("[ ]{2,}", " ")
    ]


    if subs is not None:
        mentions = apply_substitutions(mentions, *zip(*subs), apply_unidecode=apply_unidecode, return_deltas=False, with_tqdm=False)
    # mentions = pd.concat([
    #     mentions.query("source == 'real'")[["mention_id", "text", "label", "group", "source", "split"]],
    #     mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"])[["mention_id", "text", "label", "group", "source", "split"]],
    # ])

    # mentions = pd.concat([
    #     mentions.query("source == 'real'")[["mention_id", "text", "label", "group", "source", "split"]],
    #     mentions.query("source != 'real'").drop_duplicates(["label", "text", "group"])[["mention_id", "text", "label", "group", "source", "split"]],
    # ])

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    tokens = huggingface_tokenize(mentions, tokenizer, max_length=max_length, with_token_spans=False, unidecode_first=True, with_tqdm=False)
    
    # Construct charset from token->token_norm
    if vocabularies is None:
        vocabularies = {}
    [mentions, token], vocabularies = normalize_vocabularies(
        [mentions, tokens],
        vocabularies={
            "label": [*prepend_labels],
            **vocabularies},
        train_vocabularies={**{key: False for key in vocabularies}, "text": False, "label": True, }, verbose=1)

    ######################
    # CREATE THE BATCHER #
    ######################

    mention_ids = encode_ids([mentions, tokens], "mention_id")
    token_ids = encode_ids([tokens], ("mention_id", "token_id"))

    train_batcher = Batcher({
        "mention": {
            "mention_id": mentions["mention_id"],
            "token": df_to_csr(tokens["mention_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(mention_ids)),
            "token_mask": df_to_csr(tokens["mention_id"], tokens["token_idx"], n_rows=len(mention_ids)),
            "label": mentions["label"].cat.codes,
        }
    }, masks={"mention": {"token": "token_mask"}})

    return train_batcher, vocabularies, mention_ids

import numpy as np
import pandas as pd

from nlstruct.collections import Batcher
from nlstruct.scoring import compute_metrics, merge_pred_and_gold
from nlstruct.utils import evaluating, torch, torch_global as tg


def predict(batcher, classifier, batch_size=128, topk=1, save_embeds=False, with_tqdm=False, sum_embeds_by_label=False, return_loss=True, with_groups=True, device=None):
    if device is None:
        device = tg.device
    classifier.to(device)

    embeds = None
    ids = None
    top_labels = None
    top_probs = None
    new_weights = None
    index = None
    loss = None

    i = 0
    with evaluating(classifier):  # eval mode: no dropout, frozen batch norm, etc
        with torch.no_grad():  # no gradients -> faster
            with tqdm(batcher.dataloader(batch_size=batch_size, device=device, sparse_sort_on="token_mask"), disable=not with_tqdm, mininterval=10.) as bar:
                for batch in bar:
                    res = classifier.forward(
                        tokens=batch["token"],
                        mask=batch["token_mask"],
                        groups=None,#batch["group"] if with_groups else None,
                        return_scores=topk > 0,
                        return_embeds=save_embeds,
                    )
                    # Store predicted mention ids since we sort mentions by size
                    if ids is None:
                        ids = np.zeros((len(batcher),), dtype=int)

                    ids[i:i + len(batch)] = batch["mention_id"].cpu().numpy()

                    # If we want to predict new weights, directly sum mention embeddings on the device
                    if sum_embeds_by_label is not False:
                        if new_weights is None:
                            new_weights = torch.zeros(sum_embeds_by_label, res["embeds"].shape[1], device=device)
                        new_weights.index_add_(0, batch["label"], res["embeds"])

                    if return_loss:
                        if loss is None:
                            loss = np.zeros(len(batcher), dtype=float)
                        targets = batch["label"].clone()
                        targets[targets == -1] = 0

                        batch_losses = - res["scores"].log_softmax(dim=-1)[range(len(batch)), targets].cpu().numpy()
                        batch_losses[batch["label"].cpu().numpy() == -1] = -1
                        loss[i:i + len(batch)] = batch_losses
                        if index is None:
                            index = np.full(len(batcher), fill_value=-1, dtype=float)
                        index[i:i + len(batch)] = res["scores"].argsort(-1, descending=True).argsort(-1)[range(len(batch)), targets].cpu().numpy()

                    # If inference time, store topk prediction and associated probabilities
                    
                    if topk > 0:
                        if top_labels is None:
                            top_labels = np.zeros((len(batcher), topk), dtype=int)
                            top_probs = np.zeros((len(batcher), topk), dtype=float)
                        probs = res["scores"].softmax(-1)
                        best_probs, best_labels = probs.topk(dim=-1, k=topk)
                        top_labels[i:i + len(batch)] = best_labels.cpu().numpy()
                        top_probs[i:i + len(batch)] = best_probs.cpu().numpy()

                    # If we want to save embeddings
                    if save_embeds:
                        if embeds is None:
                            embeds = np.zeros((len(batcher), res["embeds"].shape[1]), dtype=float)
                        embeds[i:i + len(batch)] = res["embeds"].cpu().numpy()
                    i += len(batch)
    if return_loss:
        index[loss == -1] = np.inf
    pred = Batcher({
        "mention": {k: v for k, v in {
            "mention_id": ids,
            "embed": embeds,
            "label": top_labels,
            "prob": top_probs,
            "loss": loss,
            "index": index,
        }.items() if v is not None}}, check=False)

    if new_weights is not None:
        return pred, new_weights
    return pred


def compute_scores(pred_batcher, gold_batcher, prefix="", threshold=0):
    pred = pd.DataFrame({
        "mention_id": pred_batcher["mention_id"],
        "label": pred_batcher["label"][:, 0],
        "prob": pred_batcher['prob'][:, 0],
        "loss": pred_batcher['loss'],
    })

    loss = float(pred["loss"][(pred["loss"] >= 0) & (pred["loss"] < 10000)].mean())
    pred = pred[pred['prob'] >= threshold]
    gold = pd.DataFrame({
        "mention_id": gold_batcher["mention_id"],
        "label": gold_batcher["label"],
    })
    return {
        **compute_metrics(
            merge_pred_and_gold(pred, gold, on=["mention_id", "label"],
                                atom_pred_level=["mention_id"],
                                atom_gold_level=["mention_id"]),
            prefix=prefix),
        prefix + "loss": loss,
        prefix + "map": (1 / (pred_batcher["index"] + 1)).mean(),
    }

def preprocess_predict(
        dataset,
        bert_name,
        vocabularies,
        max_length=100,
        apply_unidecode=False,
        prepend_labels=[],
    ):

    mentions = dataset['docs']
    mentions["text"] = mentions["text"].str.lower()
    mentions["doc_id"] = mentions["doc_id"].astype(str)

    subs = [
        (r"(?<=[{}\\])(?![ ])".format(string.punctuation), r" "),
        (r"(?<![ ])(?=[{}\\])".format(string.punctuation), r" "),
        ("(?<=[a-zA-Z])(?=[0-9])", r" "),
        ("(?<=[0-9])(?=[A-Za-z])", r" "),
        ("[ ]{2,}", " ")
    ]


    if subs is not None:
        mentions = apply_substitutions(mentions, *zip(*subs), apply_unidecode=apply_unidecode, return_deltas=False, with_tqdm=False)

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    tokens = huggingface_tokenize(mentions, tokenizer, max_length=max_length, with_token_spans=False, unidecode_first=True, with_tqdm=False)
    
    # Construct charset from token->token_norm
    if vocabularies is None:
        vocabularies = {}
    [mentions, token], vocabularies = normalize_vocabularies(
        [mentions, tokens],
        vocabularies={
            **vocabularies},
        train_vocabularies={**{key: False for key in vocabularies}, "text": False}, verbose=1)

    ######################
    # CREATE THE BATCHER #
    ######################

    mention_ids = encode_ids([mentions, tokens], "doc_id")
    token_ids = encode_ids([tokens], ("doc_id", "token_id"))

    train_batcher = Batcher({
        "docs": {
            "mention_id": mentions["doc_id"],
            "token": df_to_csr(tokens["doc_id"], tokens["token_idx"], tokens["token"].cat.codes, n_rows=len(mention_ids)),
            "token_mask": df_to_csr(tokens["doc_id"], tokens["token_idx"], n_rows=len(mention_ids)),
        }
    }, masks={"docs": {"token": "token_mask"}})

    return train_batcher, vocabularies, mention_ids