import os
import random
from typing import List

import jittor as jt


def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Input:
        - tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        - token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    rows = []

    def _concat_1d(parts, dtype):
        valid = [p for p in parts if int(p.shape[0]) > 0]
        if not valid:
            return jt.zeros((0,), dtype=dtype)
        if len(valid) == 1:
            return valid[0]
        return jt.concat(valid, dim=0)

    for tok_list in token_span:
        row_masks = []
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            beg_pos = int(beg_pos)
            end_pos = int(end_pos)
            if beg_pos >= max_text_len:
                continue
            end_pos = min(end_pos, max_text_len - 1)
            if end_pos < beg_pos:
                continue

            if os.environ.get("SHILONG_DEBUG_ONLY_ONE_POS", None) == "TRUE":
                left = jt.zeros((beg_pos,), dtype=jt.float32)
                mid = jt.ones((1,), dtype=jt.float32)
                right = jt.zeros((max_text_len - beg_pos - 1,), dtype=jt.float32)
                row_masks = [_concat_1d([left, mid, right], jt.float32)]
                break
            span_len = end_pos - beg_pos + 1
            left = jt.zeros((beg_pos,), dtype=jt.float32)
            mid = jt.ones((span_len,), dtype=jt.float32)
            right = jt.zeros((max_text_len - end_pos - 1,), dtype=jt.float32)
            row_masks.append(_concat_1d([left, mid, right], jt.float32))

        if row_masks:
            row = jt.stack(row_masks, dim=0).max(dim=0)[0]
        else:
            row = jt.zeros((max_text_len,), dtype=jt.float32)
        rows.append(row.unsqueeze(0))

    if rows:
        positive_map = jt.concat(rows, dim=0)
    else:
        positive_map = jt.zeros((0, max_text_len), dtype=jt.float32)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """

    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan


def build_id2posspan_and_caption(category_dict: dict):
    """Build id2pos_span and caption from category_dict

    Args:
        category_dict (dict): category_dict
    """
    cat_list = [item["name"].lower() for item in category_dict]
    id2catname = {item["id"]: item["name"].lower() for item in category_dict}
    caption, cat2posspan = build_captions_and_token_span(cat_list, force_lowercase=True)
    id2posspan = {catid: cat2posspan[catname] for catid, catname in id2catname.items()}
    return id2posspan, caption
