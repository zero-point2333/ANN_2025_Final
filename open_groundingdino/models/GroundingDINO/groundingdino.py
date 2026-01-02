# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
import copy
from typing import List

from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast

from util import box_ops, get_tokenlizer
from util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)
from util.utils import div_trunc, get_phrases_from_posmap
from util.debug_tools import log_tensor, log_text
from util.visualizer import COCOVisualizer
from util.vl_utils import create_positive_map_from_span

from ..registry import MODULE_BUILD_FUNCS
from .backbone import build_backbone
from .bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, sigmoid_focal_loss

from .matcher import build_matcher

import jittor as jt
from jittor import nn


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias, 0)
        nn.init.xavier_uniform_(self.feat_map.weight)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.Sequential(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.Sequential(
                [
                    nn.Sequential(
                        nn.Conv(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.Sequential(box_embed_layerlist)
        self.class_embed = nn.Sequential(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def execute(self, samples: NestedTensor, targets: List = None, **kw):
        """The execute expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x num_classes]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, width, height). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
        # encoder texts

        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt")
        # Convert tokenizer tensors to Jittor Vars to keep Jittor ops consistent.
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in tokenized and not isinstance(tokenized[key], jt.Var):
                val = tokenized[key]
                if hasattr(val, "detach") and hasattr(val, "cpu") and hasattr(val, "numpy"):
                    val = val.detach().cpu().numpy()
                tokenized[key] = jt.array(val)
        one_hot_token = tokenized

        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768
        print("Bert Done!")

        last_hidden_state = bert_output["last_hidden_state"]
        if isinstance(last_hidden_state, jt.Var):
            encoded_text = self.feat_map(last_hidden_state)
        else:
            if hasattr(last_hidden_state, "detach") and hasattr(last_hidden_state, "cpu") and hasattr(last_hidden_state, "numpy"):
                last_hidden_state = last_hidden_state.detach().cpu().numpy()
            encoded_text = self.feat_map(jt.array(last_hidden_state))
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask
        
        print("Feat Map Done!")

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }


        if isinstance(samples, (list, jt.Var)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        assert isinstance(poss, list)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            assert isinstance(feat, NestedTensor)
            src, mask = feat.decompose()
            if not isinstance(src, jt.Var):
                src = jt.array(src)
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](jt.array(features[-1].tensors))
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = nn.interpolate(jt.array(m[None], dtype=jt.float32), size=src.shape[-2:]).bool()[0]
                pos_l = jt.type_as(self.backbone[1](NestedTensor(src, mask)), src)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)
            
        print("Backbone Done!")

        input_query_bbox = input_query_label = attn_mask = dn_meta = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        print("Transformer Done!")
        
        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            assert isinstance(layer_outputs_unsig, jt.Var)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = jt.stack(outputs_coord_list)


        outputs_class = jt.stack(
            [
                layer_cls_embed(layer_hs, text_dict)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        # Used to calculate losses
        bs, len_td = text_dict['text_token_mask'].shape
        # [修改] 使用 concat 替代 setitem
        if len_td < self.max_text_len:
            pad_len = self.max_text_len - len_td
            # 注意: text_token_mask 中 1 是有效，0 是 padding (假设)
            # 或者 0 是 padding? 根据原代码 zeros 填充，说明 padding 部分是 False/0
            pad = jt.zeros((bs, pad_len), dtype=text_dict['text_token_mask'].dtype)
            # 确保类型一致
            if text_dict['text_token_mask'].dtype != pad.dtype:
                pad = pad.cast(text_dict['text_token_mask'].dtype)
            
            out['text_mask'] = jt.concat([text_dict['text_token_mask'], pad], dim=1)
        else:
            out['text_mask'] = text_dict['text_token_mask'][:, :self.max_text_len]
            
        # 确保转为 bool
        out['text_mask'] = out['text_mask'].bool()

        # for intermediate outputs
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord_list)
        out['token']=one_hot_token
        # # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            print("Transformer.enc_out_class_embed Start!")
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], text_dict)
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_boxes': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_boxes': init_box_proposal}

        # outputs['pred_logits'].shape
        # torch.Size([4, 900, 256])

        # outputs['pred_boxes'].shape
        # torch.Size([4, 900, 4])

        # outputs['text_mask'].shape
        # torch.Size([256])

        # outputs['text_mask']

        # outputs['aux_outputs'][0].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])

        # outputs['aux_outputs'][img_idx]

        # outputs['token']
        # <class 'transformers.tokenization_utils_base.BatchEncoding'>

        # outputs['interm_outputs'].keys()
        # dict_keys(['pred_logits', 'pred_boxes', 'one_hot', 'text_mask'])


        # outputs['interm_outputs_for_matching_pre'].keys()
        # dict_keys(['pred_logits', 'pred_boxes'])

        # outputs['one_hot'].shape
        # torch.Size([4, 900, 256])
        
        # ================= [新增] Jittor 数值保护 =================
        # 强制截断 logits，防止 Sigmoid 后出现绝对的 0 或 1，导致 Loss 计算 log(0)
        # 同时防止 inf 传入 Matcher 导致索引越界
        safe_min = -100.0
        safe_max = 100.0
        
        # 1. 修复主输出
        if isinstance(out["pred_logits"], jt.Var):
            out["pred_logits"] = jt.clamp(out["pred_logits"], safe_min, safe_max)
            # 如果出现 nan，将其置为 0 (背景类)
            if jt.any(jt.isnan(out["pred_logits"])).item():
                out["pred_logits"] = jt.where(
                    jt.isnan(out["pred_logits"]), 
                    jt.zeros_like(out["pred_logits"]), 
                    out["pred_logits"]
                )

        # 2. 修复辅助输出 (Aux Loss)
        if "aux_outputs" in out:
            for aux in out["aux_outputs"]:
                if "pred_logits" in aux:
                    aux["pred_logits"] = jt.clamp(aux["pred_logits"], safe_min, safe_max)

        # 3. 修复中间输出 (Two-stage)
        if "interm_outputs" in out and "pred_logits" in out["interm_outputs"]:
            out["interm_outputs"]["pred_logits"] = jt.clamp(out["interm_outputs"]["pred_logits"], safe_min, safe_max)
        
        if "interm_outputs_for_matching_pre" in out and "pred_logits" in out["interm_outputs_for_matching_pre"]:
            out["interm_outputs_for_matching_pre"]["pred_logits"] = jt.clamp(out["interm_outputs_for_matching_pre"]["pred_logits"], safe_min, safe_max)
        # ========================================================

        return out

    def _set_aux_loss(self, outputs_class, outputs_coord):
        # Build auxiliary outputs without iterating directly over jt.Var.
        aux_outputs = []
        num_layers = int(outputs_class.shape[0])
        for i in range(num_layers - 1):
            aux_outputs.append(
                {"pred_logits": outputs_class[i], "pred_boxes": outputs_coord[i]}
            )
        return aux_outputs




class SetCriterion(nn.Module):
    def __init__(self, matcher, weight_dict, focal_alpha,focal_gamma, losses):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma= focal_gamma

    @jt.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        tgt_lengths = jt.array([len(v["labels"]) for v in targets])
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        assert isinstance(pred_logits, jt.Var)
        card_pred = (jt.argmax(pred_logits, -1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = nn.l1_loss(card_pred.float(), tgt_lengths.float()) # here to go
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = jt.concat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # Use elementwise L1 to keep [num_boxes, 4] for downstream slicing.
        loss_bbox = (src_boxes - target_boxes).abs()
        if loss_bbox.ndim == 1:
            loss_bbox = loss_bbox.reshape(1, -1)

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        src_xyxy = box_ops.box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_ops.box_cxcywh_to_xyxy(target_boxes)
        if src_xyxy.numel() == 0 or tgt_xyxy.numel() == 0:
            losses['loss_giou'] = jt.array(0.0)
        else:
            # ================= [Fix Start] =================
            # 移除 .item() 调用，改用无条件的 Graph 操作
            # 原代码:
            # invalid = jt.isnan(src_xyxy) | jt.isinf(src_xyxy)
            # if invalid.any().item(): # Error: no lock?
            #     src_xyxy = jt.where(invalid, jt.zeros_like(src_xyxy), src_xyxy)
            
            # 新代码: 直接执行 where，不打断计算图
            invalid_src = jt.isnan(src_xyxy) | jt.isinf(src_xyxy)
            src_xyxy = jt.where(invalid_src, jt.zeros_like(src_xyxy), src_xyxy)

            # 同理修改 target 的处理
            # 原代码:
            # invalid = jt.isnan(tgt_xyxy) | jt.isinf(tgt_xyxy)
            # if invalid.any().item():
            #     tgt_xyxy = jt.where(invalid, jt.zeros_like(tgt_xyxy), tgt_xyxy)

            # 新代码:
            invalid_tgt = jt.isnan(tgt_xyxy) | jt.isinf(tgt_xyxy)
            tgt_xyxy = jt.where(invalid_tgt, jt.zeros_like(tgt_xyxy), tgt_xyxy)
            # ================= [Fix End] =================

            src_xyxy = jt.concat(
                [jt.minimum(src_xyxy[:, :2], src_xyxy[:, 2:]),
                 jt.maximum(src_xyxy[:, :2], src_xyxy[:, 2:])],
                dim=1,
            )
            tgt_xyxy = jt.concat(
                [jt.minimum(tgt_xyxy[:, :2], tgt_xyxy[:, 2:]),
                 jt.maximum(tgt_xyxy[:, :2], tgt_xyxy[:, 2:])],
                dim=1,
            )
            loss_giou = 1 - jt.diag(box_ops.generalized_box_iou(src_xyxy, tgt_xyxy))
            losses['loss_giou'] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with jt.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses


    def token_sigmoid_binary_focal_loss(self, outputs, targets, indices, num_boxes):
        pred_logits=outputs['pred_logits']
        new_targets=outputs['one_hot']
        text_mask=outputs['text_mask']

        assert (new_targets.dim() == 3)
        assert (pred_logits.dim() == 3)  # batch x from x to
        
        bs, n, _ = pred_logits.shape
        alpha=self.focal_alpha
        gamma=self.focal_gamma
        mask_f = None
        if text_mask is not None:
            # ODVG: each sample has different mask
            assert isinstance(text_mask, jt.Var) 
            text_mask = text_mask.repeat(1, pred_logits.size(1)).view(outputs['text_mask'].shape[0],-1,outputs['text_mask'].shape[1]) # not highlighted, but doc says it has
            mask_f = text_mask.astype(pred_logits.dtype)

        new_targets = new_targets.float()
        p = jt.sigmoid(pred_logits)
        ce_loss = nn.binary_cross_entropy_with_logits(pred_logits, new_targets)
        p_t = p * new_targets + (1 - p) * (1 - new_targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * new_targets + (1 - alpha) * (1 - new_targets)
            loss = alpha_t * loss
        if mask_f is not None:
            loss = loss * mask_f

        total_num_pos=0
        for batch_indices in indices:
            total_num_pos += len(batch_indices[0])
        num_pos_avg_per_gpu = max(total_num_pos , 1.0)
        assert isinstance(loss, jt.Var)
        loss = loss.sum()/num_pos_avg_per_gpu
        
        losses = {'loss_ce': loss}
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = jt.concat([jt.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = jt.concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = jt.concat([jt.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = jt.concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.token_sigmoid_binary_focal_loss,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def execute(self, outputs, targets, cat_list, caption, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        print("Criterion Start")
        one_hot = jt.zeros(outputs['pred_logits'].size(), dtype=jt.int64) # torch.Size([bs, 900, 256])
        token = outputs['token'] 
        
        label_map_list = []
        indices = []
        for j in range(len(cat_list)): # bs
            label_map=[]
            for i in range(len(cat_list[j])):
                label_id = jt.array([i])
                per_label = create_positive_map(token[j], label_id, cat_list[j], caption[j])
                label_map.append(per_label)
            label_map = jt.stack(label_map,dim=0).squeeze(1)
            label_map_list.append(label_map)
        log_text(f"cat_list: {cat_list}")
        for j in range(len(cat_list)): # bs
            for_match = {
                "pred_logits" : outputs['pred_logits'][j].unsqueeze(0),
                "pred_boxes" : outputs['pred_boxes'][j].unsqueeze(0)
            }
            inds = self.matcher(for_match, [targets[j]], label_map_list[j])
            indices.extend(inds)
        # indices : A list of size batch_size, containing tuples of (index_i, index_j) where:
        # - index_i is the indices of the selected predictions (in order)
        # - index_j is the indices of the corresponding selected targets (in order)
        
        
        print("Criterion Main Matcher Done!")
        # ==== OutOfIndex Code ====
        # for i in range(len(indices)):
        #     tgt_ids[i]=tgt_ids[i][indices[i][1]]
        #     one_hot[i,indices[i][0]] = jt.array(label_map_list[i][tgt_ids[i]], dtype=jt.int64)
        # outputs['one_hot'] = one_hot
        # ==== OutOfIndex Code ====
        # ==== Change Code ====
        _, num_queries, dim = one_hot.shape # bs, 900, 256; stack on bs dim
        log_text("Criterion Main Matcher After Line 1")
        
        tgt_ids_ref = [v["labels"] for v in targets]
        log_text("Criterion Main Matcher After Line 2")
            
        one_hot_list = []
        for i in range(len(indices)):
            log_text("Criterion Main Matcher After Line 3")
            # 1. 创建当前样本的全 0 mask [900, 256]
            current_one_hot = jt.zeros((num_queries, dim), dtype=jt.int64)
            log_text("Criterion Main Matcher After Line 4")
            
            # 2. 获取匹配的 targets
            pred_idx = indices[i][0]
            log_text("Criterion Main Matcher After Line 5")
            matched_tgt_idx = indices[i][1]
            log_text("Criterion Main Matcher After Line 6")
            
            # [关键修复] 只有当存在匹配时才执行 scatter
            if pred_idx.numel() > 0: # jittor has numel() function, one can check in cmd line
                log_text("Criterion Main Matcher After Line 7")
                current_tgt_ids = tgt_ids_ref[i][matched_tgt_idx]
                log_text("Criterion Main Matcher After Line 8")
                
                # 获取 Token Mask [Matched, 256]
                matched_maps = label_map_list[i][current_tgt_ids]
                log_text("Criterion Main Matcher After Line 9")
                
                # 构造 scatter 索引 [Matched, 256]
                index_matrix = pred_idx.unsqueeze(1).repeat(1, dim)
                log_text("Criterion Main Matcher After Line 10")
                
                # Scatter Update
                current_one_hot = current_one_hot.scatter(0, index_matrix, matched_maps.cast(jt.int64)) # although no highlight, it is valid to use jt.Var.scatter()
                log_text("Criterion Main Matcher After Line 11")
            
            one_hot_list.append(current_one_hot)
            log_text("Criterion Main Matcher After Line 12")
            
        one_hot = jt.stack(one_hot_list, dim=0)
        log_text("Criterion Main Matcher After Line 13")
        outputs['one_hot'] = one_hot
        log_text("Criterion Main Matcher After Line 14")
        # ==== Change Code ====
        
        if return_indices:
            log_text("Criterion Main Matcher After Line 15")
            indices0_copy = indices
            log_text("Criterion Main Matcher After Line 16")
            indices_list = []
            log_text("Criterion Main Matcher After Line 17")

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_list = [len(t["labels"]) for t in targets]
        log_text("Criterion Main Matcher After Line 18")
        num_boxes = sum(num_boxes_list)
        log_text("Criterion Main Matcher After Line 19")
        num_boxes = jt.array([num_boxes], dtype=jt.float32)
        log_text("Criterion Main Matcher After Line 20")
        if is_dist_avail_and_initialized():
            print("unexpected line executing in SetCriterion.execute()")
            # torch.distributed.all_reduce(num_boxes)
        num_boxes = jt.clamp(num_boxes / get_world_size(), min_v=1).item()
        log_text("Criterion Main Matcher After Line 21")

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            log_text("Criterion Main Matcher After Line 22")
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes)) # lock error, not runtime error
            log_text("Criterion Main Matcher After Line 23")

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        log_text(f"outputs: {outputs.keys()}")
        if 'aux_outputs' in outputs:
            print("Dealing Aux Outputs!")
            for idx, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = []
                for j in range(len(cat_list)): # bs
                    aux_output_single = {
                        'pred_logits' : aux_outputs['pred_logits'][j].unsqueeze(0),
                        'pred_boxes': aux_outputs['pred_boxes'][j].unsqueeze(0)
                    }
                    inds = self.matcher(aux_output_single, [targets[j]], label_map_list[j])
                    print("Aux Outpus Matcher Done!")
                    indices.extend(inds)
                # ==== OutOfIndex Code ====
                # one_hot_aux = jt.zeros(outputs['pred_logits'].size(),dtype=jt.int64)
                # tgt_ids = [v["labels"] for v in targets]
                # for i in range(len(indices)):
                #     tgt_ids[i]=tgt_ids[i][indices[i][1]]
                #     one_hot_aux[i,indices[i][0]] = jt.array(label_map_list[i][tgt_ids[i]], dtype=jt.int64)
                # aux_outputs['one_hot'] = one_hot_aux
                # ==== OutOfIndex Code ====
                # ==== Change Code ====
                # [修改] 使用 stack + scatter 替代 inplace setitem
                _, num_queries, dim = outputs['pred_logits'].shape
                one_hot_aux_list = []
                tgt_ids_ref = [v["labels"] for v in targets]
                
                for i in range(len(indices)):
                    current_one_hot = jt.zeros((num_queries, dim), dtype=jt.int64)
                    
                    pred_idx = indices[i][0]
                    matched_tgt_idx = indices[i][1]
                    
                    # [关键修复] 非空检查
                    if pred_idx.numel() > 0:
                        current_tgt_ids = tgt_ids_ref[i][matched_tgt_idx]
                        matched_maps = label_map_list[i][current_tgt_ids]
                        
                        index_matrix = pred_idx.unsqueeze(1).repeat(1, dim)
                        current_one_hot = current_one_hot.scatter(0, index_matrix, matched_maps.cast(jt.int64))
                    one_hot_aux_list.append(current_one_hot)
                
                aux_outputs['one_hot'] = jt.stack(one_hot_aux_list, dim=0)
                # ==== Change Code ====
                
                aux_outputs['text_mask'] = outputs['text_mask']
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)                
                    l_dict = {k + f'_{idx}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if 'interm_outputs' in outputs:
            print("Dealing Interm Outputs!")
            interm_outputs = outputs['interm_outputs']
            
            # [修复开始]：获取 Logits 和 Boxes，并修正维度
            pred_logits = interm_outputs['pred_logits']
            pred_boxes = interm_outputs.get("pred_boxes")
            
            # 1. 维度对齐检查与修复
            # 如果 Boxes 的 Batch 为 1，但 Logits 的 Batch > 1，说明 Boxes 需要广播
            if pred_boxes.shape[0] == 1 and pred_logits.shape[0] > 1:
                log_text(f"[Fix] Expanding interm_pred_boxes from {pred_boxes.shape} to match logits {pred_logits.shape}")
                pred_boxes = pred_boxes.repeat(pred_logits.shape[0], 1, 1)
                # 这一步非常重要：必须更新字典中的值，因为后续 self.get_loss 会直接从 dict 里取值计算 Loss
                interm_outputs['pred_boxes'] = pred_boxes 

            # 2. 数值稳定性保护 (保持你原有的逻辑，稍作整理)
            log_text("interm_outputs stats", force=True)
            log_tensor("interm_outputs.pred_logits", pred_logits, force=True)
            log_tensor("interm_outputs.pred_boxes", pred_boxes, force=True)

            if isinstance(pred_boxes, jt.Var):
                invalid = jt.isnan(pred_boxes) | jt.isinf(pred_boxes)
                if jt.any(invalid).item():
                    pred_boxes = jt.where(invalid, jt.zeros_like(pred_boxes), pred_boxes)
                pred_boxes = jt.clamp(pred_boxes, min_v=0.0, max_v=1.0)
                interm_outputs["pred_boxes"] = pred_boxes # update back

            if isinstance(pred_logits, jt.Var):
                invalid = jt.isnan(pred_logits) | jt.isinf(pred_logits)
                if jt.any(invalid).item():
                    pred_logits = jt.where(invalid, jt.zeros_like(pred_logits), pred_logits)
                pred_logits = jt.clamp(pred_logits, min_v=-20.0, max_v=20.0)
                interm_outputs["pred_logits"] = pred_logits # update back

            # 3. 匹配循环 (逻辑简化，因为维度现在已经对其了)
            indices = []
            # 此时 pred_logits 和 pred_boxes 的 batch 维度应该一致了
            for j in range(len(cat_list)): # bs
                interm_output_single = {
                    'pred_logits' : interm_outputs['pred_logits'][j].unsqueeze(0),
                    'pred_boxes': interm_outputs['pred_boxes'][j].unsqueeze(0)
                }
                inds = self.matcher(interm_output_single, [targets[j]], label_map_list[j])
                print("Interm Outputs Matcher Done!")
                indices.extend(inds)

            # ==== Change Code ====
            _, num_queries, dim = outputs['pred_logits'].shape
            if 'pred_logits' in interm_outputs:
                 _, num_queries, _ = interm_outputs['pred_logits'].shape
            one_hot_interm_list = []
            tgt_ids_ref = []
            for v in targets:
                labels = v["labels"]
                if isinstance(labels, jt.Var):
                    tgt_ids_ref.append(labels)
                else:
                    if hasattr(labels, "detach") and hasattr(labels, "cpu") and hasattr(labels, "numpy"):
                        labels = labels.detach().cpu().numpy()
                    tgt_ids_ref.append(jt.array(labels))
                    
            for i in range(len(indices)):
                current_one_hot = jt.zeros((num_queries, dim), dtype=jt.int64)
                pred_idx = indices[i][0]
                matched_tgt_idx = indices[i][1]
                
                # [关键修复] 非空检查
                if pred_idx.numel() > 0:
                    current_tgt_ids = tgt_ids_ref[i][matched_tgt_idx]
                    matched_maps = label_map_list[i][current_tgt_ids]
                    
                    index_matrix = pred_idx.unsqueeze(1).repeat(1, dim)
                    current_one_hot = current_one_hot.scatter(0, index_matrix, matched_maps.cast(jt.int64))
                one_hot_interm_list.append(current_one_hot)

            interm_outputs['one_hot'] = jt.stack(one_hot_interm_list, dim=0)
            # ==== Change Code ====
            
            interm_outputs['text_mask'] = outputs['text_mask']
            if return_indices:
                indices_list.append(indices)
                
            for loss in self.losses:
                kwargs = {}
                # get_loss 内部会使用 interm_outputs['pred_boxes']，现在它是扩展后的 correct shape
                l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)
            # [修复结束]

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100,text_encoder_type='text_encoder_type', nms_iou_threshold=-1,use_coco_eval=False,args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item['name'] for item in category_dict]
        else:
            cat_list=args.label_list
        caption = " . ".join(cat_list) + ' .'
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = jt.arange(len(cat_list))
        pos_map=create_positive_map(tokenized,label_list,cat_list,caption)
        # build a mapping from label_id to pos_map
        if args.use_coco_eval:
            id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
            new_pos_map = jt.zeros((91, 256))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map


        self.nms_iou_threshold=nms_iou_threshold
        self.positive_map = pos_map

    @jt.no_grad()
    def execute(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        assert isinstance(target_sizes, jt.Var)
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert isinstance(out_logits, jt.Var)
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map
        for label_ind in range(len(pos_maps)):
            assert isinstance(pos_maps[label_ind], jt.Var)
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind]=pos_maps[label_ind]/pos_maps[label_ind].sum()

        prob_to_label = nn.matmul_transpose(prob_to_token, pos_maps)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = jt.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = div_trunc(topk_indexes, prob.shape[2])
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        assert isinstance(topk_boxes, jt.Var)
        boxes = jt.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = jt.unbind(target_sizes, 1)
        scale_fct = jt.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [jt.nms((x1, y1, x2, y2, s), thresh=self.nms_iou_threshold) for (x1, y1, x2, y2), s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


@MODULE_BUILD_FUNCS.registe_with_name(module_name="groundingdino")
def build_groundingdino(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
    )



    matcher = build_matcher(args)

    # prepare weight dict
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    

    clean_weight_dict = copy.deepcopy(weight_dict)

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in clean_weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.two_stage_type != 'no':
        interm_weight_dict = {}
        try:
            no_interm_box_loss = args.no_interm_box_loss
        except:
            no_interm_box_loss = False
        _coeff_weight_dict = {
            'loss_ce': 1.0,
            'loss_bbox': 1.0 if not no_interm_box_loss else 0.0,
            'loss_giou': 1.0 if not no_interm_box_loss else 0.0,
        }
        try:
            interm_loss_coef = args.interm_loss_coef
        except:
            interm_loss_coef = 1.0
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items()})
        weight_dict.update(interm_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality']
    losses = ['labels', 'boxes']

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses
                             )
    criterion
    postprocessors = {'bbox': PostProcess(num_select=args.num_select  , text_encoder_type=args.text_encoder_type,nms_iou_threshold=args.nms_iou_threshold,args=args)}

    return model, criterion, postprocessors

def create_positive_map(tokenized, tokens_positive,cat_list,caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = jt.zeros((len(tokens_positive), 256), dtype=jt.float32)
    max_text_len = int(positive_map.shape[1])
    for j, label in enumerate(tokens_positive):
        if isinstance(label, jt.Var):
            if label.ndim != 0:
                label = label.reshape(-1)[0]
            label = label.numpy().item()
        elif hasattr(label, "item"):
            label = label.item()
        label = int(label)
        if label < 0 or label >= len(cat_list):
            continue
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        # except Exception as e:
        #     print("beg:", beg, "end:", end)
        #     print("token_positive:", tokens_positive)
        #     # print("beg_pos:", beg_pos, "end_pos:", end_pos)
        #     raise e
        # if beg_pos is None:
        #     try:
        #         beg_pos = tokenized.char_to_token(beg + 1)
        #         if beg_pos is None:
        #             beg_pos = tokenized.char_to_token(beg + 2)
        #     except:
        #         beg_pos = None
        # if end_pos is None:
        #     try:
        #         end_pos = tokenized.char_to_token(end - 2)
        #         if end_pos is None:
        #             end_pos = tokenized.char_to_token(end - 3)
        #     except:
        #         end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        if beg_pos >= max_text_len:
            continue
        if end_pos >= max_text_len:
            end_pos = max_text_len - 1
        # assert beg_pos is not None and end_pos is not None
        positive_map[j,beg_pos: end_pos + 1].fill_(1)
    return positive_map 
