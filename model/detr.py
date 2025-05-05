import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import resnet34
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


def get_spatial_position_embedding(pos_emb_dim, feat_map):
    assert pos_emb_dim % 4 == 0, ('Position embedding dimension '
                                  'must be divisible by 4')
    grid_size_h, grid_size_w = feat_map.shape[2], feat_map.shape[3]
    grid_h = torch.arange(grid_size_h,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid_w = torch.arange(grid_size_w,
                          dtype=torch.float32,
                          device=feat_map.device)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)

    # grid_h_positions -> (Number of grid cell tokens,)
    grid_h_positions = grid[0].reshape(-1)
    grid_w_positions = grid[1].reshape(-1)

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=pos_emb_dim // 4,
        dtype=torch.float32,
        device=feat_map.device) / (pos_emb_dim // 4))
    )

    grid_h_emb = grid_h_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_h_emb = torch.cat([
        torch.sin(grid_h_emb),
        torch.cos(grid_h_emb)
    ], dim=-1)
    # grid_h_emb -> (Number of grid cell tokens, pos_emb_dim // 2)

    grid_w_emb = grid_w_positions[:, None].repeat(1, pos_emb_dim // 4) / factor
    grid_w_emb = torch.cat([
        torch.sin(grid_w_emb),
        torch.cos(grid_w_emb)
    ], dim=-1)
    pos_emb = torch.cat([grid_h_emb, grid_w_emb], dim=-1)

    # pos_emb -> (Number of grid cell tokens, pos_emb_dim)
    return pos_emb


class TransformerEncoder(nn.Module):
    r"""
    Encoder for transformer of DETR.
    This has sequence of encoder layers.
    Each layer has the following modules:
        1. LayerNorm for Self Attention
        2. Self Attention
        3. LayerNorm for MLP
        4. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention Module for all encoder layers
        self.attns = nn.ModuleList(
                [
                    nn.MultiheadAttention(d_model, num_heads,
                                          dropout=self.dropout_prob,
                                          batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        # MLP Module for all encoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention for all encoder layers
        self.attn_norms = nn.ModuleList(
                [
                    nn.LayerNorm(d_model)
                    for _ in range(num_layers)
                ])

        # Norm for MLP for all encoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Self Attention for all encoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP for all encoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Norm for encoder output
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x, spatial_position_embedding):
        out = x
        attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            # Add spatial position embedding
            # to q,k for self attention
            q = in_attn + spatial_position_embedding
            k = in_attn + spatial_position_embedding
            out_attn, attn_weight = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            attn_weights.append(attn_weight)
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff

        # Output Normalization
        out = self.output_norm(out)
        return out, torch.stack(attn_weights)


class TransformerDecoder(nn.Module):
    r"""
        Decoder for transformer of DETR.
        This has sequence of decoder layers.
        Each layer has the following modules:
            1. LayerNorm for Self Attention
            2. Self Attention
            3. LayerNorm for Cross Attention on
                Encoder Outputs
            4. Cross Attention
            3. LayerNorm for MLP
            4. MLP
    """
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        # Self Attention module for all decoder layers
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # Cross Attention Module for all decoder layers
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, num_heads,
                                      dropout=self.dropout_prob,
                                      batch_first=True)
                for _ in range(num_layers)
            ])

        # MLP Module for all decoder layers
        self.ffs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, ff_inner_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_prob),
                    nn.Linear(ff_inner_dim, d_model),
                )
                for _ in range(num_layers)
            ])

        # Norm for Self Attention Module for all decoder layers
        self.attn_norms = nn.ModuleList(
                [
                    nn.LayerNorm(d_model)
                    for _ in range(num_layers)
                ])

        # Norm for Cross Attention Module for all decoder layers
        self.cross_attn_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Norm for MLP Module for all decoder layers
        self.ff_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(num_layers)
            ])

        # Dropout for Attention Module for all decoder layers
        self.attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for Cross Attention Module for all decoder layers
        self.cross_attn_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Dropout for MLP Module for all decoder layers
        self.ff_dropouts = nn.ModuleList(
            [
                nn.Dropout(self.dropout_prob)
                for _ in range(num_layers)
            ])

        # Shared Output norm for all decoder outputs
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_objects, encoder_output,
                query_embedding, spatial_position_embedding):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            # Norm, Self Attention, Dropout and Residual
            in_attn = self.attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            out_attn, _ = self.attns[i](
                query=q,
                key=k,
                value=in_attn
            )
            out_attn = self.attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, Cross Attention, Dropout and Residual
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_embedding
            k = encoder_output + spatial_position_embedding
            out_attn, decoder_cross_attn = self.cross_attns[i](
                query=q,
                key=k,
                value=encoder_output
            )
            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out + out_attn

            # Norm, MLP, Dropout and Residual
            in_ff = self.ff_norms[i](out)
            out_ff = self.ffs[i](in_ff)
            out_ff = self.ff_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))

        output = torch.stack(decoder_outputs)
        return output, torch.stack(decoder_cross_attn_weights)


class DETR(nn.Module):
    r"""
    DETR model class which instantiates all
    layers of DETR.
    A forward pass goes through the following layers:
        1. Backbone Call(currently frozen resnet 34)
        2. Backbone Featuremap Projection to d_model of transformer
        3. Encoder of Transformer
        4. Decoder of Transformer
        5. Class and BBox MLP
    """
    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_decoder_layers = config['decoder_layers']
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']
        self.nms_threshold = config['nms_threshold']
        self.bg_class_idx = bg_class_idx
        valid_bg_idx = (self.bg_class_idx == 0 or
                        self.bg_class_idx == (self.num_classes-1))
        assert valid_bg_idx, "Background can only be 0 or num_classes-1"

        self.backbone = nn.Sequential(*list(resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
            norm_layer=torchvision.ops.FrozenBatchNorm2d
        ).children())[:-2])

        if config['freeze_backbone']:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone_proj = nn.Conv2d(self.backbone_channels, self.d_model,
                                       kernel_size=1)
        self.encoder = TransformerEncoder(num_layers=config['encoder_layers'],
                                          num_heads=config['encoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'])
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))
        self.decoder = TransformerDecoder(num_layers=config['decoder_layers'],
                                          num_heads=config['decoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'])
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4),
        )

    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        # x -> (B, C, H, W)
        # default d_model - 256
        # default C - 3
        # default H,W - 640,640
        # default feat_h,feat_w - 20,20

        conv_out = self.backbone(x)  # (B, C_back, feat_h, feat_w)
        # default C_back -  512

        conv_out = self.backbone_proj(conv_out)  # (B, d_model, feat_h, feat_w)

        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        # spatial_pos_embed -> (feat_h * feat_w, d_model)

        conv_out = (conv_out.reshape(batch_size, d_model, feat_h * feat_w).
                    transpose(1, 2))
        # conv_out -> (B, feat_h*feat_w, d_model)

        # Encoder Call
        enc_output, enc_attn_weights = self.encoder(conv_out,  spatial_pos_embed)
        # enc_output -> (B, feat_h*feat_w, d_model)
        # enc_attn_weights -> (num_encoder_layers, B, feat_h*feat_w, feat_h*feat_w)

        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).
                                         repeat((batch_size, 1, 1)))
        # query_objects -> (B, num_queries, d_model)

        decoder_outputs = self.decoder(
            query_objects,
            enc_output,
            self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
            spatial_pos_embed)
        query_objects, decoder_attn_weights = decoder_outputs
        # query_objects -> (num_decoder_layers, B, num_queries, d_model)
        # decoder_attn_weights -> (num_decoder_layers, B, num_queries, feat_h*feat_w)

        cls_output = self.class_mlp(query_objects)
        # cls_output -> (num_decoder_layers, B, num_queries, num_classes)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        # bbox_output -> (num_decoder_layers, B, num_queries, 4)

        losses = defaultdict(list)
        detections = []
        detr_output = {}

        if self.training:
            num_decoder_layers = self.num_decoder_layers
            # Perform matching for each decoder layer
            for decoder_idx in range(num_decoder_layers):
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]
                with torch.no_grad():
                    # Concat all prediction boxes and class prob together
                    class_prob = cls_idx_output.reshape((-1, self.num_classes))
                    class_prob = class_prob.softmax(dim=-1)
                    # class_prob -> (B*num_queries, num_classes)

                    pred_boxes = bbox_idx_output.reshape((-1, 4))
                    # pred_boxes -> (B*num_queries, 4)

                    # Concat all target boxes and labels also together
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])
                    # len(target_labels) -> num_targets_for_entire_batch
                    # target_boxes -> (num_targets_for_entire_batch, 4)

                    # Classification Cost
                    cost_classification = -class_prob[:, target_labels]
                    # cost_cls -> (B*num_queries, num_targets_for_entire_batch)

                    # DETR predicts cx,cy,w,h , we need to covert to x1y1x2y2 for giou
                    # Don't need to convert targets as they are already in x1y1x2y2
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                        pred_boxes,
                        'cxcywh',
                        'xyxy')

                    cost_localization_l1 = torch.cdist(
                        pred_boxes_x1y1x2y2,
                        target_boxes,
                        p=1
                     )
                    # cost_l1 -> (B*num_queries, num_targets_for_entire_batch)

                    cost_localization_giou = -torchvision.ops.generalized_box_iou(
                        pred_boxes_x1y1x2y2,
                        target_boxes
                    )
                    # cost_giou->(B*num_queries,num_targets_for_entire_batch)
                    total_cost = (self.l1_cost_weight * cost_localization_l1
                                  + self.cls_cost_weight * cost_classification
                                  + self.giou_cost_weight * cost_localization_giou)

                    total_cost = total_cost.reshape(batch_size,self.num_queries,-1).cpu()
                    # total_cost -> (B, num_queries, num_targets_for_entire_batch)

                    num_targets_per_image = [len(target["labels"]) for target in targets]
                    total_cost_per_batch_image = total_cost.split(
                        num_targets_per_image,
                        dim=-1
                    )
                    # total_cost_per_batch_image[0]=(B,num_queries,num_targets_0th_image)
                    # total_cost_per_batch_image[i]=(B,num_queries,num_targets_ith_image)

                    match_indices = []
                    for batch_idx in range(batch_size):
                        batch_idx_assignments = linear_sum_assignment(
                            total_cost_per_batch_image[batch_idx][batch_idx]
                        )
                        batch_idx_pred, batch_idx_target = batch_idx_assignments
                        # len(batch_idx_assignment_pred) = num_targets_ith_image
                        match_indices.append((torch.as_tensor(batch_idx_pred,
                                                              dtype=torch.int64),
                                              torch.as_tensor(batch_idx_target,
                                                              dtype=torch.int64)))
                        # match_indices -> [
                        #   ([pred_box_a1, ...],[target_box_i1, ...]),
                        #   ([pred_box_a2, ...],[target_box_i2, ...]),
                        #   ... assignment pairs for ith batch image
                        #   ]
                # pred_batch_idxs are batch indexes for each assignment pair
                pred_batch_idxs = torch.cat([
                    torch.ones_like(pred_idx) * i
                    for i, (pred_idx, _) in enumerate(match_indices)
                ])
                # pred_batch_idxs -> (num_targets_for_entire_batch, )
                # pred_query_idx are prediction box indexes(out of num_queries)
                # for each assignment pair
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])
                # pred_query_idx -> (num_targets_for_entire_batch, )

                # For all assigned prediction boxes, get the target label
                valid_obj_target_cls = torch.cat([
                    target["labels"][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)
                ])
                # valid_obj_target_cls -> (num_targets_for_entire_batch, )

                # Initialize target class for all predicted boxes to be background class
                target_classes = torch.full(
                    cls_idx_output.shape[:2],
                    fill_value=self.bg_class_idx,
                    dtype=torch.int64,
                    device=cls_idx_output.device
                )
                # target_classes -> (B, num_queries)

                # For predicted boxes that were assigned to some target,
                # update their target label accordingly
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls

                # To ensure background class is not disproportionately attended by model
                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_cls_weight

                # Compute classification loss
                loss_cls = torch.nn.functional.cross_entropy(
                    cls_idx_output.reshape(-1, self.num_classes),
                    target_classes.reshape(-1),
                    cls_weights.to(cls_idx_output.device))

                # Get pred box coordinates for all matched pred boxes
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                # matched_pred_boxes -> (num_targets_for_entire_batch, 4)

                # Get target box coordinates for all matched target boxes
                target_boxes = torch.cat([
                    target['boxes'][target_obj_idx]
                    for target, (_, target_obj_idx) in zip(targets, match_indices)],
                    dim=0
                )
                # target_boxes -> (num_targets_for_entire_batch, 4)

                # Convert matched pred boxes to x1y1x2y2 format
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(
                    matched_pred_boxes,
                    'cxcywh',
                    'xyxy'
                )
                # Don't need to convert target boxes as they are in x1y1x2y2 format
                # Compute L1 Localization loss
                loss_bbox = torch.nn.functional.l1_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes,
                    reduction='none')
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]

                # Compute GIoU loss
                loss_giou = torchvision.ops.generalized_box_iou_loss(
                    matched_pred_boxes_x1y1x2y2,
                    target_boxes
                )
                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]

                losses['classification'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(
                    loss_bbox * self.l1_cost_weight
                    + loss_giou * self.giou_cost_weight
                )
            detr_output['loss'] = losses
        else:
            # For inference we are only interested in last layer outputs
            cls_output = cls_output[-1]
            bbox_output = bbox_output[-1]
            # cls_output -> (B, num_queries, num_classes)
            # bbox_output -> (B, num_queries, 4)

            prob = torch.nn.functional.softmax(cls_output, -1)

            # Get all query boxes and their best fg class as label
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels+1
            else:
                scores, labels = prob[..., :-1].max(-1)

            # convert to x1y1x2y2 format
            boxes = torchvision.ops.box_convert(bbox_output,
                                                'cxcywh',
                                                'xyxy')

            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]

                # Low score filtering
                keep_idxs = scores_idx >= score_thresh
                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]

                # NMS filtering
                if use_nms:
                    keep_idxs = torchvision.ops.batched_nms(
                        boxes_idx,
                        scores_idx,
                        labels_idx,
                        iou_threshold=self.nms_threshold)
                    scores_idx = scores_idx[keep_idxs]
                    boxes_idx = boxes_idx[keep_idxs]
                    labels_idx = labels_idx[keep_idxs]
                detections.append(
                    {
                        "boxes": boxes_idx,
                        "scores": scores_idx,
                        "labels": labels_idx
                        ,
                    }
                )

            detr_output['detections'] = detections
            detr_output['enc_attn'] = enc_attn_weights
            detr_output['dec_attn'] = decoder_attn_weights
        return detr_output

