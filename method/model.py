import torch
import torch.nn.functional as F
from method.model_components import BertAttention, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, GlobalPool1D, GlobalConv1D
from easydict import EasyDict as edict
import copy 
import torch.nn as nn 
from utils.model_utils import generate_gauss_weight, get_center_based_props, get_props_from_indices, cos_sim
import copy 
from utils.model_utils import cos_sim


class FAWL(nn.Module):
    def __init__(self, config):
        super(FAWL, self).__init__()
        self.config = config

        self.query_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_desc_l,hidden_size=config.hidden_size, dropout=config.input_drop)
        self.clip_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,hidden_size=config.hidden_size, dropout=config.input_drop)
        self.frame_pos_embed = TrainablePositionalEncoding(max_position_embeddings=config.max_ctx_l,hidden_size=config.hidden_size, dropout=config.input_drop)
        #
        self.query_input_proj = LinearLayer(config.query_input_size, config.hidden_size, layer_norm=True,dropout=config.input_drop, relu=True)
        self.query_encoder = BertAttention(edict(hidden_size=config.hidden_size, intermediate_size=config.hidden_size,hidden_dropout_prob=config.drop, num_attention_heads=config.n_heads,attention_probs_dropout_prob=config.drop))

        self.clip_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,dropout=config.input_drop, relu=True)
        self.clip_encoder = copy.deepcopy(self.query_encoder)

        self.frame_input_proj = LinearLayer(config.visual_input_size, config.hidden_size, layer_norm=True,dropout=config.input_drop, relu=True)
        self.modular_vector_mapping = nn.Linear(config.hidden_size, out_features=1, bias=False)
        self.global_pool1d = GlobalPool1D(config.pooling_ksize, config.pooling_stride)
        self.global_conv1d = GlobalConv1D(config.visual_input_size, config.conv_ksize, config.conv_stride, out_channel=1)


        self.saliency_proj = LinearLayer(config.hidden_size, config.hidden_size, layer_norm=True,dropout=config.input_drop, relu=True)

        self.left_proj = LinearLayer(config.hidden_size, 1, layer_norm=True,dropout=config.input_drop, relu=False)
        self.right_proj = LinearLayer(config.hidden_size, 1, layer_norm=True,dropout=config.input_drop, relu=False)

        self.num_gauss_center = config.num_gauss_center # 32
        self.num_gauss_width = config.num_gauss_width # 10
        self.sigma = config.sigma # 9
        self.width_lower_bound = config.width_lower_bound
        self.width_upper_bound = config.width_upper_bound
        self.map_size = config.map_size
        self.window_size_inc = config.window_size_inc
        self.num_props_per_center = config.num_props
        self.gamma = config.gamma
        self.gauss_center, self.gauss_width = get_center_based_props(self.num_gauss_center, self.num_gauss_width, self.width_lower_bound, self.width_upper_bound)
        self.reset_parameters()
    
    def compute_visual_similarities(self, visual_features, glance_res):
        gaze_props_feature = visual_features['gaze_props_feature'].detach() # [Nv, Np, E]
        E = gaze_props_feature.shape[-1]
        glance_props_feature = visual_features['glance_props_feature'].detach() # [Nv, Np, E]

        orig_props_feature = visual_features['orig_props_feature'] # [Nv, Np, Ev]
        glance_key_prop_indices = glance_res['glance_key_prop_indices'].detach() # [Nt, Nv], Nt=Nv
        matched_indices = torch.diag(glance_key_prop_indices) # [Nv]
        matched_indices = matched_indices[:, None, None].repeat(1, 1, E) # [Nv, 1, E]

        sel_gaze_prop_feature = torch.gather(gaze_props_feature, dim=1, index=matched_indices).squeeze(1) # [Nv, E]
        sel_glance_prop_feature = torch.gather(glance_props_feature, dim=1, index=matched_indices).squeeze(1)
        sel_orig_prop_feature = torch.gather(orig_props_feature, dim=1, index=matched_indices).squeeze(1)
        
        proj_gaze_sims = cos_sim(sel_gaze_prop_feature, sel_gaze_prop_feature)
        proj_glance_sims = cos_sim(sel_glance_prop_feature, sel_glance_prop_feature)
        orig_sims = cos_sim(sel_orig_prop_feature, sel_orig_prop_feature)

        sims = {
            "proj_video_glance_sims": proj_glance_sims,
            "proj_video_gaze_sims": proj_gaze_sims,
            "orig_video_sims": orig_sims
        }
        return sims 

    def forward(self, batch, epoch): 
        """
            Inputs: (Dict)
                clip_video_feat (NOT USED)
                frame_video_feat: [B, Lf, Ev]
                frame_video_mask: [B, Lf]  # 0 for masked
                query_feat: [B, Lt, Et]
                query_mask: [B, Lt]        # 0 for masked
            Outputs: (Dict)
                video_query: [B, E]
                glance_vr_scores, glance_vr_scores_: [B(t), B(v)]
                glance_key_prop_indices: [B(t), B(v)]
                glance_vcmr_scores: [B(t), num_props_glance, B(v)]
                gaze_vr_scores, gaze_vr_scores_: [B(t), B(v)]
                left_neg_prop_scores, right_neg_prop_scores, whole_neg_prop_scores: [B(t), B(v)]
        """
        clip_video_feat = batch["clip_video_feat"]
        frame_video_feat = batch["frame_video_feat"]
        frame_video_mask = batch["frame_video_mask"]
        query_feat = batch["query_feat"]
        query_mask = batch["query_mask"]
        visual_features = self.encode_context(frame_video_feat, frame_video_mask, clip_video_feat)
        video_query = self.encode_query(query_feat, query_mask) 

        # compute saliency features
        saliency_query_feature = self.saliency_proj(video_query)
        saliency_frame_feature = self.saliency_proj(visual_features["gaze_visual_feature"])
        
        # compute left, right width predictions
        left_width_preds = torch.sigmoid(self.left_proj(visual_features["gaze_visual_feature"])).squeeze(-1) # [Nv, Lv]
        right_width_preds = torch.sigmoid(self.right_proj(visual_features["gaze_visual_feature"])).squeeze(-1) # [Nv, Lv]


        glance_res = self.glance_branch(video_query, visual_features)    
        gaze_res = self.gaze_branch(video_query, visual_features, glance_res["glance_key_prop_indices"], epoch)
        
        # compute visual similarities
        visual_similarities = self.compute_visual_similarities(visual_features, glance_res)
        pred_res = {"video_query": video_query, 
                    "saliency_query_feature": saliency_query_feature, 
                    "saliency_frame_feature": saliency_frame_feature,
                    "left_width_preds": left_width_preds,
                    "right_width_preds": right_width_preds,
        }
        pred_res.update(batch)
        pred_res.update(glance_res)
        pred_res.update(gaze_res)
        pred_res.update(visual_features)
        pred_res.update(visual_similarities)
        return pred_res


    def glance_branch(self, video_query, visual_features):
        glance_props_feature = visual_features["glance_props_feature"]
        glance_vr_scores, glance_key_prop_indices, glance_vcmr_scores, glance_prop_features = self.get_clip_scale_scores(
            video_query, glance_props_feature) # [B(t), B(v)]; [B(t), B(v)]; [B(t), num_proposals, B(v)]; []
        glance_vr_scores_ = self.get_clip_scale_scores(video_query, glance_props_feature, normalize=False)[0]
        # import pdb; pdb.set_trace() 
        glance_res = {
            "glance_vr_scores": glance_vr_scores, # [Nt, Nv]
            "glance_key_prop_indices": glance_key_prop_indices, # [Nt, Nv]
            "glance_vcmr_scores": glance_vcmr_scores, # [Nt, Np, Nv]
            "glance_vr_scores_": glance_vr_scores_, # [Nt, Nv]
            "glance_prop_features": glance_prop_features
        }
        return glance_res 
    
    def gaze_branch(self, video_query, visual_features, glance_key_prop_indices, epoch):
        gaze_visual_feature = visual_features["gaze_visual_feature"]
        num_video, max_frames, E = gaze_visual_feature.shape
        num_query= video_query.shape[0]
        
        ## 1. Generate fine-grained proposals around focus points
        focus_points = torch.diag(get_props_from_indices(glance_key_prop_indices, self.gauss_center, self.gauss_width)[0]) # [Nv==Nq]
        focused_centers = focus_points.unsqueeze(-1).expand(num_query, self.num_props_per_center).reshape(-1)                                                               
        focused_widths = torch.linspace(self.width_lower_bound, self.width_upper_bound, steps=self.num_props_per_center).unsqueeze(0).expand(num_query, -1).reshape(-1).to(gaze_visual_feature.device)
        focused_weights = generate_gauss_weight(max_frames, focused_centers, focused_widths, sigma=self.sigma) # [Nv*np, Lv]

        # 2. Compute scores for generated proposals, and select the best proposal for each focus points
        # focused_prop_scores: [Nt, Nv, num_prop_per_center]; focused_prop_features: [Nv, num_prop_per_center, E]
        focused_prop_scores, focused_prop_features = self.gauss_guided_q2v_similarity(focused_weights, video_query, gaze_visual_feature, self.num_props_per_center) # gauss_weighted pooling
        gaze_vr_scores, gaze_vr_indices = focused_prop_scores.max(dim=-1) # [Nt, Nv]
        # import pdb; pdb.set_trace() 
        matched_indices = torch.diag(gaze_vr_indices) # [Nv]
        gaze_prop_features = torch.gather(focused_prop_features, dim=1, index=matched_indices[:, None, None].repeat(1, 1, E)).squeeze(1) # [Nv, E]
        focused_prop_scores_, _ = self.gauss_guided_q2v_similarity(focused_weights, video_query, gaze_visual_feature, self.num_props_per_center, normalize=False)
        gaze_vr_scores_, _ = focused_prop_scores_.max(dim=-1)

        sel_gaze_prop_centers = torch.gather(focused_centers.reshape(num_video, -1), dim=1, index=matched_indices[:, None]).squeeze(-1) # [Nv]
        sel_gaze_prop_widths = torch.gather(focused_widths.reshape(num_video, -1), dim=1, index=matched_indices[:, None]).squeeze(-1) # [Nv]
        sel_gaze_weights = torch.gather(focused_weights.reshape(num_video, self.num_props_per_center, -1), dim=1, index=matched_indices[:, None, None].repeat(1, 1, max_frames)).squeeze()
        # import pdb; pdb.set_trace() 

        # 3. Compute side and whole negative scores for selected proposals for each focus points 
        left_neg_weight, right_neg_weight = self.negative_proposal_mining(max_frames, focused_centers, focused_widths, epoch) # v2
        left_neg_prop_scores_all, _ = self.gauss_guided_q2v_similarity(left_neg_weight, video_query, gaze_visual_feature, self.num_props_per_center)
        right_neg_prop_scores_all, _ = self.gauss_guided_q2v_similarity(right_neg_weight, video_query, gaze_visual_feature, self.num_props_per_center)
        left_neg_prop_scores = torch.gather(left_neg_prop_scores_all, dim=-1, index=gaze_vr_indices.unsqueeze(-1)).squeeze(-1)
        right_neg_prop_scores = torch.gather(right_neg_prop_scores_all, dim=-1, index=gaze_vr_indices.unsqueeze(-1)).squeeze(-1)
        whole_neg_score = self.gauss_guided_q2v_similarity(torch.ones(num_video, max_frames).type_as(gaze_visual_feature), video_query, gaze_visual_feature, 1)[0].squeeze(-1)

        gaze_res = {
            'gaze_vr_scores': gaze_vr_scores, 
            'gaze_vr_scores_': gaze_vr_scores_, 
            'left_neg_prop_scores': left_neg_prop_scores, 
            'right_neg_prop_scores': right_neg_prop_scores, 
            "whole_neg_scores": whole_neg_score, 
            "gaze_prop_features": gaze_prop_features,
            "sel_gaze_prop_centers": sel_gaze_prop_centers,
            "sel_gaze_prop_widths": sel_gaze_prop_widths,
            "sel_gaze_weights": sel_gaze_weights,
        }
        return gaze_res 


    @torch.no_grad()
    def inference(self, query_feat, query_mask, visual_context, num_props_for_each_video=10):
        """
            Inputs:
                query_feat: [Nt, Lt, Et]
                query_mask: [Nt, Lt]
                clip_proposal_feat: [Nv, num_props_glance, E]
                frame_proposal_feat: [Nv, num_propos_glance, E] # use frame features, but glance proposals
                num_props_for_each_video
            Outputs: (Dict)
                glance_vr_scores: [Nt, Nv]
                gaze_vr_scores: [Nt, Nv]
                topk_props: [Nt, Nv, num_props_for_each_video, 2] # relative st&ed
                topk_prop_scores: [Nt, Nv, num_props_for_each_video]

        """
        clip_proposal_feat = visual_context["clip_proposal_feat"]
        frame_proposal_feat = visual_context["frame_proposal_feat"]
        video_query = self.encode_query(query_feat, query_mask)
        # VR (Glance Branch)
        glance_vr_scores, glance_key_prop_indices, glance_vcmr_scores, _ = self.get_clip_scale_scores(video_query, clip_proposal_feat) # [Nt, Nv]
        # VR (Gaze Branch)
        gaze_vr_scores, gaze_key_prop_indices, gaze_vcmr_scores, _ = self.get_clip_scale_scores(video_query, frame_proposal_feat) 
        # VCMR (Gaze Branch Only)
        num_props_for_each_video = min(num_props_for_each_video, gaze_vcmr_scores.shape[1])
        topk_prop_scores, topk_prop_indices = torch.topk(gaze_vcmr_scores.permute(0,2,1), num_props_for_each_video, dim=-1) # (nq, nv, npev)
        num_query, num_video, num_props = topk_prop_scores.shape
        flatted_topk_prop_center, flatted_topk_prop_width = get_props_from_indices(topk_prop_indices.view(num_query*num_video, num_props), self.gauss_center, self.gauss_width)

        flatted_topk_props = torch.stack([torch.clamp(flatted_topk_prop_center - flatted_topk_prop_width/2, min=0), torch.clamp(flatted_topk_prop_center + flatted_topk_prop_width/2, max=1)], dim=-1)
        topk_props = flatted_topk_props.view(num_query, num_video, num_props, 2) 

        infer_res = {
            "glance_vr_scores": glance_vr_scores,
            "gaze_vr_scores": gaze_vr_scores, 
            "topk_props": topk_props,
            "topk_prop_scores": topk_prop_scores
        }
        return infer_res
    

    def gauss_guided_q2v_similarity(self, gauss_weight, modular_roberta_feat, video_feat, num_props, normalize=True):
        def gauss_weighted_pooling(frame_feat, gauss_weight, num_props):
            nv, lv, d = frame_feat.shape
            if frame_feat.shape[0] != gauss_weight.shape[0]:
                frame_feat = frame_feat.unsqueeze(1).expand(nv, num_props, lv, d).reshape(nv*num_props, lv, d)
            gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)
            global_props_vid_feat = torch.bmm(gauss_weight.unsqueeze(1), frame_feat).squeeze(1)
            return global_props_vid_feat
        
        num_video, _, _ = video_feat.shape
    
        global_props_vid_feat_ = gauss_weighted_pooling(video_feat, gauss_weight, num_props).view(num_video, num_props, -1)

        if normalize:
            modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1) # [nq(t),E]
            global_props_vid_feat = F.normalize(global_props_vid_feat_, dim=-1) # [nq(v), num_props_per_center, E]
        else:
            global_props_vid_feat = global_props_vid_feat_
        props_sim_scores = torch.einsum("nd,mpd->nmp", modular_roberta_feat, global_props_vid_feat) 

        return props_sim_scores, global_props_vid_feat_
    
    def encode_context(self, frame_video_feat, video_mask, clip_video_feat):
        glob_vid_feat, glob_vid_mask = self.global_pool1d(frame_video_feat, video_mask) 
        glob_vid_feat, glob_vid_mask = self.global_conv1d(glob_vid_feat, glob_vid_mask) 
        
        glance_visual_feature = self.encode_input(glob_vid_feat, glob_vid_mask, self.clip_input_proj, self.clip_encoder, self.clip_pos_embed)
        gaze_visual_feature = self.encode_input(frame_video_feat, video_mask, self.frame_input_proj, self.clip_encoder, self.frame_pos_embed)

        glance_props_feature = self.gauss_weighted_vid_props_feat(glance_visual_feature) 
        gaze_props_feature = self.gauss_weighted_vid_props_feat(gaze_visual_feature) 
        
        orig_props_feature = self.gauss_weighted_vid_props_feat(frame_video_feat)
        # import pdb; pdb.set_trace()

        visual_features = {
            "gaze_visual_feature": gaze_visual_feature, # [Nv, Lv_gaze, E]
            "gaze_props_feature": gaze_props_feature,   # [Nv, Np, E]; Np=Nc*Nw
            "glance_visual_feature": glance_visual_feature, # [Nv, Lv_glance, E]
            "glance_props_feature": glance_props_feature,   # [Nv, Np, E]
            "orig_props_feature": orig_props_feature    # [Nv, Np, Ev]
        }

        return visual_features
    
    def gauss_weighted_vid_props_feat(self, x_feat):
        # import pdb; pdb.set_trace() 
        nv, lv, dv = x_feat.shape
        # import pdb; pdb.set_trace() 
        self.gauss_center = self.gauss_center.to(x_feat.device) # [Np=Nc*Nw]
        self.gauss_width = self.gauss_width.to(x_feat.device)   # [Np=Nc*Nw]
        gauss_weight = generate_gauss_weight(lv, self.gauss_center, self.gauss_width, self.sigma) # gauss weight
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)
        gauss_weight = gauss_weight.unsqueeze(0).expand(nv, -1, lv)
        proposal_feat_map = torch.bmm(gauss_weight, x_feat)
        return proposal_feat_map

    def negative_proposal_mining(self, props_len, center, width, epoch=1): 
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327 
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device) # [B*num_props, max_frames], [1024,64]

        left_width = torch.clamp(center-width/2, min=0)
        left_center = torch.zeros_like(center)
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = torch.ones_like(center)
        left_neg_weight = Gauss(weight, left_width, left_center) 
        right_neg_weight = Gauss(weight, right_width, right_center) 

        return left_neg_weight, right_neg_weight


    def get_clip_scale_scores(self, modularied_query, context_feat, normalize=True): 
        # modularized_query: [Nv, E]; context_feat: [Nv, Np, E]
        #import pdb; pdb.set_trace() 
        context_feat_ = context_feat
        if normalize:
            modularied_query = F.normalize(modularied_query, dim=-1)
            context_feat = F.normalize(context_feat, dim=-1)

        clip_level_query_context_scores = torch.matmul(context_feat, modularied_query.t()).permute(2, 1, 0) # [Nt, Np, Nv] 
        
        # import pdb; pdb.set_trace() 
        query_context_scores, indices = torch.max(clip_level_query_context_scores, dim=1)  # [Nt, Nv]
        E = context_feat.shape[-1]
        matched_indices = torch.diag(indices)[:, None, None].repeat(1, 1, E) # [Nv, 1, E]
        prop_features = torch.gather(context_feat_, dim=1, index=matched_indices).squeeze(1) # [Nv, E]
        return query_context_scores, indices, clip_level_query_context_scores, prop_features

    
    def encode_input(self, feat, mask, input_proj_layer, encoder_layer, pos_embed_layer):
        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1) 
        return encoder_layer(feat, mask) 

    def encode_query(self, query_feat, query_mask): 
        encoded_query = self.encode_input(query_feat, query_mask, self.query_input_proj, self.query_encoder,self.query_pos_embed) 
        modular_attention_scores = self.modular_vector_mapping(encoded_query)  # (N, L, 2 or 1)
        modular_attention_scores = F.softmax(mask_logits(modular_attention_scores, query_mask.unsqueeze(2)), dim=1)
        modular_queries = torch.bmm(modular_attention_scores.transpose(2,1), encoded_query)
        video_query = modular_queries.squeeze()
        return video_query


    def reset_parameters(self):
        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)