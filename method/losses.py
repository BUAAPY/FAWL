import torch 
import torch.nn as nn
import copy  

from utils.model_utils import cos_sim
import torch.nn.functional as F
import math 

class SetCriterion(nn.Module):
    def __init__(self, config, loss_names=None):
        super().__init__()
        self.config = config 
        self.use_hard_negative = False
        self.loss_map = {
            "glance_inter_loss": (self.get_glance_inter_loss, self.config.alpha1),
            "gaze_inter_loss": (self.get_gaze_inter_loss, self.config.alpha2),
            "gaze_intra_loss": (self.get_gaze_intra_loss, self.config.alpha3),
            "saliency_loss": (self.get_saliency_loss, self.config.alpha4),
            "boundary_loss": (self.get_boundary_loss,self.config.alpha5)
        }
        self.loss_names = loss_names
        if self.loss_names is None:
            self.loss_names = list(self.loss_map.keys())
    

    def get_saliency_loss(self, pred_res,loss_dict, epoch):
        gaze_visual_feature_ = F.normalize(pred_res["saliency_frame_feature"],dim=-1)
        video_query_ = F.normalize(pred_res["saliency_query_feature"],dim=-1)
        saliency_scores= torch.bmm(gaze_visual_feature_, video_query_.unsqueeze(-1)).squeeze(-1) # [Nq, Lv]
        saliency_scores = saliency_scores.clamp(min=0, max=1)

        sel_gaze_prop_centers = pred_res["sel_gaze_prop_centers"]
        sel_gaze_prop_widths = pred_res["sel_gaze_prop_widths"]
        Lv = gaze_visual_feature_.shape[1]
        left = ((sel_gaze_prop_centers-sel_gaze_prop_widths/2)*Lv).clamp(min=0).type(torch.long)
        right = ((sel_gaze_prop_centers+sel_gaze_prop_widths/2)*Lv).clamp(max=Lv-1).type(torch.long)
        target_scores = torch.zeros_like(saliency_scores)
        for i, (l, r) in enumerate(zip(left, right)):
            target_scores[i][l:r+1] = 1. 

        saliency_loss = F.binary_cross_entropy(saliency_scores, target_scores)
        loss_dict["saliency_loss"] = saliency_loss
        return saliency_loss

    def get_boundary_loss(self, pred_res, loss_dict, epoch):
        sel_gaze_prop_centers = pred_res["sel_gaze_prop_centers"] # [Nv]
        sel_gaze_prop_widths = pred_res["sel_gaze_prop_widths"] # [Nv]
        left_width_preds = pred_res["left_width_preds"] # [Nv, Lv]
        Nv, Lv = left_width_preds.shape 
        right_width_preds = pred_res["right_width_preds"]
        
        self_x = torch.linspace(0, 1, steps=Lv).unsqueeze(0).repeat(Nv, 1).to(left_width_preds.device)
        pred_left = self_x - left_width_preds
        pred_right = self_x + right_width_preds

        target_left = (sel_gaze_prop_centers-sel_gaze_prop_widths/2).clamp(min=0., max=1.)
        target_right = (sel_gaze_prop_centers+sel_gaze_prop_widths/2).clamp(min=0., max=1.) # [Nv]

        target_left_ids = (target_left * Lv).type(torch.long)
        target_right_ids = (target_right * Lv).type(torch.long)

        loss_left, loss_right = 0, 0
        for i in range(Nv):
            p_l = pred_left[i, target_left_ids[i]:target_right_ids[i]+1]
            p_r = pred_right[i, target_left_ids[i]:target_right_ids[i]+1]
            t_l = torch.ones_like(p_l) * target_left[i]
            t_r = torch.ones_like(p_r) * target_right[i]
            loss_left = loss_left + F.mse_loss(p_l, t_l)
            loss_right = loss_right + F.mse_loss(p_r, t_r)

        loss_left = loss_left / Nv 
        loss_right = loss_right / Nv 

        boundary_loss = loss_left + loss_right
        loss_dict["boundary_loss"] = boundary_loss
        return boundary_loss


    def forward(self, pred_res, epoch):
        # import pdb; pdb.set_trace() 
        loss_dict = {}
        loss = 0 
        for loss_name in self.loss_names:
            loss_func, loss_weight = self.loss_map[loss_name]
            loss = loss + loss_weight * loss_func(pred_res, loss_dict, epoch)
        # glance_inter_loss = self.get_glance_inter_loss(pred_res, loss_dict, epoch)
        # gaze_inter_loss = self.get_gaze_inter_loss(pred_res, loss_dict, epoch)
        # gaze_intra_loss = self.get_gaze_intra_loss(pred_res, loss_dict, epoch) 
        # saliency_loss = self.get_saliency_loss(pred_res, loss_dict, epoch)
        # boundary_loss = self.get_boundary_loss(pred_res, loss_dict, epoch)
        # loss = self.config.alpha1*glance_inter_loss + self.config.alpha2*gaze_inter_loss + self.config.alpha3*gaze_intra_loss+\
        #     self.config.alpha4 * saliency_loss+ self.config.alpha5 * boundary_loss
        
        loss_dict["loss_overall"] = loss 

        return loss, loss_dict
 
    def info_nce_loss(self, scores, video_sims, epoch):
        device = scores.device
        bsz = scores.shape[0]
        scores = scores.view(bsz, bsz, -1)
        nominator = scores * torch.eye(bsz, dtype=torch.float32, device=device)[:, :, None] # [bsz, bsz, 1]
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1) # [bsz]

        denominator = torch.cat((scores, scores.permute(1, 0, 2)), dim=1).view(bsz, -1)
        if video_sims is not None:
            video_weights = video_sims * 0.5 + 0.5 # [0,1]
            text_weights = torch.ones_like(video_weights)
            weights = torch.cat((video_weights, text_weights), dim=1) # [[Nv,Nv],[Nt,Nt]]->[bsz, 2*bsz] 
            denominator = torch.exp(denominator) * weights
            denominator = torch.log(torch.sum(denominator, dim=1))
        else:
            denominator = torch.logsumexp(denominator, dim=1)
        
        
        return torch.mean(denominator - nominator)
    

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.use_hard_negative = use_hard_negative
        self.hard_pool_size = hard_pool_size
    
    def get_glance_inter_loss(self, pred_res, loss_dict, epoch):
        proj_video_sims = pred_res["proj_video_glance_sims"]
        orig_video_sims = pred_res["orig_video_sims"]
        if epoch < self.config.warmup_epoch:
            video_sims = orig_video_sims
        else:
            video_sims = proj_video_sims
        glance_inter_nce_loss = 0.04 * self.info_nce_loss(pred_res["glance_vr_scores_"], video_sims, epoch) 
        glance_inter_trip_loss = self.get_frame_trip_loss(pred_res["glance_vr_scores"]) 
        glance_inter_loss = glance_inter_nce_loss + glance_inter_trip_loss
        loss_dict["glance_inter_nce_loss"] = glance_inter_nce_loss
        loss_dict["glance_inter_trip_loss"] = glance_inter_trip_loss
        return glance_inter_loss 

    def get_gaze_inter_loss(self, pred_res, loss_dict, epoch):
        proj_video_sims = pred_res["proj_video_gaze_sims"]
        orig_video_sims = pred_res["orig_video_sims"]
        if epoch < self.config.warmup_epoch:
            video_sims = orig_video_sims
        else:
            video_sims = proj_video_sims
        gaze_inter_nce_loss = 0.04 * self.info_nce_loss(pred_res['gaze_vr_scores_'], video_sims, epoch)
        gaze_inter_trip_loss = self.get_frame_trip_loss(pred_res['gaze_vr_scores']) 
        gaze_inter_loss = gaze_inter_nce_loss + gaze_inter_trip_loss 
        loss_dict["gaze_inter_nce_loss"] = gaze_inter_nce_loss
        loss_dict["gaze_inter_trip_loss"] = gaze_inter_trip_loss
        return gaze_inter_loss

    def get_gaze_intra_loss(self, pred_res, loss_dict, epoch):
        left_neg_scores = torch.diag(pred_res['left_neg_prop_scores']) #[idx, idx] [B]
        right_neg_scores = torch.diag(pred_res['right_neg_prop_scores']) # [B]
        whole_neg_score = torch.diag(pred_res['whole_neg_scores'])        # [B]
        matched_scores= torch.diag(pred_res['gaze_vr_scores']) # [B]
        # side negative:
        _, gaze_intra_left_trip_loss = self.vector_trip_loss(matched_scores, left_neg_scores, self.config.intra_margin) 
        _, gaze_intra_right_trip_loss = self.vector_trip_loss(matched_scores, right_neg_scores, self.config.intra_margin)
        frame_intra_side_loss = gaze_intra_left_trip_loss + gaze_intra_right_trip_loss 
        # hard negative: whole video
        _, gaze_intra_whole_trip_loss = self.vector_trip_loss(matched_scores, whole_neg_score, 0.1)
        gaze_intra_loss = self.config.beta1 * frame_intra_side_loss + self.config.beta2 * gaze_intra_whole_trip_loss 
        loss_dict["gaze_intra_left_trip_loss"] = gaze_intra_left_trip_loss
        loss_dict["gaze_intra_right_trip_loss"] = gaze_intra_right_trip_loss
        loss_dict["gaze_intra_whole_trip_loss"] = gaze_intra_whole_trip_loss
        return gaze_intra_loss
        

    def get_ranking_loss(self, pos_score, neg_score):
        return torch.clamp(self.config.margin + neg_score - pos_score, min=0).sum() / len(pos_score)

    def vector_trip_loss(self, pos_sim_score, neg_sim_score, margin): 
        trip_loss = torch.clamp(neg_sim_score - pos_sim_score + margin, min=0)

        return trip_loss, trip_loss.mean()

    def get_neg_scores(self, scores, scores_masked):
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)

        _, sorted_scores_indices = torch.sort(scores_masked, descending=True, dim=1)

        sample_min_idx = 1  # skip the masked positive

        sample_max_idx = min(sample_min_idx + self.hard_pool_size, bsz) if self.use_hard_negative else bsz 
        sampled_neg_score_indices = sorted_scores_indices[batch_indices, torch.randint(sample_min_idx, sample_max_idx,
                                                                                       size=(bsz,)).to(scores.device)]

        sampled_neg_scores = scores[batch_indices, sampled_neg_score_indices]  
        return sampled_neg_scores

    def get_frame_trip_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        # import pdb; pdb.set_trace() 
        bsz = len(query_context_scores)

        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        pos_scores = query_context_scores[diagonal_indices, diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(query_context_scores.transpose(0, 1),
                                                           query_context_scores_masked.transpose(0, 1))       

        loss_neg_ctx = self.get_ranking_loss(pos_scores, pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores, neg_query_pos_context_scores)
        return loss_neg_ctx + loss_neg_q 
    