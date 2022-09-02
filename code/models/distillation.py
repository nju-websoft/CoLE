import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_ranks, get_norms, get_scores
from .nbert import NBert
from .nformer import NFormer


def js_loss(p_logits, q_logits):
    """
    JS(p||q) = 0.5 * KL( p || (p+q)/2 ) + 0.5 * KL( q || (p+q)/2 )
    KL(p||q) = sum(p*log(p)/log(q)), CAUTION, in pytorch, KL(p||q) is KLDivLOSS(q, log(p))
    :param p_logits:
    :param q_logits:
    :return: a Jensen loss
    """
    kl_div = nn.KLDivLoss(reduction='batchmean')
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    log_mean_probs = torch.log((p_probs + q_probs) / 2.)

    return (kl_div(log_mean_probs, p_probs) + kl_div(log_mean_probs, q_probs)) / 2


def dkd_loss(logits_student, logits_teacher, target, alpha=1.0, beta=1.0, temperature=1.0):
    """
    :param logits_student: batch_size * num_class
    :param logits_teacher: batch_size * num_class
    :param target: batch_size
    :return:
    """
    # 得到一个矩阵, 目标位置是1, 其他位置是0
    def _get_gt_mask(logits, target):
        target = target.reshape(-1)  # N*C
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    # 得到一个矩阵, 目标位置是0, 其他位置是1
    def _get_other_mask(logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    # 得到 batch_size * 2的矩阵, 分别是target的概率和not_target的概率
    def cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)  # N * 1
        t2 = (t * mask2).sum(1, keepdims=True)  # N * (C-1)
        rt = torch.cat([t1, t2], dim=1)  # 保证target在最前面
        return rt

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)

    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def decoupled_distillation_loss(logits1, logits2, labels, alpha=1.0, beta=1.0, T=1.0):
    """
    :param logits1: logits for model A (Student)
    :param logits2: logits for model B (Teacher)
    :param labels: tensor with shape (batch_size, )
    :param alpha: hyper-parameter
    :param beta: hyper-parameter
    :param T: temperature for distillation
    :return:
    """
    kl_div = nn.KLDivLoss(reduction='batchmean')
    batch_size, num_class = logits1.shape
    label_matrix = torch.arange(num_class).repeat(batch_size, 1).to(labels)  # batch_size * num_class
    label_matrix = torch.eq(label_matrix, labels.unsqueeze(dim=1))  # (batch_size * num_class) vs (batch_size, 1)

    probs1 = F.softmax(logits1 / T, dim=-1)
    probs2 = F.softmax(logits2 / T, dim=-1)

    target_probs1 = probs1[label_matrix]  # (batch_size, )
    dark_probs1 = probs1[torch.logical_not(label_matrix)].reshape(batch_size, -1)  # (batch_size, num_class-1)
    not_target_probs1 = torch.sum(dark_probs1, dim=-1)  # (batch_size, )
    probs1 = torch.stack([target_probs1, not_target_probs1], dim=1)

    target_probs2 = probs2[label_matrix]  # (batch_size, )
    dark_probs2 = probs2[torch.logical_not(label_matrix)].reshape(batch_size, -1)  # (batch_size, num_class-1)
    not_target_probs2 = torch.sum(dark_probs2, dim=-1)
    probs2 = torch.stack([target_probs2, not_target_probs2], dim=1)

    # TCKD
    loss1 = kl_div(torch.log(probs1), probs2) * (T ** 2)

    # 重新归一化
    dark_logits1 = logits1[torch.logical_not(label_matrix)].reshape(batch_size, -1)
    dark_probs1 = F.softmax(dark_logits1 / T, dim=-1)
    dark_logits2 = logits2[torch.logical_not(label_matrix)].reshape(batch_size, -1)
    dark_probs2 = F.softmax(dark_logits2 / T, dim=-1)

    # NCKD
    loss2 = kl_div(torch.log(dark_probs1), dark_probs2) * (T ** 2)

    return alpha * loss1 + beta * loss2


class CoLE(nn.Module):
    def __init__(self, args: dict, nbert: NBert, nformer: NFormer):
        super(CoLE, self).__init__()

        self.device = torch.device(args['device'])
        self.alpha = args['alpha']
        self.beta = args['beta']

        self.text_encoder = nbert
        self.struc_encoder = nformer

    def reload_hyper_params(self, config: dict):
        self.device = torch.device(config['device'])
        self.alpha = config['alpha']
        self.beta = config['beta']

        self.text_encoder.reload_hyper_params(config)
        self.struc_encoder.reload_hyper_params(config)

    def forward(self, batch_data):
        output = self.co_distill(batch_data)
        return output['text_loss'], output['struc_loss']

    def training_step(self, batch, batch_idx):
        return self.forward(batch)

    def training_epoch_end(self, outputs):
        text_losses, struc_losses = [], []
        for text_loss, struc_loss in outputs:
            text_losses.append(text_loss.item())
            struc_losses.append(struc_loss.item())
        return np.round(np.mean(text_losses), 2), np.round(np.mean(struc_losses), 2)

    def validation_step(self, batch, batch_idx):
        output = self.co_distill_validation(batch)

        text_loss, text_output = output['text_loss'], output['text_output']
        text_rank, text_logits = text_output['rank'], text_output['logits']

        struc_loss, struc_output = output['struc_loss'], output['struc_output']
        struc_rank, struc_logits = struc_output['rank'], struc_output['logits']

        # 3.3 get ranks
        text_probs = F.softmax(text_logits, dim=-1)  # batch_size * num_class
        struc_probs = F.softmax(struc_logits, dim=-1)
        # TODO 之后要把a修改成超参数
        a = .8
        probs = a * text_probs + (1 - a) * struc_probs
        labels = batch['labels'].to(self.device)
        filters = batch['filters'].to(self.device)
        rank = get_ranks(probs, labels, filters)
        return text_loss.item(), struc_loss.item(), text_rank, struc_rank, rank

    def validation_epoch_end(self, outputs):
        loss1, loss2, rank1, rank2, rank3 = list(), list(), list(), list(), list()
        for batch_text_loss, batch_struc_loss, batch_text_rank, batch_struc_rank, batch_rank in outputs:
            loss1.append(batch_text_loss)
            loss2.append(batch_struc_loss)
            rank1 += batch_text_rank
            rank2 += batch_struc_rank
            rank3 += batch_rank
        loss1, loss2 = np.mean(loss1), np.mean(loss2)
        text_score = get_scores(rank1, loss1)
        struc_score = get_scores(rank2, loss2)
        cole_score = get_scores(rank3, loss1+loss2)
        return text_score, struc_score, cole_score

    def co_distill_validation(self, batch_data):
        # 1. encode with text encoder and struc encoder respectively, output: {loss, rank, logits}
        text_output = self.text_encoder.link_prediction(batch_data)
        struc_output = self.struc_encoder.link_prediction_validation(batch_data)

        # 2. calculate the distillation loss
        labels = batch_data['labels'].to(self.device)

        # logits: (batch_size, vocab_size)
        text_logits = text_output['logits']
        struc_logits = struc_output['logits']

        # 3.2 decoupled distillation loss
        text_cls_loss = text_output['loss']
        struc_cls_loss = struc_output['loss']
        text_kd_loss = decoupled_distillation_loss(text_logits, struc_logits.detach(), labels)
        struc_kd_loss = decoupled_distillation_loss(struc_logits, text_logits.detach(), labels)

        text_loss = (1 - self.alpha) * text_cls_loss + self.alpha * text_kd_loss
        struc_loss = (1 - self.beta) * struc_cls_loss + self.beta * struc_kd_loss

        return {
            'text_loss': text_loss, 'text_output': text_output,
            'struc_loss': struc_loss, 'struc_output': struc_output
        }

    def co_distill(self, batch_data):
        # 1. encode with text encoder and struc encoder respectively, output: {loss, rank, logits}
        text_output = self.text_encoder.link_prediction(batch_data)
        struc_output = self.struc_encoder.link_prediction(batch_data)

        # 2. calculate the distillation loss
        labels = batch_data['labels'].to(self.device)

        # logits: (batch_size, vocab_size)
        text_logits = text_output['logits']
        struc_logits = struc_output['logits']

        # 3.2 decoupled distillation loss
        text_cls_loss = text_output['loss']
        struc_cls_loss = struc_output['loss']
        text_kd_loss = decoupled_distillation_loss(text_logits, struc_logits.detach(), labels)
        struc_kd_loss = decoupled_distillation_loss(struc_logits, text_logits.detach(), labels)

        # default KL loss
        # kl_loss = nn.KLDivLoss(reduction='batchmean')
        # text_kd_loss = kl_loss(torch.log(F.softmax(text_logits, dim=-1)), F.softmax(struc_logits.detach(), dim=-1))
        # struc_kd_loss = kl_loss(torch.log(F.softmax(struc_logits, dim=-1)), F.softmax(text_logits.detach(), dim=-1))

        text_loss = (1 - self.alpha) * text_cls_loss + self.alpha * text_kd_loss
        struc_loss = (1 - self.beta) * struc_cls_loss + self.beta * struc_kd_loss

        return {
            'text_loss': text_loss, 'text_output': text_output,
            'struc_loss': struc_loss, 'struc_output': struc_output
        }

    def configure_optimizers(self, total_steps: int):
        return self.text_encoder.configure_optimizers(total_steps), self.struc_encoder.configure_optimizers(total_steps)

    # useless
    def clip_grad_norm(self):
        raise ValueError('Do not call clip_gram_norm for Co-distillation')
        text_info = self.text_encoder.clip_grad_norm()
        struc_info = self.struc_encoder.clip_grad_norm()
        info = f'{text_info}; {struc_info}'
        return info

    def grad_norm(self):
        return self.text_encoder.grad_norm(), self.struc_encoder.grad_norm()


