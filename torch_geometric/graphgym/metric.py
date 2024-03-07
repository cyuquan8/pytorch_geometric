import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric

@register_metric('ogbn_products_accuracy')
def ogbn_products_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbn-products', f"Dataset is {cfg.dataset.name}. Metric is for ogbn-products."
	assert task_type == 'classification_multi', f"Task type is {task_type}. ogbn-products requires classification_multi"

	from ogb.nodeproppred import Evaluator

	# (num_nodes, num_tasks == 1), (num_nodes, num_classes == 47)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (num_nodes, num_tasks == 1)
	pred_int = pred_score.max(dim=1, keepdim=True)[1]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	acc = evaluator.eval({'y_true': true, 'y_pred': pred_int})['acc']

	return acc

@register_metric('ogbn_proteins_rocauc')
def ogbn_proteins_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbn-proteins', f"Dataset is {cfg.dataset.name}. Metric is for ogbn-proteins."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbn-proteins requires classification_binary"

	from ogb.nodeproppred import Evaluator

	# (num_nodes, num_tasks == 112), (num_nodes, num_tasks == 112)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	rocauc = evaluator.eval({'y_true': true, 'y_pred': pred_score})['rocauc']

	return rocauc

@register_metric('ogbn_arxiv_accuracy')
def ogbn_arxiv_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbn-arxiv', f"Dataset is {cfg.dataset.name}. Metric is for ogbn-arxiv."
	assert task_type == 'classification_multi', f"Task type is {task_type}. ogbn-arxiv requires classification_multi"

	from ogb.nodeproppred import Evaluator

	# (num_nodes, num_tasks == 1), (num_nodes, num_classes == 40)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (num_nodes, num_tasks == 1)
	pred_int = pred_score.max(dim=1, keepdim=True)[1]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	acc = evaluator.eval({'y_true': true, 'y_pred': pred_int})['acc']

	return acc

@register_metric('ogbn_papers100M_accuracy')
def ogbn_papers100M_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbn-papers100M', f"Dataset is {cfg.dataset.name}. Metric is for ogbn-papers100M."
	assert task_type == 'classification_multi', f"Task type is {task_type}. ogbn-papers100M requires classification_multi"

	from ogb.nodeproppred import Evaluator

	# (num_nodes, num_tasks == 1), (num_nodes, num_classes == 172)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (num_nodes, num_tasks == 1)
	pred_int = pred_score.max(dim=1, keepdim=True)[1]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	acc = evaluator.eval({'y_true': true, 'y_pred': pred_int})['acc']

	return acc

@register_metric('ogbn_mag_accuracy')
def ogbn_mag_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbn-mag', f"Dataset is {cfg.dataset.name}. Metric is for ogbn-mag."
	assert task_type == 'classification_multi', f"Task type is {task_type}. ogbn-mag requires classification_multi"

	from ogb.nodeproppred import Evaluator

	# (num_nodes, num_tasks == 1), (num_nodes, num_classes == 349)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (num_nodes, num_tasks == 1)
	pred_int = pred_score.max(dim=1, keepdim=True)[1]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	acc = evaluator.eval({'y_true': true, 'y_pred': pred_int})['acc']

	return acc

@register_metric('ogbl_ppa_hits_100')
def ogbl_ppa_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbl-ppa', f"Dataset is {cfg.dataset.name}. Metric is for ogbl-ppa."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbl-ppa requires classification_binary"

	from ogb.linkproppred import Evaluator

	# (num_edges, ), (num_edges, )
	true, pred = torch.cat(true), torch.cat(pred)
	# (pos_num_edges, ), (neg_num_edges, )
	y_pred_pos = pred[true == 1]
	y_pred_neg = pred[true == 0]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	hits_100 = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['hits@100']

	return hits_100

@register_metric('ogbl_collab_hits_50')
def ogbl_collab_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbl-collab', f"Dataset is {cfg.dataset.name}. Metric is for ogbl-collab."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbl-collab requires classification_binary"

	from ogb.linkproppred import Evaluator

	# (num_edges, ), (num_edges, )
	true, pred = torch.cat(true), torch.cat(pred)
	# (pos_num_edges, ), (neg_num_edges, )
	y_pred_pos = pred[true == 1]
	y_pred_neg = pred[true == 0]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	hits_50 = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['hits@50']

	return hits_50

@register_metric('ogbl_ddi_hits_20')
def ogbl_ddi_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbl-ddi', f"Dataset is {cfg.dataset.name}. Metric is for ogbl-ddi."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbl-ddi requires classification_binary"

	from ogb.linkproppred import Evaluator

	# (num_edges, ), (num_edges, )
	true, pred = torch.cat(true), torch.cat(pred)
	# (pos_num_edges, ), (neg_num_edges, )
	y_pred_pos = pred[true == 1]
	y_pred_neg = pred[true == 0]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	hits_20 = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['hits@20']

	return hits_20

@register_metric('ogbl_vessel_rocauc')
def ogbl_vessel_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbl-vessel', f"Dataset is {cfg.dataset.name}. Metric is for ogbl-vessel."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbl-vessel requires classification_binary"

	from ogb.linkproppred import Evaluator

	# (num_edges, ), (num_edges, )
	true, pred = torch.cat(true), torch.cat(pred)
	# (pos_num_edges, ), (neg_num_edges, )
	y_pred_pos = pred[true == 1]
	y_pred_neg = pred[true == 0]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	rocauc = evaluator.eval({'y_pred_pos': y_pred_pos, 'y_pred_neg': y_pred_neg})['rocauc']

	return rocauc

@register_metric('ogbg_molhiv_rocauc')
def ogbg_molhiv_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbg-molhiv', f"Dataset is {cfg.dataset.name}. Metric is for ogbg-molhiv."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbg-molhiv requires classification_binary"

	from ogb.graphproppred import Evaluator

	# (batch, num_tasks == 1), (batch, )
	true, pred_score = torch.cat(true), torch.cat(pred)
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	rocauc = evaluator.eval({'y_true': true, 'y_pred': pred_score[:, None]})['rocauc']

	return rocauc

@register_metric('ogbg_molpcba_average_precision')
def ogbg_molpcba_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbg-molpcba', f"Dataset is {cfg.dataset.name}. Metric is for ogbg-molpcba."
	assert task_type == 'classification_binary', f"Task type is {task_type}. ogbg-molpcba requires classification_binary"

	from ogb.graphproppred import Evaluator

	# (batch, num_tasks == 128), (batch, num_tasks == 128)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (batch, num_tasks == 128)
	pred_int = (pred_score > cfg.model.thresh).long()
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	ap = evaluator.eval({'y_true': true, 'y_pred': pred_int})['ap']

	return ap

@register_metric('ogbg_ppa_accuracy')
def ogbg_ppa_evaluator(true, pred, task_type):
	assert cfg.dataset.name == 'ogbg-ppa', f"Dataset is {cfg.dataset.name}. Metric is for ogbg-ppa."
	assert task_type == 'classification_multi', f"Task type is {task_type}. ogbg-ppa requires classification_multi"

	from ogb.graphproppred import Evaluator

	# (batch, num_tasks == 1), (batch, num_classes == 37)
	true, pred_score = torch.cat(true), torch.cat(pred)
	# (batch, num_tasks == 1)
	pred_int = pred_score.max(dim=1, keepdim=True)[1]
	# float
	evaluator = Evaluator(name=cfg.dataset.name)
	acc = evaluator.eval({'y_true': true, 'y_pred': pred_int})['acc']

	return acc