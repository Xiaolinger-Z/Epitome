import torch
import torch.nn as nn
import numpy as np


class TripletLoss(nn.Module):
	def __init__(self, loss_type='batch_hard', margin=0.5):
		super(TripletLoss, self).__init__()
		self.loss_type = loss_type
		self.margin = margin

	def forward(self, labels, embeddings):
		if self.loss_type == 'batch_hard':
			loss = self.batch_hard_triplet_loss(labels, embeddings)
		else:
			loss = nn.CrossEntropyLoss()
		return loss

	def batch_hard_triplet_loss(self, labels, embeddings, squared=False):
		pairwise_distances = self._pairwise_distance(embeddings)

		mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
		mask_anchor_positive = torch.tensor(mask_anchor_positive, dtype=torch.float).cuda()

		anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_distances)
		hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]

		mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
		mask_anchor_negative = torch.tensor(mask_anchor_negative, dtype=torch.float).cuda()

		max_anchor_negative_dist = torch.max(pairwise_distances, dim=1, keepdim=True)[0]
		anchor_negative_dist = pairwise_distances + max_anchor_negative_dist * (torch.tensor(1.0) - mask_anchor_negative)
		hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]

		triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + self.margin, torch.zeros(hardest_negative_dist.shape).cuda())

		triplet_loss = torch.mean(triplet_loss)
		return triplet_loss

	@staticmethod
	def _get_anchor_positive_triplet_mask(labels):
		indices_equal = torch.eye(labels.shape[0]).cuda()

		indices_not_equal = np.logical_not(indices_equal.cpu().data.numpy())
		labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
		mask = np.logical_and(indices_not_equal, labels_equal.cpu().data.numpy())

		return mask

	@staticmethod
	def _get_anchor_negative_triplet_mask(labels):
		labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
		mask = np.logical_not(labels_equal.cpu().data.numpy())
		return mask

	@staticmethod
	def _pairwise_distance(embeddings, squared=False):
		dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

		square_norm = torch.diag(dot_product)

		distances = torch.unsqueeze(square_norm, dim=1) - 2.0 * dot_product + torch.unsqueeze(square_norm, dim=0)

		distances = torch.max(distances, torch.zeros(distances.shape).cuda())

		# print('distances:%s' % distances)
		if not squared:

			mask = torch.eq(distances, 0.0).float().clone().detach()

			distances = distances + mask * 1e-16
			distances = torch.sqrt(distances)
			distances = distances * (torch.tensor(1.0) - mask)
		return distances