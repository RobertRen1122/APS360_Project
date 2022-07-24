import torch
import torch.nn as nn
from utils import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=1):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions -> (batch_size, S*S*(C+B*5))
        # first, we need to reshape the tensor to a 4D tensor:
        # [batch_size, cell_horizontal, cell_vertical, pred_for_cell]
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # calculate the iou for the 2 predicted bounding boxes over the ground truth box
        iou_b1 = intersection_over_union(predictions[..., self.C+1:self.C+5], target[..., self.C+1:self.C+5])
        iou_b2 = intersection_over_union(predictions[..., self.C+6:self.C+10], target[..., self.C+1:self.C+5])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # We want the box with the highest iou out of the 2 predictions
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # We are npt really going to use the iou_maxes
        # We really want the bestbox, which gives us the index of the best bounding box
        # The index is either 0 or 1 because we are predicting 2 boxes for each cell
        exists_box = target[..., self.C].unsqueeze(3) # 0 or 1 depending on if there is an object in the box or not

        # === Now we calculate the loss for bounding box coordinate predictions===
        box_predictions = exists_box * (
            bestbox * predictions[..., self.C+6:self.C+10]
            + (1 - bestbox) * predictions[..., self.C+1:self.C+5]
        )
        # We want to set boxes with no object in them to 0. We take out the bounding box that's better in terms of iou,
        # which is calculated previously using iou_maxes, bestbox = torch.max(ious, dim=0)
        box_targets = exists_box * target[..., self.C+1:self.C+5]
        # Then we take the square root of width and height of the bounding boxes to reduce the penalty
        # on large bounding box coordinates.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # === Now we calculate the loss for when there is an object ===
        pred_box = (
            bestbox * predictions[..., self.C+5:self.C+6] + (1-bestbox) * predictions[..., self.C:self.C+1]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., self.C:self.C+1])
        )

        # === Now we calculate the loss function for when there is no object ===
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C:self.C+1], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., self.C+5:self.C+6], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., self.C:self.C+1], start_dim=1),
        )

        # === Now we calculate the loss for if the class prediction is correct ===
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2),
            torch.flatten(exists_box * predictions[..., :self.C], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss

