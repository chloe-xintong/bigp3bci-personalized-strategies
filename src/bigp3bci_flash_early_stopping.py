"""Flash AUC/log-loss checkpoint decisions for future EEGNet; no network training.

The caller saves model weights on improved=True and restores that best
checkpoint after stopping. Character accuracy is not accepted as an input.
"""
import numpy as np
from bigp3bci_swlda import auc_rank
from bigp3bci_probability import balanced_log_loss


class FlashAUCEarlyStopping:
    def __init__(self,patience=15,min_delta=0.):
        if patience<1 or min_delta<0:raise ValueError('Invalid stopping settings')
        self.patience=patience;self.min_delta=min_delta
        self.best_auc=-np.inf;self.best_epoch=None;self.bad_epochs=0;self.last_epoch=-1

    def update(self,flash_labels,positive_scores,epoch):
        y=np.asarray(flash_labels);score=np.asarray(positive_scores,float)
        if y.ndim!=1 or score.shape!=y.shape or not np.isfinite(score).all():
            raise ValueError('Pass finite per-flash scores and matching one-dimensional labels')
        if not np.array_equal(np.unique(y),[0,1]):raise ValueError('Validation flashes must contain both binary classes')
        if epoch<=self.last_epoch:raise ValueError('Epoch numbers must increase')
        self.last_epoch=epoch
        auc=auc_rank(y,score);improved=auc>self.best_auc+self.min_delta
        if improved:self.best_auc=auc;self.best_epoch=epoch;self.bad_epochs=0
        else:self.bad_epochs+=1
        return dict(flash_roc_auc=auc,improved=improved,should_stop=self.bad_epochs>=self.patience,
                    best_epoch=self.best_epoch,best_auc=self.best_auc)


class FlashLogLossEarlyStopping:
    """Minimize balanced validation flash log-loss; inputs must be probabilities."""
    def __init__(self,patience=15,min_delta=0.):
        if patience<1 or min_delta<0:raise ValueError('Invalid stopping settings')
        self.patience=patience;self.min_delta=min_delta
        self.best_loss=np.inf;self.best_epoch=None;self.bad_epochs=0;self.last_epoch=-1

    def update(self,flash_labels,positive_probabilities,epoch):
        if epoch<=self.last_epoch:raise ValueError('Epoch numbers must increase')
        loss=balanced_log_loss(flash_labels,positive_probabilities)
        self.last_epoch=epoch;improved=loss<self.best_loss-self.min_delta
        if improved:self.best_loss=loss;self.best_epoch=epoch;self.bad_epochs=0
        else:self.bad_epochs+=1
        return dict(weighted_flash_log_loss=loss,improved=improved,should_stop=self.bad_epochs>=self.patience,
                    best_epoch=self.best_epoch,best_loss=self.best_loss)
