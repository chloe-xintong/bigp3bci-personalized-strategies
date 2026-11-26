"""Train-only SWLDA probability calibration and class-balanced flash log-loss."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from bigp3bci_swlda import fit_swlda,score_swlda


def binary_class_weights(labels):
    y=np.asarray(labels)
    if y.ndim!=1 or not np.array_equal(np.unique(y),[0,1]):
        raise ValueError('Both binary classes are required')
    return {c:len(y)/(2*int((y==c).sum())) for c in [0,1]}


def balanced_log_loss(labels,probabilities,eps=1e-7):
    y=np.asarray(labels);p=np.asarray(probabilities,float)
    binary_class_weights(y)
    if p.shape!=y.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Pass finite positive-class probabilities in [0,1]')
    p=np.clip(p,eps,1-eps)
    return float(-.5*(np.log(p[y==1]).mean()+np.log1p(-p[y==0]).mean()))


def fit_platt(scores,labels,l2=.001):
    s=np.asarray(scores,float);y=np.asarray(labels,float)
    binary_class_weights(y)
    if s.shape!=y.shape or not np.isfinite(s).all():raise ValueError('Invalid training calibration scores')
    mean=float(s.mean());scale=float(s.std());scale=scale if scale>1e-12 else 1.
    z=(s-mean)/scale
    def objective(theta):
        a,b=theta;t=a*z+b;residual=expit(t)-y
        loss=np.mean(np.logaddexp(0,t)-y*t)+.5*l2*a*a
        return loss,np.array([np.mean(residual*z)+l2*a,residual.mean()])
    initial=[0.,float(np.log(y.mean()/(1-y.mean())))]
    result=minimize(objective,initial,jac=True,bounds=[(0,None),(None,None)],method='L-BFGS-B',
                    options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not result.success:raise ValueError('Train-only Platt fitting failed: '+str(result.message))
    return dict(slope=float(result.x[0]),intercept=float(result.x[1]),score_mean=mean,score_scale=scale,
                l2=l2,training_target_count=int(y.sum()),training_nontarget_count=int((y==0).sum()),
                training_class_weights=binary_class_weights(y),optimizer_success=True)


def predict_platt(calibrator,scores):
    return expit(calibrator['slope']*(np.asarray(scores,float)-calibrator['score_mean'])/calibrator['score_scale']+calibrator['intercept'])


def fit_grouped_training_platt(x,y,groups,swlda_cfg,final_model,n_folds=3,l2=.001):
    """All calibration inputs are within the current model's training budget.

    Cross-fit whole characters so overlapping flashes stay together. At k=1
    current-only there is only one character: explicitly use in-sample scores,
    never validation data, and flag this limited fallback in the report.
    """
    x=np.asarray(x,float);y=np.asarray(y,float);groups=np.asarray(groups)
    unique=list(dict.fromkeys(groups.tolist()))
    scores=np.full(len(y),np.nan)
    folds=[]
    if len(unique)<2:
        scores=score_swlda(final_model,x);method='training_in_sample_single_character'
    else:
        method='training_character_grouped_oof'
        for fold,held in enumerate(np.array_split(np.asarray(unique),min(n_folds,len(unique)))):
            validation=np.isin(groups,held);training=~validation
            model=fit_swlda(x[training],y[training],swlda_cfg)
            scores[validation]=score_swlda(model,x[validation])
            folds.append(dict(fold=fold,fit_characters=int(len(set(groups[training]))),
                              held_characters=held.tolist(),fit_status=model['status']))
    if not np.isfinite(scores).all():raise ValueError('Missing training-only calibration scores')
    result=fit_platt(scores,y,l2=l2)
    result.update(method=method,training_characters=len(unique),folds=folds,
                  validation_labels_used=False,calibration_class_weighting='none; natural training prior')
    return result
