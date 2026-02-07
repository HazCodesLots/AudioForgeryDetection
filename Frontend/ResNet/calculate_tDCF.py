import os
import numpy as np

def load_asv_metrics(asv_score_file, asv_protocol_file):
    asv_data = {}
    
    with open(asv_protocol_file, 'r') as f_prot, open(asv_score_file, 'r') as f_score:
        prot_lines = f_prot.readlines()
        score_lines = f_score.readlines()
        
        if len(prot_lines) != len(score_lines):
            print(f"[WARNING] Protocol lines ({len(prot_lines)}) != Score lines ({len(score_lines)})")
            
        for prot_line, score_line in zip(prot_lines, score_lines):
            prot_parts = prot_line.strip().split()
            score_parts = score_line.strip().split()
            
            audio_id = prot_parts[1]
            try:
                score = float(score_parts[-1])
                key = score_parts[1] 
                asv_data[audio_id] = {'score': score, 'key': key}
            except ValueError:
                continue
                
    print(f"Loaded {len(asv_data)} ASV scores.")
    return asv_data

def compute_eer(target_scores, nontarget_scores):
    """ Returns EER and Threshold """
    if len(target_scores) == 0 or len(nontarget_scores) == 0: return 0.0, 0.0
    
    tgt_scores = sorted(target_scores)
    non_scores = sorted(nontarget_scores)
    from sklearn.metrics import roc_curve
    y_true = np.concatenate([np.ones(len(tgt_scores)), np.zeros(len(non_scores))])
    y_score = np.concatenate([tgt_scores, non_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    threshold = thresholds[eer_idx]
    return eer, threshold

def compute_tDCF_legacy(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost=False):
    # Official constants from ASVspoof 2019 evaluation plan
    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + \
         cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv
         
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] * (1 - Pmiss_asv)
         
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * (1 - Pmiss_spoof_asv)
    
    if C0 == 0: 
        return 0.0

    scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    labels = np.concatenate((np.ones(len(bonafide_score_cm)), np.zeros(len(spoof_score_cm))))
    indices = np.argsort(scores)[::-1]
    sorted_labels = labels[indices]

    n_bonafide = len(bonafide_score_cm)
    n_spoof = len(spoof_score_cm)
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    Pmiss_cm = (n_bonafide - tps) / n_bonafide
    Pfa_cm = fps / n_spoof

    tDCF_raw = C1 * Pmiss_cm + C2 * Pfa_cm
    tDCF_norm = tDCF_raw / t_DCF_default
    return float(np.min(tDCF_norm))

def compute_min_tDCF(cm_scores, audio_ids, asv_data, cost_model=None):
    if cost_model is None:
        cost_model = {
            'Ptar': 0.05,
            'Pnon': 0.90, # Official value
            'Pspoof': 0.05,
            'Cmiss': 1,
            'Cfa': 10, 
            'Cfa_spoof': 10
        }
    
    bonafide_cm = []
    bonafide_asv = []
    target_asv = []
    nontarget_asv = []
    spoof_cm = []
    spoof_asv = []
    
    for score, aid in zip(cm_scores, audio_ids):
        if aid not in asv_data:
            continue
        
        asv_score = asv_data[aid]['score']
        key = asv_data[aid]['key']
        
        if key == 'target':
            bonafide_cm.append(score)
            bonafide_asv.append(asv_score)
            target_asv.append(asv_score)
        elif key == 'nontarget':
            bonafide_cm.append(score)
            bonafide_asv.append(asv_score)
            nontarget_asv.append(asv_score)
        elif key == 'spoof':
            spoof_cm.append(score)
            spoof_asv.append(asv_score)
    
    bonafide_cm = np.array(bonafide_cm)
    bonafide_asv = np.array(bonafide_asv)
    spoof_cm = np.array(spoof_cm)
    spoof_asv = np.array(spoof_asv)
    target_asv = np.array(target_asv) if target_asv else np.array([])
    nontarget_asv = np.array(nontarget_asv) if nontarget_asv else np.array([])
    
    if len(bonafide_cm) == 0 or len(spoof_cm) == 0:
        return 1.0
    
    all_asv = np.concatenate([bonafide_asv, spoof_asv])
    all_labels = np.concatenate([np.ones(len(bonafide_asv)), np.zeros(len(spoof_asv))])
    from sklearn.metrics import roc_curve
    fpr, tpr, asv_thresholds = roc_curve(all_labels, all_asv, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    asv_threshold = asv_thresholds[eer_idx]
    
    Pmiss_asv = (target_asv < asv_threshold).mean() if len(target_asv) > 0 else 0.0
    Pfa_asv = (nontarget_asv >= asv_threshold).mean() if len(nontarget_asv) > 0 else 0.0
    Pmiss_spoof_asv = (spoof_asv >= asv_threshold).mean() if len(spoof_asv) > 0 else 0.0
    
    bonafide_cm_accept = 1.0 - bonafide_cm
    spoof_cm_accept = 1.0 - spoof_cm
    return compute_tDCF_legacy(
        bonafide_cm_accept, spoof_cm_accept,
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv,
        cost_model
    )