import numpy as np
from datasets import load_dataset

def scorer(top5_preds, lang, split):
    """
    Compute MRR@5 given top 5 predictions, language and split information.

    Args:
        top5_preds: list of lists (shape [num_queries, 5])
        lang: str to indicate the language (de, en, fr)
        split: str to indicate the data split (train, dev)

    Returns:
        float: MRR@5 score
    """
        
    
    assert lang in ["de", "en", "fr"], "You need to provide a correct language parameter (de, en, fr)"
    assert split in ["train", "dev"], "You need to provide a correct split parameter (train, dev)"

    data = load_dataset("sschellhammer/CT26_Task1_SourceRetrievalForScientificWebClaims", lang)
    
    datasplit = data[split]
    labels = [str(label) for label in np.array(datasplit["pubkey"]).tolist()]

    mrr_scores = []
    for preds, label in zip(top5_preds, labels):
        assert len(preds) == 5, "exactly 5 predictions per query should be provided"
        preds_as_str = [str(pred) for pred in preds]
        
        if label in preds_as_str:
            mrr_scores.append(1 / (preds_as_str.index(label) + 1))
        else:
            mrr_scores.append(0)  

    return float(np.mean(mrr_scores))
