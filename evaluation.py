from utils.eval.f1 import F1
from utils.metric import AverageMeter
from utils.eval.meteor import Meteor
from utils.eval.rouge import Rouge
import utils.eval.google_bleu as google_bleu


def eval_accuracies(hypothesis_list, reference_list, mode='valid'):
    """An unofficial evalutation helper.
     Arguments:
        hypothese_list: A mapping from instance id to predicted sequences.
        reference_list: A mapping from instance id to ground truth sequences.
        copy_info: Map of id --> copy information.
        sources: Map of id --> input text sequence.
        filename:
        print_copy_info:
    """
    assert(sorted(reference_list.keys()) == sorted(hypothesis_list.keys()))
    # Compute BLEU
    _, bleu, ind_bleu = google_bleu.corpus_bleu(reference_list, hypothesis_list)

    # Compute ROGUE
    rouge_l, ind_rogue = Rouge().compute_score(reference_list, hypothesis_list)

    # Compute METEOR
    if mode == 'test':
        meteor, _ = Meteor().compute_score(reference_list, hypothesis_list)
    else:
        meteor = 0

    # Compute F1, Precision, Recall
    f1, precision, recall = AverageMeter(), AverageMeter(), AverageMeter()
    """
    hypothesis_list example
    {
        0: ['the the the given fo...e the </s>'], 
        1: ['return the first for...e the </s>'], 
        2: ['setup the given for ...e the </s>'], 
        3: ['expand the given for...e the </s>'], 
        4: ['test that the given ...e the </s>'], 
        5: ['attach the given for...e the </s>'], 
        6: ['add the given for th...e the </s>'], 
        7: ['guess the given for ...e the </s>'], 
        8: ['return the given for...e the </s>'], 
        9: ['open the filepath fo...e the </s>'], 
        10: ['open the pathname fo...e the </s>'], 
        11: ['delete the given for...e the </s>'], 
        12: ['get the given for th...e the </s>'],
        13: ['write the given for ...e the </s>'], ...
    }
    """
    for key in reference_list.keys():
        _precision, _recall, _f1 = F1().compute_eval_score(hypothesis_list[key][0], reference_list[key])
        # update() - updates the dictionary with the element with other
        precision.update(_precision)
        recall.update(_recall)
        f1.update(_f1)
    
    return bleu, rouge_l, meteor, precision.avg, recall.avg, f1.avg