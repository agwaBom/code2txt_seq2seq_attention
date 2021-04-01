from collections import OrderedDict, Counter


class F1():
    def __init__(self):
        pass

    def normalize_answer(self, s):
        """Lower text and remove extra whitespace"""
        def white_space_fix(text):
            return ' '.join(text.split())
        return white_space_fix(s.lower())

    def eval_score(self, prediction, ground_truth):
        # True - 모델이 실제로 맞춤, False - 모델이 틀림
        # Positive - 맞다고 예측, Negative - 틀렸다고 예측

        # Precision = TP / TP + FP
        # Recall    = TP / TP + FN
        # F1        = Precision * Recall / Precision + Recall
        precision, recall, f1 = 0, 0, 0

        # if there is no ground truth
        if len(ground_truth) == 0:
            if len(prediction) == 0:
                precision, recall, f1 = 1, 1, 1
        else:
            prediction_tokens = self.normalize_answer(prediction).split()
            ground_truth_tokens = self.normalize_answer(ground_truth).split()
            # common = if both token has same alphabet it gets in
            # token 단위가 단어라면 단어가 맞아야지만 들어가겠지?
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            # 총 몇 개를 맞췄냐.
            num_same = sum(common.values())

            if num_same != 0:
                # 맞다고 예측한 것들중 실제 맞은거
                precision = 1.0 * num_same / len(prediction_tokens)
                # 실제 정답이 True인 것중 모델이 얼마나 맞췄냐.
                recall = 1.0 * num_same / len(ground_truth_tokens)
                # 왜 2를 곱했는지...
                f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def compute_eval_score(self, prediction, ground_truth_list):
        assert isinstance(prediction, str)
        precision, recall, f1 = 0, 0, 0
        for ground_truth in ground_truth_list:
            _precision, _recall, _f1 = self.eval_score(prediction, ground_truth)
            # 이전 f1보다 새 f1이 더 크면 다 갈아 엎어!!!
            if _f1 > f1:
                precision, recall, f1 = _precision, _recall, _f1
        return precision, recall, f1