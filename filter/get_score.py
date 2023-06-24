

def get_con_score(pred_by_origin, pred_by_augment, labels):
    scores = []
    for i, label in enumerate(labels):
        origin = pred_by_origin[i]
        augment = pred_by_augment[i]
        topk = int(sum(label))
        max_index = len(origin)
        if topk <= max_index:
            origin_top_k = origin[:topk]
            augment_top_k = augment[:topk]
        else:
            origin_top_k = origin
            augment_top_k = augment
        intersection = set(augment_top_k).intersection(origin_top_k)
        score = len(intersection)/topk
        scores.append(score)
    return scores


def get_overall_score(div_score, con_score):
    def norm(a):
        min_value = min(a)
        max_value = max(a)
        normalized = [(x - min_value) / (max_value - min_value) for x in a]
        return normalized

    div_score = norm(div_score)
    con_score = norm(con_score)

    return div_score + con_score
