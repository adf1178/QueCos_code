import sys
import numpy as np
from ... import lib

# import code_retrieval
# cr = code_retrieval.CrCritic()

# a few configs to set up in a2c-train.py
cal_mode_train = "sentence"  # sample a pool or use the batch as a pool
cal_mode_eval = "batch"
reward_lambda = 0.75
reward_gamma = 1.0


def empty_check(ids):
    return len(ids) == 0


def clean_up_ids(noisy_ids):
    # if lib.Constants.EOS in noisy_ids:
    #     noisy_ids = noisy_ids[:noisy_ids.index(lib.Constants.EOS)]

    # if empty_check(noisy_ids):
    #     return None
    # if noisy_ids[0] == lib.Constants.BOS:
    #     noisy_ids = noisy_ids[1:]

    # if empty_check(noisy_ids):
    #     return np.array([])
    # while noisy_ids[-1] == lib.Constants.PAD:
    #     noisy_ids = noisy_ids[:-1]
    #     if empty_check(noisy_ids):
    #         return None

    return noisy_ids


def clean_up_ids_list(list_of_noisy_ids):
    fed_ids_list, cleaned_ids_list = [], []
    for ids in list_of_noisy_ids:
        length = len(ids)
        clean_ids = clean_up_ids(ids)
        if clean_ids is None:
            fed_ids_list.append(np.array([lib.Constants.PAD] * length))
            cleaned_ids_list.append(
                [lib.Constants.EOS] + [lib.Constants.PAD] * (length - 1))
        else:
            fed_ids_list.append(clean_ids)
            clean_ids = list(clean_ids)
            if len(clean_ids) < length:
                clean_ids += [lib.Constants.EOS] + \
                    [lib.Constants.PAD] * (length - len(clean_ids) - 1)
            cleaned_ids_list.append(clean_ids)
    return fed_ids_list, cleaned_ids_list


#############################################################
unif_data = {}


def sentence_retrieval_unif_mrr(data_name, code, des_gen, qt, number_of_runs=1):
    if len(unif_data) == 0:
        from ....UNIF.validate import validate_for_segments, model, config
        unif_data["validate_for_segments"] = validate_for_segments
        unif_data["model"] = model
        unif_data["config"] = config
    length = len(des_gen)
    code, des_gen, qt = clean_up_ids(
        code), clean_up_ids(des_gen), clean_up_ids(qt)
    if des_gen is None:
        des_gen = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + \
            [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(des_gen)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + \
                [lib.Constants.PAD] * (length - len(des_gen) - 1)

    result = unif_data["validate_for_segments"]([(np.array(code), len(code), np.array(
        des_gen), len(des_gen))], unif_data["model"], 1000, 1, unif_data["config"]['sim_measure'])
    mrr = result['mrr']
    print(mrr)

    return mrr, cleaned_annotation


def batch_retrieval_unif_mrr(codes, des_gens, qts):
    if len(unif_data) == 0:
        from ....UNIF.validate import validate_for_segments, model, config
        unif_data["validate_for_segments"] = validate_for_segments
        unif_data["model"] = model
        unif_data["config"] = config
    fed_codes, _ = clean_up_ids_list(codes)
    fed_annotations, cleaned_annotations = clean_up_ids_list(des_gens)
    fed_qts, _ = clean_up_ids_list(qts)

    assert len(codes) == len(des_gens)

    data = []
    for i in range(len(codes)):
        code = codes[i]
        des_gen = des_gens[i]
        data.append((np.array(code), len(code),
                     np.array(des_gen), len(des_gen)))

    result = unif_data["validate_for_segments"](
        data, unif_data["model"], 1000, 1, unif_data["config"]['sim_measure'])
    mrrs = result['mrrs']

    return mrrs, cleaned_annotations


def retrieval_unif_mrr_train(annotations, qts, codes, **kwargs):
    if cal_mode_train == "sentence":
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt in zip(codes, annotations, qts):
            mrr, cleaned_annotation = sentence_retrieval_unif_mrr("train", code, annotation, qt,
                                                                  number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_unif_mrr_eval(annotations, qts, codes, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_unif_mrr(
        codes, annotations, qts)

    return mrrs, cleaned_annotations


####################################################
ocor_data = {}


def sentence_retrieval_ocor_mrr(data_name, code, des_gen, qt, number_of_runs=1):
    if len(ocor_data) == 0:
        from ....OCoR.run import eval_for_segments_easy_to_use
        ocor_data["eval_for_segments_easy_to_use"] = eval_for_segments_easy_to_use
    length = len(des_gen)
    code, des_gen = clean_up_ids(code), clean_up_ids(des_gen)
    if des_gen is None:
        des_gen = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + \
            [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(des_gen)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + \
                [lib.Constants.PAD] * (length - len(des_gen) - 1)
    des_gen = np.array(des_gen[1:])
    des_gen[des_gen == lib.data.Constants.EOS] = lib.data.Constants.PAD
    des_gen[des_gen == 3] = 1
    result = ocor_data["eval_for_segments_easy_to_use"](
        1000, [(list(code), list(des_gen))])
    mrr = result['mrr']
    print(mrr)
    sys.stdout.flush()

    return mrr, cleaned_annotation


def batch_retrieval_ocor_mrr(codes, des_gens, qts):
    if len(ocor_data) == 0:
        from ....OCoR.run import eval_for_segments_easy_to_use
        ocor_data["eval_for_segments_easy_to_use"] = eval_for_segments_easy_to_use
    fed_codes, _ = clean_up_ids_list(codes)
    fed_annotations, cleaned_annotations = clean_up_ids_list(des_gens)
    fed_qts, _ = clean_up_ids_list(qts)

    assert len(codes) == len(des_gens)

    data = []
    for i in range(len(codes)):
        code = codes[i]
        des_gen = des_gens[i]
        des_gen = np.array(des_gen[1:])
        des_gen[des_gen == lib.data.Constants.EOS] = lib.data.Constants.PAD
        des_gen[des_gen == 3] = 1
        data.append((list(code), list(des_gen)))

    result = ocor_data["eval_for_segments_easy_to_use"](1000, data)
    mrrs = result['mrrs']

    return mrrs, cleaned_annotations


def retrieval_ocor_mrr_train(annotations, qts, codes, **kwargs):
    if cal_mode_train == "sentence":
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt in zip(codes, annotations, qts):
            mrr, cleaned_annotation = sentence_retrieval_ocor_mrr("train", code, annotation, qt,
                                                                  number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_ocor_mrr_eval(annotations, qts, codes, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_ocor_mrr(
        codes, annotations, qts)

    return mrrs, cleaned_annotations


#############################################################
deepcs_data = {}


def sentence_retrieval_deepcs_mrr(data_name, code, des_gen, qt, name, number_of_runs=1):
    if len(deepcs_data) == 0:
        from ....deepcs.validate import validate_for_segments, model, config
        deepcs_data["validate_for_segments"] = validate_for_segments
        deepcs_data["model"] = model
        deepcs_data["config"] = config
    length = len(des_gen)
    if des_gen is None:
        des_gen = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + \
            [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(des_gen)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + \
                [lib.Constants.PAD] * (length - len(des_gen) - 1)

    result = deepcs_data["validate_for_segments"]([(np.array(name), len(name), np.array(code), len(code), np.array(
        des_gen), len(des_gen))], deepcs_data["model"], 1000, 1, deepcs_data["config"]['sim_measure'])
    mrr = result['mrr']
    print(mrr)

    return mrr, cleaned_annotation


def batch_retrieval_deepcs_mrr(codes, des_gens, qts, names):
    if len(deepcs_data) == 0:
        from ....deepcs.validate import validate_for_segments, model, config
        deepcs_data["validate_for_segments"] = validate_for_segments
        deepcs_data["model"] = model
        deepcs_data["config"] = config
    fed_annotations, cleaned_annotations = clean_up_ids_list(des_gens)

    assert len(codes) == len(des_gens)

    data = []
    for i in range(len(codes)):
        name = names[i]
        code = codes[i]
        des_gen = des_gens[i]
        data.append((np.array(name), len(name),
                     np.array(code), len(code),
                     np.array(des_gen), len(des_gen)))

    result = deepcs_data["validate_for_segments"](
        data, deepcs_data["model"], 1000, 1, deepcs_data["config"]['sim_measure'])
    mrrs = result['mrrs']

    return mrrs, cleaned_annotations


def retrieval_deepcs_mrr_train(annotations, qts, codes, names, **kwargs):
    if cal_mode_train == "sentence":
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt, name in zip(codes, annotations, qts, names):
            mrr, cleaned_annotation = sentence_retrieval_deepcs_mrr("train", code, annotation, qt, name,
                                                                    number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_deepcs_mrr_eval(annotations, qts, codes, names, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_deepcs_mrr(
        codes, annotations, qts, names)

    return mrrs, cleaned_annotations


#############################################################
def unif_mix_train(annotations, qts, codes, tgt_dict, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_unif_mrr_train(
        annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations


def unif_mix_eval(annotations, qts, codes, tgt_dict, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_unif_mrr_eval(
        annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations

#############################################################


def ocor_mix_train(annotations, qts, codes, tgt_dict, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_ocor_mrr_train(
        annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations


def ocor_mix_eval(annotations, qts, codes, tgt_dict, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_ocor_mrr_eval(
        annotations, qts, codes)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations


#############################################################


def deepcs_mix_train(annotations, qts, codes, tgt_dict, names, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_deepcs_mrr_train(
        annotations, qts, codes, names)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations


def deepcs_mix_eval(annotations, qts, codes, tgt_dict, names, data_name=None, indices=None, **kwargs):
    mrrs, cleaned_annotations = retrieval_deepcs_mrr_eval(
        annotations, qts, codes, names)
    bleus, _ = lib.Reward.wrapped_sentence_bleu(
        annotations, qts, tgt_dict, data_name=data_name, indices=indices)
    rewards = [mrr*reward_lambda + bleu *
               (1-reward_lambda)for mrr, bleu in zip(mrrs, bleus)]

    return rewards, cleaned_annotations

####################################################
code_bert_data = {
    'data_name': 'data-python4csn'
}

def init_code_cert():
    if len(code_bert_data) == 1:
        import os
        import json
        from ....CodeBERT.CodeBERT.codesearch.demo.demo import eval_easy_to_use
        code_bert_data["eval_easy_to_use"] = eval_easy_to_use
        with open(os.path.join(os.path.dirname(__file__), '../../../../{}/descri.json'.format(code_bert_data['data_name']))) as f:
            labelToIdx = json.loads(f.read())
            code_bert_data["discri_dict"] = {labelToIdx[i]: i for i in labelToIdx}
        with open(os.path.join(os.path.dirname(__file__), '../../../../{}/token.json'.format(code_bert_data['data_name']))) as f:
            labelToIdx = json.loads(f.read())
            code_bert_data["code_dict"] = {labelToIdx[i]: i for i in labelToIdx}

def sentence_retrieval_code_bert_mrr(data_name, code, des_gen, qt, raw_code, number_of_runs=1):
    if len(code_bert_data) == 1:
        init_code_cert()
    length = len(des_gen)
    des_gen = clean_up_ids(des_gen)
    if des_gen is None:
        des_gen = [lib.Constants.PAD] * length
        cleaned_annotation = [lib.Constants.EOS] + \
            [lib.Constants.PAD] * (length - 1)
    else:
        cleaned_annotation = list(des_gen)
        if len(cleaned_annotation) < length:
            cleaned_annotation += [lib.Constants.EOS] + \
                [lib.Constants.PAD] * (length - len(des_gen) - 1)
    des_gen = np.array(des_gen[1:])
    # code = np.array(code[1:])
    des_gen[des_gen == lib.data.Constants.EOS] = lib.data.Constants.PAD
    # code[code == lib.data.Constants.EOS] = lib.data.Constants.PAD
    des_gen = ' '.join([code_bert_data["discri_dict"][i] if i != lib.data.Constants.PAD else '' for i in des_gen])
    # code = ' '.join([code_bert_data["code_dict"][i] if i != lib.data.Constants.PAD else '' for i in code])
    # print(des_gen)
    # print(code)
    result = code_bert_data["eval_easy_to_use"]([des_gen], [raw_code])
    mrr = result['mrr']
    print(mrr)
    sys.stdout.flush()

    return mrr, cleaned_annotation


def batch_retrieval_code_bert_mrr(codes, des_gens, raw_codes, qts):
    if len(code_bert_data) == 1:
        init_code_cert()
    # fed_codes, _ = clean_up_ids_list(codes)
    fed_annotations, cleaned_annotations = clean_up_ids_list(des_gens)
    fed_qts, _ = clean_up_ids_list(qts)

    assert len(raw_codes) == len(des_gens)

    _des_gens = []
    for i in range(len(raw_codes)):
        des_gen = des_gens[i]
        des_gen = np.array(des_gen[1:])
        des_gen[des_gen == lib.data.Constants.EOS] = lib.data.Constants.PAD
        des_gen = ' '.join([code_bert_data["discri_dict"][i] if i != lib.data.Constants.PAD else '' for i in des_gen])
        _des_gens.append(des_gen)

    result = code_bert_data["eval_easy_to_use"](_des_gens, raw_codes)
    mrrs = result['mrrs']

    return mrrs, cleaned_annotations


def retrieval_code_bert_mrr_train(annotations, qts, codes, raw_codes, **kwargs):
    if len(code_bert_data) == 1:
        init_code_cert()
    if cal_mode_train == "sentence":
        cleaned_annotations = []
        mrrs = []

        for code, annotation, qt, raw_code in zip(codes, annotations, qts, raw_codes):
            mrr, cleaned_annotation = sentence_retrieval_code_bert_mrr("train", code, annotation, qt, raw_code,
                                                                  number_of_runs=1)
            mrrs.append(mrr)
            cleaned_annotations.append(cleaned_annotation)
    else:
        raise Exception("Invalid cal_mode_train %s!" % cal_mode_train)

    return mrrs, cleaned_annotations


def retrieval_code_bert_mrr_eval(annotations, qts, codes, raw_codes, **kwargs):
    # no "sentence" cal_mode is supported
    mrrs, cleaned_annotations = batch_retrieval_code_bert_mrr(
        codes, annotations, raw_codes, qts)

    return mrrs, cleaned_annotations