from typing import *
import tensorflow
from .dataProcess import *
import os
from tqdm import tqdm

unk = 'unk'
char_unk = 'Unknown'

def train():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    ds = DataSet(args)
    vds = DataSet(args, "val")
    args.Vocsize = len(ds.Char_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Code_Vocsize = len(ds.Code_Voc)
    Model = ModelWrapper(config)
    try:
        Model.load_checkpoint()
        print("load model successfully!!!")
    except:
        print("FAIL!!!")
        pass
    sys.stdout.flush()

    MaxMrr = 0
    writer = tf.summary.FileWriter('logs', graph=Model.model.graph)
    global_step = 0
    for i in range(10000000):
        num = 0
        # eval
        if i % 10 == 9:
            Model.save_checkpoint()
            print("eval")
            mrr = eval(Model, vds)["mrr"]
            print("print current mrr " + str(mrr))
            if mrr > MaxMrr:
                print("find better mrr " + str(mrr))
                MaxMrr = mrr
        #eval(Model, vds)
        for dBatch in tqdm(ds.Get_Train(args.batch_size)):
            num += 1
            global_step += 1
            feed_dic = {
                Model.model.inputNl: dBatch[0],
                Model.model.inputNlChar: dBatch[1],
                Model.model.inputCode: dBatch[2],
                Model.model.inputCodeChar: dBatch[3],
                Model.model.inputCodeNeg: dBatch[6],
                # Model.model.inputCodeCharNeg:dBatch[5],
                Model.model.inputNl_Overlap: dBatch[4],
                # Model.model.inputNl_Overlap_Neg:getOverlap(dBatch[0], dBatch[4]),
                Model.model.inputCode_Overlap: dBatch[5],
                # Model.model.inputCode_Overlap_Neg:getOverlap(dBatch[4], dBatch[0]),
                Model.model.keep_prob: 0.8
            }
            Model.model.optim.run(session=Model.sess, feed_dict=feed_dic)
            if num % 10 == 0:
                s = Model.model.merge.eval(
                    session=Model.sess, feed_dict=feed_dic)
                writer.add_summary(s, global_step)
                writer.flush()
    writer.close()


def getOverlap(Aseq, Bseq):
    Aseq = np.array(Aseq, np.int32)
    Bseq = np.array(Bseq, np.int32)
    mask = np.not_equal(Aseq, 0).astype(np.int32)
    mask_N = np.equal(Aseq, 0).astype(np.int32)
    Astack = np.stack([Aseq] * len(Bseq[0]), axis=-1)
    Bstack = np.stack([Bseq] * len(Aseq[0]), axis=1)

    subStack = Astack - Bstack
    equalResult = np.equal(subStack, 0)
    equalResult = equalResult.astype(np.int32)
    equalResult = np.max(equalResult, axis=-1)
    equalResult *= mask
    equalResult += 2 * mask_N
    return equalResult


def eval(Model, vds):
    restotal = []
    r1s = 0
    r5s = 0
    r10s = 0
    index = 0
    # wf = open("res.txt", "w")
    # f = open("resD.txt", "a")
    data_lolder = tqdm(vds.Get_Train(args.poolsize, "val"),
                       total=len(vds.data[0]) // args.poolsize)

    if args.batch_size > args.poolsize:
        t = args.batch_size // args.poolsize
        for dBatch in data_lolder:
            tmpa = []
            for i in range(7):
                tmpa.append([])
            for j in tqdm(range(dBatch[0].shape[0]), leave=False):
                tmpBatch = np.repeat(
                    [dBatch[0][j]], dBatch[0].shape[0], axis=0)
                tmpBatchChar = np.repeat(
                    [dBatch[1][j]], dBatch[0].shape[0], axis=0)
                tmpa[0].append(tmpBatch)
                tmpa[1].append(tmpBatchChar)
                tmpa[2].append(dBatch[2])
                tmpa[3].append(dBatch[3])
                tmp1 = []
                tmp2 = []
                for k in range(dBatch[0].shape[0]):
                    a, b = vds.get_overlap_indices(
                        vds.Nls[index*args.poolsize+j], vds.Codes[index*args.poolsize + k])
                    tmp1.append(np.array(a, np.int32))
                    tmp2.append(np.array(b, np.int32))
                tmpa[4].append(tmp1)
                tmpa[5].append(tmp2)
                tmpa[6].append(dBatch[6])
                if j % t == t - 1:
                    for i in range(len(tmpa)):
                        tmpa[i] = np.concatenate(tmpa[i], axis=0)
                    feed_dic = {
                        Model.model.inputNl: tmpa[0],
                        Model.model.inputNlChar: tmpa[1],
                        Model.model.inputCode: tmpa[2],
                        Model.model.inputCodeChar: tmpa[3],
                        Model.model.inputCodeNeg: tmpa[6],
                        # Model.model.inputCodeCharNeg: dBatch[5],
                        Model.model.inputNl_Overlap: tmpa[4],
                        # Model.model.inputNl_Overlap_Neg: getOverlap(tmpBatch, dBatch[4]),
                        Model.model.inputCode_Overlap: tmpa[5],
                        # Model.model.inputCode_Overlap_Neg: getOverlap(dBatch[4], tmpBatch),
                        Model.model.keep_prob: 1.0
                    }
                    res = Model.model.result.eval(
                        session=Model.sess, feed_dict=feed_dic)
                    tmpans = []
                    # for x in res:
                    #     f.write(str(x[1]) + "\n")
                    #     f.flush()
                    for i in range(t):
                        tmpans.append(
                            res[args.poolsize * i: args.poolsize * (i + 1), 1:])
                    for k, res in enumerate(tmpans):
                        res = np.max(res, axis=-1)
                        negres = np.negative(res)
                        predict = np.argsort(negres)
                        i = np.where(predict == j - (t - 1 - k))[0][0]
                        restotal.append(1 / (i + 1))
                    tmpa = []
                    for i in range(7):
                        tmpa.append([])
                data_lolder.set_description(
                    'mrr:{}'.format(np.mean(restotal[-50:])))
            index += 1
    else:
        t = args.poolsize // args.batch_size
        for dBatch in data_lolder:
            tmpa = []
            for i in range(7):
                tmpa.append([])
            for j in tqdm(range(dBatch[0].shape[0]), leave=False):
                tmpans = []
                processed = 0
                while processed < args.poolsize:
                    processs_count = min(
                        args.batch_size, args.poolsize - processed)
                    processed += processs_count
                    tmpa = []
                    for i in range(7):
                        tmpa.append([])
                    tmpBatch = np.repeat(
                        [dBatch[0][j]], processs_count, axis=0)
                    tmpBatchChar = np.repeat(
                        [dBatch[1][j]], processs_count, axis=0)
                    tmpa[0].append(tmpBatch)
                    tmpa[1].append(tmpBatchChar)
                    tmpa[2].append(
                        dBatch[2][processed - processs_count:processed])
                    tmpa[3].append(
                        dBatch[3][processed - processs_count:processed])
                    tmp1 = []
                    tmp2 = []
                    for k in range(processs_count):
                        a, b = vds.get_overlap_indices(
                            vds.Nls[index*args.poolsize+j], vds.Codes[index*args.poolsize + k])
                        tmp1.append(np.array(a, np.int32))
                        tmp2.append(np.array(b, np.int32))
                    tmpa[4].append(tmp1)
                    tmpa[5].append(tmp2)
                    tmpa[6].append(
                        dBatch[6][processed - processs_count:processed])
                    for i in range(len(tmpa)):
                        tmpa[i] = np.concatenate(tmpa[i], axis=0)
                    feed_dic = {
                        Model.model.inputNl: tmpa[0],
                        Model.model.inputNlChar: tmpa[1],
                        Model.model.inputCode: tmpa[2],
                        Model.model.inputCodeChar: tmpa[3],
                        Model.model.inputCodeNeg: tmpa[6],
                        # Model.model.inputCodeCharNeg: dBatch[5],
                        Model.model.inputNl_Overlap: tmpa[4],
                        # Model.model.inputNl_Overlap_Neg: getOverlap(tmpBatch, dBatch[4]),
                        Model.model.inputCode_Overlap: tmpa[5],
                        # Model.model.inputCode_Overlap_Neg: getOverlap(dBatch[4], tmpBatch),
                        Model.model.keep_prob: 1.0
                    }
                    res = Model.model.result.eval(
                        session=Model.sess, feed_dict=feed_dic)
                    # for x in res:
                    #     f.write(str(x[1]) + "\n")
                    #     f.flush()
                    tmpans.append(res[:, 1:])
                tmpans = np.vstack(tmpans)
                res = np.max(tmpans, axis=-1)
                negres = np.negative(res)
                predict = np.argsort(negres)
                i = np.where(predict == j)[0][0]
                if i == 0:
                    r1s += 1
                if i < 5:
                    r5s += 1
                if i < 10:
                    r10s += 1
                restotal.append(1 / (i + 1))
                data_lolder.set_description(
                    'mrr:{}'.format(np.mean(restotal[-50:])))
            index += 1
    return {"mrr": np.mean(restotal), "r1": r1s / len(restotal), "r5": r5s / len(restotal), "r10": r10s / len(restotal)}


def eval_for_segments(
    Model,
    vds,
    poolsize,
    valid_segments: Union[
        List[
            Tuple[
                List,   # codeVec
                List,   # descVec
            ]
        ],
        List[
            Tuple[
                List,   # codeVec
                List,   # descVec
                List,   # queryVec
            ]
        ]
    ],
    alpha=1.0,
        verbose=False):
    restotal = []
    r1s = 0
    r5s = 0
    r10s = 0

    index = 0
    # wf = open("res.txt", "w")
    # f = open("resD.txt", "a")

    assert poolsize >= args.batch_size
    assert poolsize >= len(valid_segments)

    codeDict = {vds.Code_Voc[i]: i for i in vds.Code_Voc}
    descDict = {vds.Nl_Voc[i]: i for i in vds.Nl_Voc}

    codeVec = [vds.pad_seq(list(i[0]), vds.Code_Len)[0]
               for i in valid_segments]
    descVec = [vds.pad_seq(list(i[1]), vds.Nl_Len)[0] for i in valid_segments]
    codeWords = [[codeDict[i] if i in codeDict else unk for i in j]
                 for j in codeVec]
    descWords = [[descDict[i] if i in descDict else unk for i in j]
                 for j in descVec]
    codeChar = [[vds.pad_seq([vds.Char_Voc[j] if j in vds.Char_Voc else vds.Char_Voc[char_unk]
                              for j in i], vds.Char_Len)[0] for i in _i] for _i in codeWords]
    descChar = [[vds.pad_seq([vds.Char_Voc[j] if j in vds.Char_Voc else vds.Char_Voc[char_unk]
                              for j in i], vds.Char_Len)[0] for i in _i] for _i in descWords]

    if len(valid_segments[0]) == 3 and alpha != 1.0:
        queryVec = [vds.pad_seq(list(i[2]), vds.Nl_Len)[0]
                    for i in valid_segments]
        queryWords = [
            [descDict[i] if i in descDict else unk for i in j] for j in queryVec]
        queryChar = [[vds.pad_seq([vds.Char_Voc[j] if j in vds.Char_Voc else vds.Char_Voc[char_unk]
                                   for j in i], vds.Char_Len)[0] for i in _i] for _i in queryWords]
    else:
        queryVec = queryChar = []

    for dBatch in vds.Get_Train(poolsize, "val"):
        tmpa = []
        descVec = np.vstack([descVec, dBatch[0]])[: poolsize]
        descChar = np.vstack([descChar, dBatch[1]])[: poolsize]
        codeVec = np.vstack([codeVec, dBatch[2]])[: poolsize]
        codeChar = np.vstack([codeChar, dBatch[3]])[: poolsize]
        if len(valid_segments[0]) == 3 and alpha != 1.0:
            queryVec = np.vstack([queryVec, dBatch[0]])[: poolsize]
            queryChar = np.vstack([queryChar, dBatch[1]])[: poolsize]

        Nls = (descWords + vds.Nls)[: poolsize]
        Codes = (codeWords + vds.Codes)[: poolsize]
        for i in range(7):
            tmpa.append([])
        if verbose:
            r = tqdm(range(len(valid_segments)))
        else:
            r = range(len(valid_segments))
        for j in r:
            tmpans = []
            processed = 0
            while processed < poolsize:
                processs_count = min(args.batch_size, poolsize - processed)
                processed += processs_count
                res = [None, None]
                if len(valid_segments[0]) == 3 and alpha != 1.0:
                    c = [(descVec, descChar), (queryVec, queryChar)]
                else:
                    c = [(descVec, descChar)]
                for x, (dv, dc) in enumerate(c):
                    tmpa = []
                    for i in range(7):
                        tmpa.append([])
                    tmpBatch = np.repeat([dv[j]], processs_count, axis=0)
                    tmpBatchChar = np.repeat([dc[j]], processs_count, axis=0)
                    tmpa[0].append(tmpBatch)
                    tmpa[1].append(tmpBatchChar)
                    tmpa[2].append(
                        codeVec[processed - processs_count:processed])
                    tmpa[3].append(
                        codeChar[processed - processs_count:processed])
                    tmp1 = []
                    tmp2 = []
                    for k in range(processs_count):
                        a, b = vds.get_overlap_indices(
                            Nls[index*poolsize+j], Codes[index*poolsize + k])
                        tmp1.append(np.array(a, np.int32))
                        tmp2.append(np.array(b, np.int32))
                    tmpa[4].append(tmp1)
                    tmpa[5].append(tmp2)
                    tmpa[6].append(
                        dBatch[6][processed - processs_count:processed])
                    for i in range(len(tmpa)):
                        tmpa[i] = np.concatenate(tmpa[i], axis=0)
                    feed_dic = {
                        Model.model.inputNl: tmpa[0],
                        Model.model.inputNlChar: tmpa[1],
                        Model.model.inputCode: tmpa[2],
                        Model.model.inputCodeChar: tmpa[3],
                        Model.model.inputCodeNeg: tmpa[6],
                        # Model.model.inputCodeCharNeg: dBatch[5],
                        Model.model.inputNl_Overlap: tmpa[4],
                        # Model.model.inputNl_Overlap_Neg: getOverlap(tmpBatch, dBatch[4]),
                        Model.model.inputCode_Overlap: tmpa[5],
                        # Model.model.inputCode_Overlap_Neg: getOverlap(dBatch[4], tmpBatch),
                        Model.model.keep_prob: 1.0
                    }
                    res[x] = Model.model.result.eval(
                        session=Model.sess, feed_dict=feed_dic)
                    # for x in res:
                    #     f.write(str(x[1]) + "\n")
                    #     f.flush()
                if len(valid_segments[0]) == 3 and alpha != 1.0:
                    res = alpha * \
                        res[0].astype('float') + (1 - alpha) * \
                        res[1].astype('float')
                else:
                    res = res[0]
                tmpans.append(res[:, 1:])
            tmpans = np.vstack(tmpans)
            res = np.max(tmpans, axis=-1)
            negres = np.negative(res)
            predict = np.argsort(negres)
            i = np.where(predict == j)[0][0]
            if i == 0:
                r1s += 1
            if i < 5:
                r5s += 1
            if i < 10:
                r10s += 1
            restotal.append(1 / (i + 1))
            if verbose:
                r.set_description('mrr:{}'.format(np.mean(restotal[-50:])))
        break
    return {"mrrs": restotal, "mrr": np.mean(restotal), "r1": r1s / len(valid_segments), "r5": r5s / len(valid_segments), "r10": r10s / len(valid_segments)}


ds_easy_to_use = None
Model_easy_to_use = None


def eval_for_segments_easy_to_use(
    poolsize,
    valid_segments: Union[
        List[
            Tuple[
                List,   # codeVec
                List,   # descVec
            ]
        ],
        List[
            Tuple[
                List,   # codeVec
                List,   # descGenVec
                List,   # queryVec
            ]
        ]
    ],
    alpha=1.0,
        verbose=False):
    global ds_easy_to_use
    global Model_easy_to_use
    if ds_easy_to_use is None:
        ds_easy_to_use = DataSet(args, "val_s")
    args.Vocsize = len(ds_easy_to_use.Char_Voc)
    args.Nl_Vocsize = len(ds_easy_to_use.Nl_Voc)
    args.Code_Vocsize = len(ds_easy_to_use.Code_Voc)
    if Model_easy_to_use is None:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(
            allow_soft_placement=True, gpu_options=gpu_options)
        Model_easy_to_use = ModelWrapper(config)
        Model_easy_to_use.load_checkpoint()
    return eval_for_segments(Model_easy_to_use, ds_easy_to_use, poolsize, valid_segments, alpha=alpha, verbose=verbose)


def evalMrr(Model, ds):
    f = open("res.txt", "w")
    wf = open("resP.txt", "a")
    resTotal = []
    ins = 0
    for dBatch in tqdm(ds.Get_Train(50, "dev")):
        feed_dic = {
            Model.model.inputNl: dBatch[0],
            Model.model.inputNlChar: dBatch[1],
            Model.model.inputCode: dBatch[2],
            Model.model.inputCodeChar: dBatch[3],
            Model.model.inputCodeNeg: dBatch[6],
            # Model.model.inputCodeCharNeg:dBatch[5],
            Model.model.inputNl_Overlap: dBatch[4],
            # Model.model.inputNl_Overlap_Neg:getOverlap(dBatch[0], dBatch[4]),
            Model.model.inputCode_Overlap: dBatch[5],
            # Model.model.inputCode_Overlap_Neg:getOverlap(dBatch[4], dBatch[0]),
            Model.model.keep_prob: 1.0
        }
        res = Model.model.result.eval(session=Model.sess, feed_dict=feed_dic)
        # print(res)
        res = np.max(res[:, 1:], axis=-1)
        negres = np.negative(res)
        predict = np.argsort(negres)
        for x in res:
            wf.write(str(x))
            wf.write("\n")
            wf.flush()
        for i, t in enumerate(predict):
            if t == 0:
                f.write(str(i) + "\n")
                f.flush()
                resTotal.append(1 / (i + 1))
    return np.mean(resTotal)


def test():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    ds = DataSet(args, "test")
    args.Vocsize = len(ds.Char_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Code_Vocsize = len(ds.Code_Voc)
    Model = ModelWrapper(config)
    Model.load_checkpoint()
    res = eval(Model, ds)
    print("mrr is " + str(res))


def evals():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    ds = DataSet(args, "eval")
    args.Vocsize = len(ds.Char_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Code_Vocsize = len(ds.Code_Voc)
    Model = ModelWrapper(config)
    Model.load_checkpoint()
    res = evalMrr(Model, ds)
    print("mrr is " + str(res))


def dev():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    ds = DataSet(args, "dev")
    args.Vocsize = len(ds.Char_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Code_Vocsize = len(ds.Code_Voc)
    Model = ModelWrapper(config)
    Model.load_checkpoint()
    res = evalMrr(Model, ds)
    print("mrr is " + str(res))


if __name__ == '__main__':
    mode = input("train or test: \n")
    if mode == "train":
        train()
    elif mode == "test":
        test()
    elif mode == "eval":
        evals()
    elif mode == "dev":
        dev()
