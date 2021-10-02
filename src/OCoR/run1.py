import tensorflow
from dataProcess import *
import os
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"]="1"
def train():
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    ds = DataSet(args)
    vds = DataSet(args, "val")
    args.Vocsize = len(ds.Char_Voc)
    args.Nl_Vocsize = len(ds.Nl_Voc)
    args.Code_Vocsize = len(ds.Code_Voc)
    Model = ModelWrapper(config)
    Model.load_checkpoint()
    MaxMrr = 0.53
    writer = tf.summary.FileWriter('logs', graph=Model.model.graph)
    global_step = 0
    f = open("bestresult.txt", "w")
    for i in range(10000000):
        num = 0
        #eval(Model, vds)
        for dBatch in tqdm(ds.Get_Train(args.batch_size)):
            num += 1
            global_step += 1
            feed_dic = {
                Model.model.inputNl:dBatch[0],
                Model.model.inputNlChar:dBatch[1],
                Model.model.inputCode:dBatch[2],
                Model.model.inputCodeChar:dBatch[3],
                Model.model.inputCodeNeg:dBatch[6],
                #Model.model.inputCodeCharNeg:dBatch[5],
                Model.model.inputNl_Overlap:dBatch[4],
                #Model.model.inputNl_Overlap_Neg:getOverlap(dBatch[0], dBatch[4]),
                Model.model.inputCode_Overlap:dBatch[5],
                #Model.model.inputCode_Overlap_Neg:getOverlap(dBatch[4], dBatch[0]),
                Model.model.keep_prob:0.8
            }
            Model.model.optim.run(session=Model.sess, feed_dict = feed_dic)
            if num % 10 == 0:
                s = Model.model.merge.eval(session=Model.sess, feed_dict=feed_dic)
                writer.add_summary(s, global_step)
                writer.flush()
        #eval
        if i % 3 == 0:
            print("eval")
            mrr = eval(Model, vds)
            print("print current mrr " + str(mrr))
            if mrr > MaxMrr:
                print("find better mrr " + str(mrr))
                MaxMrr = mrr
                Model.save_checkpoint()
                f.write(str(mrr))
                f.write("\n")
                f.flush()
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
    index = 0
    for dBatch in tqdm(vds.Get_Train(args.poolsize, "val")):
        tmpa = []
        for i in range(7):
            tmpa.append([])
        for j in tqdm(range(len(dBatch[0]))):
            tmpBatch = np.array([dBatch[0][j]] * len(dBatch[0]))
            tmpBatchChar = np.array([dBatch[1][j]] * len(dBatch[0]))
            tmpa[0].append(tmpBatch)
            tmpa[1].append(tmpBatchChar)
            tmpa[2].append(dBatch[2])
            tmpa[3].append(dBatch[3])
            tmp1 = []
            tmp2 = []
            for k in range(len(dBatch[0])):
                a, b = vds.get_overlap_indices(vds.Nls[index*50+j], vds.Codes[index*50 + k])
                tmp1.append(np.array(a, np.int32))
                tmp2.append(np.array(b, np.int32))
            tmpa[4].append(tmp1)
            tmpa[5].append(tmp2)
            tmpa[6].append(dBatch[6])
            if j % 10 == 9:
                for i in range(len(tmpa)):
                    tmpa[i] = np.concatenate(tmpa[i], axis = 0)
                feed_dic = {
                    Model.model.inputNl: tmpa[0],
                    Model.model.inputNlChar: tmpa[1],
                    Model.model.inputCode: tmpa[2],
                    Model.model.inputCodeChar: tmpa[3],
                    Model.model.inputCodeNeg: tmpa[6],
                    #Model.model.inputCodeCharNeg: dBatch[5],
                    Model.model.inputNl_Overlap: tmpa[4],
                    #Model.model.inputNl_Overlap_Neg: getOverlap(tmpBatch, dBatch[4]),
                    Model.model.inputCode_Overlap: tmpa[5],
                    #Model.model.inputCode_Overlap_Neg: getOverlap(dBatch[4], tmpBatch),
                    Model.model.keep_prob: 1.0
                }
                res = Model.model.result.eval(session=Model.sess, feed_dict=feed_dic)
                tmpans = []
                for i in range(10):
                    tmpans.append(res[50 * i: 50 * (i + 1),1:])
                for k, res in enumerate(tmpans):
                    res = np.max(res, axis=-1)
                    negres = np.negative(res)
                    predict = np.argsort(negres)
                    for i, t in enumerate(predict):
                        if t == j - (9 - k):
                            restotal.append(1 / (i + 1))
                tmpa = []
                for i in range(7):
                    tmpa.append([])
        index += 1
    return np.mean(restotal)
def evalMrr(Model, ds):
    resTotal = []
    ins = 0
    for dBatch in tqdm(ds.Get_Train(50, "dev")):
        feed_dic = {
            Model.model.inputNl:dBatch[0],
            Model.model.inputNlChar:dBatch[1],
            Model.model.inputCode:dBatch[2],
            Model.model.inputCodeChar:dBatch[3],
            Model.model.inputCodeNeg:dBatch[6],
            #Model.model.inputCodeCharNeg:dBatch[5],
            Model.model.inputNl_Overlap:dBatch[4],
            #Model.model.inputNl_Overlap_Neg:getOverlap(dBatch[0], dBatch[4]),
            Model.model.inputCode_Overlap:dBatch[5],
            #Model.model.inputCode_Overlap_Neg:getOverlap(dBatch[4], dBatch[0]),
            Model.model.keep_prob:1.0
        }
        res = Model.model.result.eval(session=Model.sess, feed_dict=feed_dic)
        #print(res)
        res = np.max(res[:,1:], axis=-1)
        negres = np.negative(res)
        predict = np.argsort(negres)
        for i, t in enumerate(predict):
            if t == 0:
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
