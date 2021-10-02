from __future__ import division
import time
import random
from torch.autograd import Variable
import os.path
import numpy as np
import argparse
import torch
import torch.nn as nn
from . import lib
import os
import sys
import json
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)
sys.path.append("..")


def get_opt():
    parser = argparse.ArgumentParser(description='a2c-train.py')
    # Data options
    parser.add_argument('-data_name', default="",
                        help="Data name, such as toy")
    parser.add_argument('-save_dir', required=True,
                        help='Directory to save models')
    parser.add_argument("-load_from", help="Path to load a pretrained model.")
    parser.add_argument("-show_str", required=True,
                        help="string of arguments for saving models.")

    # Model options
    parser.add_argument('-model_name', type=str, default='Seq2Seq', choices=['Seq2Seq', 'Transformer'],
                        help='23333')
    parser.add_argument('-layers', type=int, default=1,
                        help='Number of layers in the LSTM encoder/decoder')
    parser.add_argument('-rnn_size', type=int, default=512,
                        help='Size of LSTM hidden states')
    parser.add_argument('-word_vec_size', type=int,
                        default=512, help='Word embedding sizes')
    parser.add_argument('-input_feed', type=int, default=1, help="""Feed the context vector at each time step as
                        additional input (via concatenation with the word embeddings) to the decoder.""")
    parser.add_argument('-brnn', action='store_true',
                        help='Use a bidirectional encoder')
    parser.add_argument('-brnn_merge', default='concat',
                        help="""Merge action for the bidirectional hidden states: [concat|sum]""")
    parser.add_argument('-has_attn', type=int, default=1,
                        help="""attn model or not""")
    parser.add_argument('-has_baseline', type=int,
                        default=1, help="baseline model")

    # Optimization options
    parser.add_argument('-batch_size', type=int,
                        default=128, help='Maximum batch size')
    parser.add_argument("-max_generator_batches", type=int, default=128,
                        help="""Split softmax input into small batches for memory efficiency. Higher is faster, but uses more memory.""")

    parser.add_argument("-end_epoch", type=int, default=50,
                        help="Epoch to stop training.")
    parser.add_argument("-start_epoch", type=int, default=1,
                        help="Epoch to start training.")

    parser.add_argument('-param_init', type=float, default=0.1,
                        help="""Parameters are initialized over uniform distribution with support (-param_init, param_init). Use 0 to not use initialization""")
    parser.add_argument('-optim', default='adam',
                        help="Optimization method. [sgd|adagrad|adadelta|adam]")
    parser.add_argument("-lr", type=float, default=1e-3,
                        help="Initial learning rate")
    parser.add_argument('-max_grad_norm', type=float, default=5, help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")
    parser.add_argument('-dropout', type=float, default=0.3,
                        help='Dropout probability; applied between LSTM stacks.')

    parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                        help="""If update_learning_rate, decay learning rate by
                        this much if (i) perplexity does not decrease on the
                        validation set and (ii) epoch has gone past start_decay_at""")
    parser.add_argument('-start_decay_at', type=int, default=5,
                        help="""Start decaying every epoch after and including this epoch""")

    # GPU
    parser.add_argument(
        '-gpus', default=[0], nargs='+', type=int, help="Use CUDA on the listed devices.")
    parser.add_argument('-log_interval', type=int, default=50,
                        help="Print stats at this interval.")
    parser.add_argument('-seed', type=int, default=3435,  # default=-1
                        help="""Random seed used for the experiments reproducibility.""")
    # Critic
    parser.add_argument("-start_reinforce", type=int, default=None,
                        help="""Epoch to start reinforcement training. Use -1 to start immediately.""")
    parser.add_argument("-critic_pretrain_epochs", type=int, default=0,
                        help="Number of epochs to pretrain critic (actor fixed).")
    parser.add_argument("-reinforce_lr", type=float, default=1e-4,
                        help="""Learning rate for reinforcement training.""")

    # Generation
    parser.add_argument("-max_predict_length", required=True,
                        type=int, default=20, help="Maximum length of predictions.")
    parser.add_argument("-predict_mask", type=int, default=0,
                        help="Set to 1 for avoiding repeatitive words and UNK in eval.")

    # Evaluation
    parser.add_argument("-eval", action="store_true",
                        help="Evaluate model only")
    parser.add_argument("-sent_reward", default="unif",
                        choices=["unif", "ocor", "deepcs", "bleu", "unif_mix", "ocor_mix", "deepcs_mix", "code_bert"], help="Sentence reward.")
    parser.add_argument("-eval_codenn", action="store_true",
                        help="Set to True to evaluate on codenn DEV/EVAL. Used for evaluation only.")
    parser.add_argument("-eval_codenn_all", action="store_true",
                        help="Set to True to evaluate on codenn test set. Used for evaluation only.")
    parser.add_argument("-collect_anno", action="store_true",
                        help="Set to True to collect generated annotations.")

    opt = parser.parse_args()
    opt.iteration = 0
    return opt


def load_data(opt):
    dicts = {}
    with open(os.path.join(os.path.dirname(__file__), "../../data-python4csn/descri.json")) as f:
        labelToIdx = json.loads(f.read())
        dicts["src"] = lib.Dict(labelToIdx)
    with open(os.path.join(os.path.dirname(__file__), "../../data-python4csn/descri.json")) as f:
        labelToIdx = json.loads(f.read())
        dicts["tgt"] = lib.Dict(labelToIdx)
    dicts["qt"] = dicts["tgt"]
    dataset_train = np.load(os.path.join(os.path.dirname(
        __file__), "../../data-python4csn/train.npy"), allow_pickle=True).item()
    supervised_data = lib.Dataset(
        dataset_train, "sl_train", opt.batch_size, opt.cuda, eval=False)
    rl_data = lib.Dataset(
        dataset_train, "rl_train", opt.batch_size, opt.cuda, eval=False)

    dataset_valid = np.load(os.path.join(os.path.dirname(
        __file__), "../../data-python4csn/valid.npy"), allow_pickle=True).item()
    valid_data = lib.Dataset(
        dataset_valid, "val", 50, opt.cuda, eval=True)  # opt.batch_size
    test_data = lib.Dataset(dataset_valid, "test", 50, opt.cuda, eval=True)

    print(supervised_data[0][0])
    DEV = None
    EVAL = None

    print(" * vocabulary size. source = %d; target = %d" %
          (dicts["src"].size(), dicts["tgt"].size()))
    # print(" * number of XENT training sentences. %d" %
    #       len(dataset_train["token_array"]))
    # print(" * number of PG training sentences. %d" %
    #       len(dataset_train["token_array"]))
    # print(" * number of val sentences. %d" % len(dataset_valid["token_array"]))
    # print(" * number of test sentences. %d" %
    #       len(dataset_valid["token_array"]))
    print(" * maximum batch size. %d" % opt.batch_size)

    return dicts, supervised_data, rl_data, valid_data, test_data, DEV, EVAL


def get_aligned_embedding(emb_old, dict):
    """
    Get an aligned embedding. Missing values will be randomly initialized.
    :param emb_old: a matrix of shape [vocab_size, vec_dim].
    :param dict: a Dict type of dictionary.
    :return:
    """
    w2v = emb_old.wv
    print("The pretrained emb matrix contains %d words, while the given dict contains %d words..." % (
        len(w2v.vocab), dict.size()))

    emb = []
    for idx, word in dict.idxToLabel.items():
        if word in w2v:
            emb.append(w2v[word])
        else:
            emb.append(np.random.uniform(-opt.param_init,
                                         opt.param_init, opt.word_vec_size))

    emb = torch.Tensor(emb)
    if opt.cuda:
        emb = torch.nn.DataParallel(emb)
        emb = emb.cuda()

    return emb


def init(model, dicts):
    for p in model.parameters():
        p.data.uniform_(-opt.param_init, opt.param_init)


def create_optim(model):
    optim = lib.Optim(
        model.parameters(), opt.optim, opt.lr, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay, start_decay_at=opt.start_decay_at
    )
    return optim


def get_model_class():
    if opt.model_name == 'Transformer':
        from .lib.model.transformer.Models import TransformerWapper
        return TransformerWapper
    return lib.Seq2SeqModel

def create_model(model_class, dicts, gen_out_size):
    encoder = lib.Encoder(opt, dicts["src"])
    decoder = lib.TreeDecoder(opt, dicts["tgt"])
    # Use memory efficient generator when output size is large and
    # max_generator_batches is smaller than batch_size.
    if opt.max_generator_batches < opt.batch_size and gen_out_size > 1:
        generator = lib.MemEfficientGenerator(
            nn.Linear(opt.rnn_size, gen_out_size), opt)
    else:
        generator = lib.BaseGenerator(
            nn.Linear(opt.rnn_size, gen_out_size), opt)
    model = model_class(encoder, decoder, generator, opt)
    init(model, dicts)
    optim = create_optim(model)

    return model, optim


def create_critic(checkpoint, dicts, opt):
    if opt.load_from is not None and "critic" in checkpoint:
        critic = checkpoint["critic"]
        critic_optim = checkpoint["critic_optim"]
    else:
        critic, critic_optim = create_model(lib.Seq2SeqModel, dicts, 1)
    if opt.cuda:
        critic.cuda(opt.gpus[0])
    return critic, critic_optim


def main():
    print("Start...")
    global opt
    opt = get_opt()

    # Set seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    opt.cuda = torch.cuda.is_available() and len(opt.gpus)

    if opt.save_dir and not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with -gpus 1")

    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    dicts, supervised_data, rl_data, valid_data, test_data, DEV, EVAL = load_data(
        opt)
    print("Building model...")

    use_critic = opt.start_reinforce is not None
    print("use_critic: ", use_critic)
    print("has_baseline: ", opt.has_baseline)
    if not opt.has_baseline:
        assert opt.critic_pretrain_epochs == 0

    if opt.load_from is None:
        model, optim = create_model(
            get_model_class(), dicts, dicts["tgt"].size())
        checkpoint = None

    else:
        print("Loading from checkpoint at %s" % opt.load_from)
        # , map_location=lambda storage, loc: storage)
        checkpoint = torch.load(opt.load_from)
        model = checkpoint["model"]
        # config testing
        for attribute in ["predict_mask", "max_predict_length"]:
            if hasattr(model, 'opt'):
                model.opt.__dict__[attribute] = opt.__dict__[attribute]
        optim = checkpoint["optim"]
        optim.start_decay_at = opt.start_decay_at
        if optim.start_decay_at > opt.end_epoch:
            print("No decay!")
        opt.start_epoch = checkpoint["epoch"] + 1

    print("model: ", model)
    print("optim: ", optim)

    # GPU.
    if opt.cuda:
        model.cuda(opt.gpus[0])

    # Start reinforce training immediately.
    print("opt.start_reinforce: ", opt.start_reinforce)

    # Check if end_epoch is large enough.
    if use_critic:
        assert opt.start_epoch + opt.critic_pretrain_epochs - 1 <= \
            opt.end_epoch, "Please increase -end_epoch to perform pretraining!"

    nParams = sum([p.nelement() for p in model.parameters()])
    print("* number of parameters: %d" % nParams)

    # Metrics.
    print("sent_reward: %s" % opt.sent_reward)
    metrics = {}
    metrics["xent_loss"] = lib.Loss.weighted_xent_loss
    metrics["critic_loss"] = lib.Loss.weighted_mse
    if opt.sent_reward == "bleu":
        metrics["sent_reward"] = {"train": lib.Reward.wrapped_sentence_bleu,
                                  "eval": lib.Reward.wrapped_sentence_bleu}
    elif opt.sent_reward == "unif":
        metrics["sent_reward"] = {"train": lib.RetReward.retrieval_unif_mrr_train,
                                  "eval": lib.RetReward.retrieval_unif_mrr_eval}
    elif opt.sent_reward == "ocor":
        metrics["sent_reward"] = {"train": lib.RetReward.retrieval_ocor_mrr_train,
                                  "eval": lib.RetReward.retrieval_ocor_mrr_eval}
    elif opt.sent_reward == "deepcs":
        metrics["sent_reward"] = {"train": lib.RetReward.retrieval_deepcs_mrr_train,
                                  "eval": lib.RetReward.retrieval_deepcs_mrr_eval}
    elif opt.sent_reward == "unif_mix":
        metrics["sent_reward"] = {"train": lib.RetReward.unif_mix_train,
                                  "eval": lib.RetReward.unif_mix_eval}
    elif opt.sent_reward == "ocor_mix":
        metrics["sent_reward"] = {"train": lib.RetReward.ocor_mix_train,
                                  "eval": lib.RetReward.ocor_mix_eval}
    elif opt.sent_reward == "deepcs_mix":
        metrics["sent_reward"] = {"train": lib.RetReward.deepcs_mix_train,
                                  "eval": lib.RetReward.deepcs_mix_eval}
    elif opt.sent_reward == "code_bert":
        metrics["sent_reward"] = {"train": lib.RetReward.retrieval_code_bert_mrr_train,
                                  "eval": lib.RetReward.retrieval_code_bert_mrr_eval}
    else:
        raise Exception(NotImplemented)

    print("opt.eval: ", opt.eval)
    print("opt.eval_codenn: ", opt.eval_codenn)
    print("opt.eval_codenn_all: ", opt.eval_codenn_all)
    print("opt.collect_anno: ", opt.collect_anno)

    sys.stdout.flush()

    # Evaluate model
    if opt.eval:
        evaluator = lib.Evaluator(model, metrics, dicts, opt)
        pred_file = opt.load_from.replace(".pt", ".valid.pred")
        if opt.eval_codenn:
            pred_file = pred_file.replace("valid", "DEV")
            valid_data = DEV
        elif opt.eval_codenn_all:
            pred_file = pred_file.replace("valid", "DEV_all")
            print("* Please input valid data = DEV_all")
        print("valid_data.src: ", len(valid_data.src))
        if opt.predict_mask:
            pred_file += ".masked"
        pred_file += ".metric%s" % opt.sent_reward
        evaluator.eval(valid_data, pred_file)

    else:
        print("supervised_data.src: ", len(supervised_data.src))
        print("supervised_data.tgt: ", len(supervised_data.tgt))
        xent_trainer = lib.Trainer(
            model, supervised_data, valid_data, metrics, dicts, optim, opt, DEV=DEV)

        if use_critic:
            start_time = time.time()
            # Supervised training.
            print("supervised training..")
            print("start_epoch: ", opt.start_epoch)

            xent_trainer.train(
                opt.start_epoch, opt.start_reinforce - 1, start_time)

            if opt.sent_reward == "bleu":
                _valid_data = DEV
            else:
                _valid_data = valid_data

            if opt.has_baseline:
                # Create critic here to not affect random seed.
                critic, critic_optim = create_critic(checkpoint, dicts, opt)
                print("Building critic...")
                print("Critic: ", critic)
                print("Critic optim: ", critic_optim)

                # Pretrain critic.
                print("pretrain critic...")
                if opt.critic_pretrain_epochs > 0 and opt.start_epoch <= opt.start_reinforce + opt.critic_pretrain_epochs - 1:
                    reinforce_trainer = lib.ReinforceTrainer(
                        model, critic, supervised_data, _valid_data, metrics, dicts, optim, critic_optim, opt)
                    reinforce_trainer.train(
                        max(opt.start_epoch, opt.start_reinforce), opt.start_reinforce + opt.critic_pretrain_epochs - 1, True, start_time)
            else:
                print("NOTE: do not have a baseline model")
                critic, critic_optim = None, None

            # Reinforce training.
            print("reinforce training...")
            reinforce_trainer = lib.ReinforceTrainer(
                model, critic, rl_data, _valid_data, metrics, dicts, optim, critic_optim, opt)
            reinforce_trainer.train(
                opt.start_reinforce + opt.critic_pretrain_epochs, opt.end_epoch, False, start_time)

        else:  # Supervised training only. Set opt.start_reinforce to None
            xent_trainer.train(opt.start_epoch, opt.end_epoch)


if __name__ == '__main__':
    main()
