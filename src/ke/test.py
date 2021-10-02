import os
import torch
from ..UNIF.validate import validate, model, testset, config
from ..UNIF.mydata import mydataset_test, mydataset_wordnet
from ..UNIF.data_loader import CodeSearchDataset

ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_xentke_java_mrr_code_bert_transformer_v2/model_xentke_java_mrr_code_bert_transformer_v2_19.pt')

if __name__ == "__main__":
    load = torch.load(ke_path)
    ke = load['model']
    # this is the test function, you can modify pool size and VALID or TEST set
    # testresult = validate(mydataset_test(6, 50, 30, 30), model,
    #                       1000, 1, 'cos_integrate', ke, 0.6)
    testresult = validate(mydataset_test(6, 50, 30, 30), model,
                          1000, 1, 'cos', ke)
    print(testresult)
    print("done")
