import os
import torch
from ..deepcs.validate import validate, model, testset, config
from ..deepcs.mydata import mydataset_test, mydataset_wordnet
from ..deepcs.data_loader import CodeSearchDataset

ke_path = os.path.join(os.path.dirname(
    __file__), '../../save/model_rf_hasBaselineke_java_deepcs_mrr_reinforce/model_rf_hasBaselineke_java_deepcs_mrr_reinforce_50.pt')

if __name__ == "__main__":
    load = torch.load(ke_path)
    ke = load['model']
    # this is the test function, you can modify pool size and VALID or TEST set
    testresult = validate(mydataset_test(
        6, 50, 30, 30), model, 1000, 1, 'cos_integrate', ke, 0.6)
    print(testresult)
    print("done")
