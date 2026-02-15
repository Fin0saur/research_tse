import wesep.models.tse_bsrnn_spk as bsrnn_spk
import wesep.models.tse_bsrnn_spatial as bsrnn_spatial
import wesep.models.tse_nbc2_spatial as nbc2_spatial

def get_model(model_name: str):
    if model_name.startswith("TSE_BSRNN_SPK"):
        return getattr(bsrnn_spk, model_name)
    elif model_name.startswith("TSE_BSRNN_SPATIAL"):
        return getattr(bsrnn_spatial,model_name)
    elif model_name.startswith("TSE_NBC2_SPATIAL"):
        return getattr(nbc2_spatial,model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
