import wesep.models.tse_bsrnn_spk as bsrnn_spk
import wesep.models.tse_bsrnn_spatial as bsrnn_spatial
import wesep.models.tse_nbc2_spatial as nbc2_spatial
import wesep.models.tse_new as tse_new
import wesep.models.tse_bsrnn_spk_spatial as bsrnn_spk_spatial
import wesep.models.tse_nbc2_spk_spatial as nbc2_spk_spatial

def get_model(model_name: str):
    if model_name.startswith("TSE_BSRNN_SPK_SPATIAL"):
        return getattr(bsrnn_spk_spatial,model_name)
    elif model_name.startswith("TSE_NBC2_SPK_SPATIAL"):
        return getattr(nbc2_spk_spatial,model_name)
    elif model_name.startswith("TSE_BSRNN_SPK"):
        return getattr(bsrnn_spk, model_name)
    elif model_name.startswith("TSE_NEW"):
        return getattr(tse_new,model_name)
    elif model_name.startswith("TSE_BSRNN_SPATIAL"):
        return getattr(bsrnn_spatial,model_name)
    elif model_name.startswith("TSE_NBC2_SPATIAL"):
        return getattr(nbc2_spatial,model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
