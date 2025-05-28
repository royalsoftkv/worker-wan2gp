import os

from pathlib import Path

def download_models(transformer_filename):
    def computeList(filename):
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]

    def process_files_def(repoId, sourceFolderList, fileList):
        targetRoot = "ckpts/"
        for sourceFolder, files in zip(sourceFolderList,fileList ):
            if len(files)==0:
                if not Path(targetRoot + sourceFolder).exists():
                    print("Downloading " + sourceFolder)
                    snapshot_download(repo_id=repoId,  allow_patterns=sourceFolder +"/*", local_dir= targetRoot)
            else:
                for onefile in files:
                    if len(sourceFolder) > 0:
                        if not os.path.isfile(targetRoot + sourceFolder + "/" + onefile ):
                            print("Downloading " + sourceFolder + "/" + onefile)
                            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot, subfolder=sourceFolder)
                    else:
                        if not os.path.isfile(targetRoot + onefile ):
                            print("Downloading " + onefile)
                            hf_hub_download(repo_id=repoId,  filename=onefile, local_dir = targetRoot)

    from huggingface_hub import hf_hub_download, snapshot_download

    shared_def = {
        "repoId" : "DeepBeepMeep/Wan2.1",
        "sourceFolderList" : [ "pose", "depth", "mask", "wav2vec", ""  ],
        "fileList" : [ [],[], ["sam_vit_h_4b8939_fp16.safetensors"], ["config.json", "feature_extractor_config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
                [ "flownet.pkl"  ] ]
    }
    process_files_def(**shared_def)

    text_encoder_filename = 'ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors'
    model_def = {
        "repoId": "DeepBeepMeep/LTX_Video",
        "sourceFolderList": ["T5_xxl_1.1", ""],
        "fileList": [
            ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"] + computeList(
                text_encoder_filename), ["ltxv_0.9.7_VAE.safetensors", "ltxv_0.9.7_spatial_upscaler.safetensors",
                                         "ltxv_scheduler.json"] + computeList(transformer_filename)]
    }

    process_files_def(**model_def)

model_filelist = ['ckpts/ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors', 'ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors']
for filename in model_filelist:
    print("Downloading", filename)
    download_models(filename)