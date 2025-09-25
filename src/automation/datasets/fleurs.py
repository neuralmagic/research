# coding=utf-8
# Copyright 2022 The Google and HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import OrderedDict
import datasets

logger = datasets.logging.get_logger(__name__)


""" FLEURS Dataset"""

# Mapping from full language names to ISO language codes (e.g., "English" -> "en")
_FLEURS_LANG_TO_ID = OrderedDict([("Afrikaans", "af"), ("Amharic", "am"), ("Arabic", "ar"), ("Armenian", "hy"), ("Assamese", "as"), ("Asturian", "ast"), ("Azerbaijani", "az"), ("Belarusian", "be"), ("Bengali", "bn"), ("Bosnian", "bs"), ("Bulgarian", "bg"), ("Burmese", "my"), ("Catalan", "ca"), ("Cebuano", "ceb"), ("Mandarin Chinese", "cmn_hans"), ("Cantonese Chinese", "yue_hant"), ("Croatian", "hr"), ("Czech", "cs"), ("Danish", "da"), ("Dutch", "nl"), ("English", "en"), ("Estonian", "et"), ("Filipino", "fil"), ("Finnish", "fi"), ("French", "fr"), ("Fula", "ff"), ("Galician", "gl"), ("Ganda", "lg"), ("Georgian", "ka"), ("German", "de"), ("Greek", "el"), ("Gujarati", "gu"), ("Hausa", "ha"), ("Hebrew", "he"), ("Hindi", "hi"), ("Hungarian", "hu"), ("Icelandic", "is"), ("Igbo", "ig"), ("Indonesian", "id"), ("Irish", "ga"), ("Italian", "it"), ("Japanese", "ja"), ("Javanese", "jv"), ("Kabuverdianu", "kea"), ("Kamba", "kam"), ("Kannada", "kn"), ("Kazakh", "kk"), ("Khmer", "km"), ("Korean", "ko"), ("Kyrgyz", "ky"), ("Lao", "lo"), ("Latvian", "lv"), ("Lingala", "ln"), ("Lithuanian", "lt"), ("Luo", "luo"), ("Luxembourgish", "lb"), ("Macedonian", "mk"), ("Malay", "ms"), ("Malayalam", "ml"), ("Maltese", "mt"), ("Maori", "mi"), ("Marathi", "mr"), ("Mongolian", "mn"), ("Nepali", "ne"), ("Northern-Sotho", "nso"), ("Norwegian", "nb"), ("Nyanja", "ny"), ("Occitan", "oc"), ("Oriya", "or"), ("Oromo", "om"), ("Pashto", "ps"), ("Persian", "fa"), ("Polish", "pl"), ("Portuguese", "pt"), ("Punjabi", "pa"), ("Romanian", "ro"), ("Russian", "ru"), ("Serbian", "sr"), ("Shona", "sn"), ("Sindhi", "sd"), ("Slovak", "sk"), ("Slovenian", "sl"), ("Somali", "so"), ("Sorani-Kurdish", "ckb"), ("Spanish", "es"), ("Swahili", "sw"), ("Swedish", "sv"), ("Tajik", "tg"), ("Tamil", "ta"), ("Telugu", "te"), ("Thai", "th"), ("Turkish", "tr"), ("Ukrainian", "uk"), ("Umbundu", "umb"), ("Urdu", "ur"), ("Uzbek", "uz"), ("Vietnamese", "vi"), ("Welsh", "cy"), ("Wolof", "wo"), ("Xhosa", "xh"), ("Yoruba", "yo"), ("Zulu", "zu")])

# Reverse mapping from ISO language codes to full language names (e.g., "en" -> "English")
_FLEURS_LANG_SHORT_TO_LONG = {v: k for k, v in _FLEURS_LANG_TO_ID.items()}

# List of all 102 language codes in format "language_country" (e.g., "en_us", "fr_fr")
# These are the actual dataset identifiers used for data paths and configuration
_FLEURS_LANG = sorted(["af_za", "am_et", "ar_eg", "as_in", "ast_es", "az_az", "be_by", "bn_in", "bs_ba", "ca_es", "ceb_ph", "cmn_hans_cn", "yue_hant_hk", "cs_cz", "cy_gb", "da_dk", "de_de", "el_gr", "en_us", "es_419", "et_ee", "fa_ir", "ff_sn", "fi_fi", "fil_ph", "fr_fr", "ga_ie", "gl_es", "gu_in", "ha_ng", "he_il", "hi_in", "hr_hr", "hu_hu", "hy_am", "id_id", "ig_ng", "is_is", "it_it", "ja_jp", "jv_id", "ka_ge", "kam_ke", "kea_cv", "kk_kz", "km_kh", "kn_in", "ko_kr", "ckb_iq", "ky_kg", "lb_lu", "lg_ug", "ln_cd", "lo_la", "lt_lt", "luo_ke", "lv_lv", "mi_nz", "mk_mk", "ml_in", "mn_mn", "mr_in", "ms_my", "mt_mt", "my_mm", "nb_no", "ne_np", "nl_nl", "nso_za", "ny_mw", "oc_fr", "om_et", "or_in", "pa_in", "pl_pl", "ps_af", "pt_br", "ro_ro", "ru_ru", "bg_bg", "sd_in", "sk_sk", "sl_si", "sn_zw", "so_so", "sr_rs", "sv_se", "sw_ke", "ta_in", "te_in", "tg_tj", "th_th", "tr_tr", "uk_ua", "umb_ao", "ur_pk", "uz_uz", "vi_vn", "wo_sn", "xh_za", "yo_ng", "zu_za"])

# Mapping from full language names to language codes (e.g., "English" -> "en_us")
_FLEURS_LONG_TO_LANG = {_FLEURS_LANG_SHORT_TO_LONG["_".join(k.split("_")[:-1]) or k]: k for k in _FLEURS_LANG}

# Reverse mapping from language codes to full language names (e.g., "en_us" -> "English")
_FLEURS_LANG_TO_LONG = {v: k for k, v in _FLEURS_LONG_TO_LANG.items()}

# Geographic grouping of languages into 7 regions for analysis and classification
# Each group contains languages from a specific geographic area
_FLEURS_GROUP_TO_LONG = OrderedDict({
    "western_european_we": ["Asturian", "Bosnian", "Catalan", "Croatian", "Danish", "Dutch", "English", "Finnish", "French", "Galician", "German", "Greek", "Hungarian", "Icelandic", "Irish", "Italian", "Kabuverdianu", "Luxembourgish", "Maltese", "Norwegian", "Occitan", "Portuguese", "Spanish", "Swedish", "Welsh"],
    "eastern_european_ee": ["Armenian", "Belarusian", "Bulgarian", "Czech", "Estonian", "Georgian", "Latvian", "Lithuanian", "Macedonian", "Polish", "Romanian", "Russian", "Serbian", "Slovak", "Slovenian", "Ukrainian"],
    "central_asia_middle_north_african_cmn": ["Arabic", "Azerbaijani", "Hebrew", "Kazakh", "Kyrgyz", "Mongolian", "Pashto", "Persian", "Sorani-Kurdish", "Tajik", "Turkish", "Uzbek"],
    "sub_saharan_african_ssa": ["Afrikaans", "Amharic", "Fula", "Ganda", "Hausa", "Igbo", "Kamba", "Lingala", "Luo", "Northern-Sotho", "Nyanja", "Oromo", "Shona", "Somali", "Swahili", "Umbundu", "Wolof", "Xhosa", "Yoruba", "Zulu"],
    "south_asian_sa": ["Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Oriya", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"],
    "south_east_asian_sea": ["Burmese", "Cebuano", "Filipino", "Indonesian", "Javanese", "Khmer", "Lao", "Malay", "Maori", "Thai", "Vietnamese"],
    "chinese_japanase_korean_cjk": ["Mandarin Chinese", "Cantonese Chinese", "Japanese", "Korean"],
})

# Mapping from full language names to their geographic group (e.g., "English" -> "western_european_we")
_FLEURS_LONG_TO_GROUP = {a: k for k, v in _FLEURS_GROUP_TO_LONG.items() for a in v}

# Mapping from language codes to their geographic group (e.g., "en_us" -> "western_european_we")
_FLEURS_LANG_TO_GROUP = {_FLEURS_LONG_TO_LANG[k]: v for k, v in _FLEURS_LONG_TO_GROUP.items()}

# Alias for _FLEURS_LANG for clarity
_ALL_LANG = _FLEURS_LANG

# List of all available dataset configurations
# Contains all individual language codes plus an "all" configuration for loading all languages
_ALL_CONFIGS = []

for langs in _FLEURS_LANG:
    _ALL_CONFIGS.append(langs)

_ALL_CONFIGS.append("all")

# TODO(FLEURS)
_DESCRIPTION = "FLEURS is the speech version of the FLORES machine translation benchmark, covering 2000 n-way parallel sentences in n=102 languages."
_CITATION = ""
_HOMEPAGE_URL = ""

# HuggingFace Hub repository URLs
_HF_REPO_ID = "google/fleurs"
_BASE_PATH = "data/{langs}/"
_DATA_URL = f"https://huggingface.co/datasets/{_HF_REPO_ID}/resolve/main/" + _BASE_PATH + "audio/{split}.tar.gz"
_META_URL = f"https://huggingface.co/datasets/{_HF_REPO_ID}/resolve/main/" + _BASE_PATH + "{split}.tsv"


def load_fleurs_dataset(name="en_us", split=None):
    """
    Load FLEURS dataset following the same syntax as datasets.load_dataset.
    
    Args:
        name (str): Language code (e.g., "en_us", "fr_fr") or "all"
        split (str, optional): Dataset split to load ("train", "validation", "test"). 
                              If None, loads all splits.
    
    Returns:
        dict: Dataset with the specified split(s)
    """
    try:
        # Validate language code
        if name not in _ALL_CONFIGS:
            raise ValueError(f"Invalid language code. Must be one of: {_ALL_CONFIGS}")
        
        # Validate split if provided
        if split is not None and split not in ["train", "validation", "test"]:
            raise ValueError(f"Invalid split. Must be one of: ['train', 'validation', 'test']")
        
        # Create dataset instance with the specific config name
        fleurs = Fleurs(config_name=name)
        
        # Download and prepare the dataset first
        fleurs.download_and_prepare()
        
        # Load the dataset (all splits if split=None, otherwise specific split)
        if split is None:
            dataset = fleurs.as_dataset()
            print(f"Successfully loaded FLEURS dataset for: {name}")
            print(f"Available splits: {list(dataset.keys())}")
        else:
            dataset = fleurs.as_dataset(split=split)
            print(f"Successfully loaded FLEURS dataset for: {name}")
            print(f"Loaded split: {split}")
            print(f"Dataset size: {len(dataset)} examples")
               
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
        

class FleursConfig(datasets.BuilderConfig):
    """BuilderConfig for xtreme-s"""

    def __init__(
        self, name, description, citation, homepage
    ):
        super(FleursConfig, self).__init__(
            name=self.name,
            version=datasets.Version("2.0.0", ""),
            description=self.description,
        )
        self.name = name
        self.description = description
        self.citation = citation
        self.homepage = homepage


def _build_config(name):
    return FleursConfig(
        name=name,
        description=_DESCRIPTION,
        citation=_CITATION,
        homepage=_HOMEPAGE_URL,
    )


class Fleurs(datasets.GeneratorBasedBuilder):

    DEFAULT_WRITER_BATCH_SIZE = 1000
    BUILDER_CONFIGS = [_build_config(name) for name in _ALL_CONFIGS]

    def _info(self):
        langs = _ALL_CONFIGS
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "num_samples": datasets.Value("int32"),
                "path": datasets.Value("string"),
                "audio": datasets.Audio(sampling_rate=16_000),
                "transcription": datasets.Value("string"),
                "raw_transcription": datasets.Value("string"),
                "gender": datasets.ClassLabel(names=["male", "female", "other"]),
                "lang_id": datasets.ClassLabel(names=langs),
                "language": datasets.Value("string"),
                "lang_group_id": datasets.ClassLabel(
                    names=list(_FLEURS_GROUP_TO_LONG.keys())
                ),
            }
        )

        return datasets.DatasetInfo(
            description=self.config.description + "\n" + _DESCRIPTION,
            features=features,
            supervised_keys=("audio", "transcription"),
            homepage=self.config.homepage,
            citation=self.config.citation + "\n" + _CITATION,
        )

    # Fleurs
    def _split_generators(self, dl_manager):
        splits = ["train", "dev", "test"]

        # metadata_path = dl_manager.download_and_extract(_METADATA_URL)

        if self.config.name == "all":
            data_urls = {split: [_DATA_URL.format(langs=langs,split=split) for langs in _FLEURS_LANG] for split in splits}
            meta_urls = {split: [_META_URL.format(langs=langs,split=split) for langs in _FLEURS_LANG] for split in splits}
        else:
            data_urls = {split: [_DATA_URL.format(langs=self.config.name, split=split)] for split in splits}
            meta_urls = {split: [_META_URL.format(langs=self.config.name, split=split)] for split in splits}

        archive_paths = dl_manager.download(data_urls)
        local_extracted_archives = dl_manager.extract(archive_paths) if not dl_manager.is_streaming else {}
        archive_iters = {split: [dl_manager.iter_archive(path) for path in paths] for split, paths in archive_paths.items()}

        meta_paths = dl_manager.download(meta_urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archives": local_extracted_archives.get("train", [None] * len(meta_paths.get("train"))),
                    "archive_iters": archive_iters.get("train"),
                    "text_paths": meta_paths.get("train")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "local_extracted_archives": local_extracted_archives.get("dev", [None] * len(meta_paths.get("dev"))),
                    "archive_iters": archive_iters.get("dev"),
                    "text_paths": meta_paths.get("dev")
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archives": local_extracted_archives.get("test", [None] * len(meta_paths.get("test"))),
                    "archive_iters": archive_iters.get("test"),
                    "text_paths": meta_paths.get("test")
                },
            ),
        ]

    def _get_data(self, lines, lang_id):
        data = {}
        gender_to_id = {"MALE": 0, "FEMALE": 1, "OTHER": 2}
        for line in lines:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            (
                _id,
                file_name,
                raw_transcription,
                transcription,
                _,
                num_samples,
                gender,
            ) = line.strip().split("\t")

            lang_group = _FLEURS_LANG_TO_GROUP[lang_id]

            data[file_name] = {
                "id": int(_id),
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "num_samples": int(num_samples),
                "gender": gender_to_id[gender],
                "lang_id": _FLEURS_LANG.index(lang_id),
                "language": _FLEURS_LANG_TO_LONG[lang_id],
                "lang_group_id": list(_FLEURS_GROUP_TO_LONG.keys()).index(
                    lang_group
                ),
            }

        return data

    def _generate_examples(self, local_extracted_archives, archive_iters, text_paths):
        assert len(local_extracted_archives) == len(archive_iters) == len(text_paths)
        key = 0

        if self.config.name == "all":
            langs = _FLEURS_LANG
        else:
            langs = [self.config.name]

        for archive, text_path, local_extracted_path, lang_id in zip(archive_iters, text_paths, local_extracted_archives, langs):
            with open(text_path, encoding="utf-8") as f:
                lines = f.readlines()
                data = self._get_data(lines, lang_id)

            for audio_path, audio_file in archive:
                audio_filename = audio_path.split("/")[-1]
                if audio_filename not in data.keys():
                    continue

                result = data[audio_filename]
                extracted_audio_path = (
                    os.path.join(local_extracted_path, audio_filename)
                    if local_extracted_path is not None
                    else None
                )
                result["path"] = extracted_audio_path
                result["audio"] = {"path": audio_path, "bytes": audio_file.read()}
                yield key, result
                key += 1
