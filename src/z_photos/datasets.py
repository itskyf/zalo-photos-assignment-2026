import unicodedata
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

import fiftyone as fo


class CanonicalLabel(StrEnum):
    BABY_PLAYING = auto()
    LUNAR_NEW_YEAR = auto()
    OTHER = auto()
    NATURE = auto()
    TREKKING = auto()
    GATHERING = auto()


class BronzeField(StrEnum):
    RAW_LABEL = auto()
    GROUND_TRUTH = auto()
    METADATA = auto()
    IS_EXACT_DUP = auto()
    IS_NEAR_DUP = auto()
    IS_LEAKY = auto()
    RADIO_UNIQUENESS = auto()


class SilverField(StrEnum):
    SIGLIP2_EMB = auto()
    ZS_SCORE = auto()
    LR_OOF_SCORE = auto()
    CLEAN_SCORE = auto()
    SAMPLE_WEIGHT_PRELIM = auto()


class GoldField(StrEnum):
    KEEP_FOR_TRAIN = auto()
    SAMPLE_WEIGHT = auto()
    PRED_HEAD = auto()
    PRED_FINAL = auto()


@dataclass(frozen=True)
class LabelMetadata:
    canonical_label: CanonicalLabel
    en_name: str
    vi_name: str
    raw_label: str


LABEL_REGISTRY: tuple[LabelMetadata, ...] = (
    LabelMetadata(
        canonical_label=CanonicalLabel.BABY_PLAYING,
        en_name="baby playing",
        vi_name="em bé chơi",
        raw_label="em_bé_chơi_verified",
    ),
    LabelMetadata(
        canonical_label=CanonicalLabel.LUNAR_NEW_YEAR,
        en_name="lunar new year",
        vi_name="ngày tết",
        raw_label="ngày_tết_verified",
    ),
    LabelMetadata(
        canonical_label=CanonicalLabel.OTHER,
        en_name="other",
        vi_name="khác",
        raw_label="other",
    ),
    LabelMetadata(
        canonical_label=CanonicalLabel.NATURE,
        en_name="nature",
        vi_name="thiên nhiên",
        raw_label="thiennhien",
    ),
    LabelMetadata(
        canonical_label=CanonicalLabel.TREKKING,
        en_name="trekking",
        vi_name="trekking",
        raw_label="trekking_verified",
    ),
    LabelMetadata(
        canonical_label=CanonicalLabel.GATHERING,
        en_name="gathering",
        vi_name="tụ họp",
        raw_label="tụ_họp_verified",
    ),
)

RAW_TO_CANONICAL = {
    metadata.raw_label: metadata.canonical_label.value for metadata in LABEL_REGISTRY
}
# Path in macOS may use NFD
RAW_TO_CANONICAL.update(
    {unicodedata.normalize("NFD", k): v for k, v in RAW_TO_CANONICAL.items()}
)


def build_bronze_dataset(name: str, bronze_dir: Path):
    if name in fo.list_datasets():
        return fo.load_dataset(name)

    dataset = fo.Dataset(name=name, persistent=True)
    dataset.default_classes = list(RAW_TO_CANONICAL.values())
    dataset.add_sample_field(
        BronzeField.GROUND_TRUTH,
        fo.EmbeddedDocumentField,
        embedded_doc_type=fo.Classification,
    )

    for split in ["train", "test"]:
        dataset.add_dir(
            dataset_dir=bronze_dir / f"take_home_test/data_{split}",
            dataset_type=fo.types.ImageClassificationDirectoryTree,
            label_field=BronzeField.RAW_LABEL,
            tags=[split],
        )

    raw_labels = dataset.values(f"{BronzeField.RAW_LABEL}.label")

    dataset.delete_sample_field(BronzeField.RAW_LABEL)
    dataset.add_sample_field(BronzeField.RAW_LABEL, fo.StringField)
    dataset.set_values(BronzeField.RAW_LABEL, raw_labels)

    dataset.set_values(
        f"{BronzeField.GROUND_TRUTH}.label",
        [RAW_TO_CANONICAL[label] for label in raw_labels],
    )
    dataset.compute_metadata()
    return dataset
