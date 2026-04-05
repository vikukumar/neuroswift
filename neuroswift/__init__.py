from ._version import __version__
from .artifacts_io import ArtifactSaver, OmniArtifact, save_audio, save_image, save_video_frames
from .auto_trainer import AutoTrainer
from .data_pipeline import (
    PipelineStats,
    TrainPair,
    augment_pairs,
    deduplicate,
    filter_pair,
    ingest_directory,
    ingest_file,
    ingest_path,
    normalize_pair,
    normalize_text,
    run_pipeline,
    split_train_val,
)
from .ingest import DatasetFolderReader, MultimodalSample, pad_tensor_list
from .layers import CrossModalAttention, LinearSSM, RMSNorm, SparseMoE, auto_device
from .model import NeuroSwiftBlock, NeuroSwiftConfig, NeuroSwiftLM
from .omni import (
    AudioChunkEncoder,
    ImageCNNDecoder,
    ImagePatchEncoder,
    NeuroSwiftAssistant,
    NeuroSwiftOmni,
    NeuroSwiftOmniConfig,
    VideoCNNDecoder,
    VideoFrameEncoder,
)
from .plasticity import HebbianUpdater
from .prompting import PromptEngineer, PromptIntent, infer_prompt_intent
from .rag import NeuroSwiftRAG, RAGDocument, RetrievalHit
from .tokenizer import CharTokenizer, WordTokenizer, load_tokenizer

__all__ = [
    # Artifacts
    "ArtifactSaver",
    "OmniArtifact",
    "save_audio",
    "save_image",
    "save_video_frames",
    # Auto-training
    "AutoTrainer",
    # Data pipeline
    "PipelineStats",
    "TrainPair",
    "augment_pairs",
    "deduplicate",
    "filter_pair",
    "ingest_directory",
    "ingest_file",
    "ingest_path",
    "normalize_pair",
    "normalize_text",
    "run_pipeline",
    "split_train_val",
    # Layers
    "CrossModalAttention",
    "LinearSSM",
    "RMSNorm",
    "SparseMoE",
    "auto_device",
    # Tokenizers
    "CharTokenizer",
    "WordTokenizer",
    "load_tokenizer",
    # Data ingestion
    "DatasetFolderReader",
    "MultimodalSample",
    "pad_tensor_list",
    # Hebbian plasticity
    "HebbianUpdater",
    # Core LM
    "NeuroSwiftBlock",
    "NeuroSwiftConfig",
    "NeuroSwiftLM",
    # Omni multimodal
    "AudioChunkEncoder",
    "ImageCNNDecoder",
    "ImagePatchEncoder",
    "NeuroSwiftAssistant",
    "NeuroSwiftOmni",
    "NeuroSwiftOmniConfig",
    "VideoCNNDecoder",
    "VideoFrameEncoder",
    # Prompting
    "PromptEngineer",
    "PromptIntent",
    "infer_prompt_intent",
    # RAG
    "NeuroSwiftRAG",
    "RAGDocument",
    "RetrievalHit",
    # Version
    "__version__",
]
