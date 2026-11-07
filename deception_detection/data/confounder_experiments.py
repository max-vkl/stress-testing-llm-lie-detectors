"""Names for the immutable supplied-answer rollouts bundled with this repository."""
from deception_detection.data.base import DialogueDataset

class ConfounderExperimentsDataset(DialogueDataset):
    base_name = 'confounder_experiments'
    all_variants = ['compliance_brackets', 'persona_stance_nora', 'probability_score_learning_16turn']
    default_variant = 'compliance_brackets'

    def _get_dialogues(self):
        raise RuntimeError("Load this dataset using its registered model-specific rollouts.")
