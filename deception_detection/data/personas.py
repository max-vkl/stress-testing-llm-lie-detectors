"""Names for the immutable supplied-answer rollouts bundled with this repository."""
from deception_detection.data.base import DialogueDataset

class PersonasDataset(DialogueDataset):
    base_name = 'personas'
    all_variants = ['paper_ready_ancient_astronaut', 'paper_ready_ancient_astronaut_deceptOnlySysPrompt', 'paper_ready_astrology', 'paper_ready_astrology_deceptOnlySysPrompt', 'paper_ready_conspiracy', 'paper_ready_conspiracy_deceptOnlySysPrompt', 'paper_ready_flat_earth', 'paper_ready_flat_earth_deceptOnlySysPrompt', 'paper_ready_harrypotter', 'paper_ready_harrypotter_deceptOnlySysPrompt', 'paper_ready_man_in_high_castle', 'paper_ready_man_in_high_castle_deceptOnlySysPrompt', 'paper_ready_medievalphy', 'paper_ready_medievalphy_deceptOnlySysPrompt', 'paper_ready_miasma', 'paper_ready_miasma_deceptOnlySysPrompt', 'paper_ready_orwell', 'paper_ready_orwell_deceptOnlySysPrompt', 'paper_ready_phlogiston', 'paper_ready_phlogiston_deceptOnlySysPrompt', 'paper_ready_phrenologist', 'paper_ready_phrenologist_deceptOnlySysPrompt', 'paper_ready_pteolomeic_astronomer', 'paper_ready_pteolomeic_astronomer_deceptOnlySysPrompt', 'paper_ready_quantumcoach', 'paper_ready_quantumcoach_deceptOnlySysPrompt', 'paper_ready_startrek', 'paper_ready_startrek_deceptOnlySysPrompt', 'paper_ready_wakanda', 'paper_ready_wakanda_deceptOnlySysPrompt']
    default_variant = 'paper_ready_ancient_astronaut'

    def _get_dialogues(self):
        raise RuntimeError("Load this dataset using its registered model-specific rollouts.")
