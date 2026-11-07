"""Names shared by the paper-ready importer and dataset registration."""

PAPER_READY_PERSONAS = (
    "ancient_astronaut",
    "astrology",
    "conspiracy",
    "flat_earth",
    "harrypotter",
    "man_in_high_castle",
    "medievalphy",
    "miasma",
    "orwell",
    "phlogiston",
    "quantumcoach",
    "startrek",
    "wakanda",
    "pteolomeic_astronomer",
    "phrenologist",
)
PAPER_READY_CONDITIONS = ("", "_deceptOnlySysPrompt")
PAPER_READY_VARIANTS = [
    f"paper_ready_{persona}{condition}"
    for persona in PAPER_READY_PERSONAS
    for condition in PAPER_READY_CONDITIONS
]
