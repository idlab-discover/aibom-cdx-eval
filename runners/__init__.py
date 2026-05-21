"""Runner package — exports the tool_id → Runner class mapping."""
from runners.run_aibomgen_cli import AibomgenCliRunner
from runners.run_aibom_generator import AibomGeneratorRunner
from runners.run_aloha import ALOHARunner
from runners.run_akumar_aibom import AkumarAibomRunner
from runners.run_jorisbot_aibom import JorisbotAibomRunner
from runners.run_kriskimmerle_aibom import KriskimmerleAibomRunner
from runners.run_lab700xorg_aisbom import Lab700xorgAisbomRunner
from runners.run_trusera_aibom import TruseraAibomRunner

REGISTRY: dict[str, type] = {
    "aibomgen-cli":       AibomgenCliRunner,
    "aibom-generator":    AibomGeneratorRunner,
    "ALOHA":              ALOHARunner,
    "akumar-aibom":       AkumarAibomRunner,
    "jorisbot-aibom":     JorisbotAibomRunner,
    "kriskimmerle-aibom": KriskimmerleAibomRunner,
    "lab700xorg-aisbom":  Lab700xorgAisbomRunner,
    "trusera-aibom":      TruseraAibomRunner,
}
