"""Check the copy-copy-compare condition primitives on cases that failed in dev runs, plus controls."""
from tau2.runner import build_environment

from plvr.llm import Server
from plvr.models import MODELS
from plvr.primitive import Context, Runner
from plvr.telecom import primitives as P
from plvr.telecom.policy import compile_policy

env = build_environment("telecom")
run = Runner(Server(MODELS["qwen3-8b"]))
compiled = compile_policy(env.get_policy(), env.get_tools(), run)
rules = [r for a in compiled.areas.values() for r in a.rules]
by = lambda section, n=0: [r for r in rules if r.section == section][n]
print("sections:", sorted({r.section for r in rules}))

cases = [
    (P.DOES_CONDITION_HOLD, by("SIM Card Issues", 0), "check_sim_status: The SIM card is active.", "no"),
    (P.DOES_CONDITION_HOLD, by("SIM Card Issues", 1), "check_sim_status: The SIM card is active.", "no"),
    (P.DOES_CONDITION_HOLD, by("SIM Card Issues", 1), "check_sim_status: Your SIM card is locked with a PIN code.", "yes"),
    (P.DOES_CONDITION_HOLD, by("Airplane Mode"), 'check_network_status: Airplane Mode is on, and my phone says "no service."', "yes"),
    (P.DOES_CONDITION_HOLD, by("Airplane Mode"), "check_network_status: Airplane Mode: OFF, Mobile Data Enabled: Yes", "no"),
    (P.DOES_CONDITION_HOLD, by("Data Saver Mode"), "check_data_restriction_status: Data Saver mode is off", "no"),
    (P.DOES_CONDITION_HOLD, by("Data Saver Mode"), "check_data_restriction_status: Oh, it says Data Saver mode is ON", "yes"),
    (P.DOES_CONDITION_HOLD, by("VPN Connection Issues"), "check_vpn_status: VPN is ON and connected, performance level Poor", "yes"),
    (P.DOES_CONDITION_HOLD, by("Mobile Data Disabled"), "check_network_status: it says mobile data is not enabled", "yes"),
    (P.DOES_CONDITION_HOLD, by("Mobile Data Disabled"), "check_network_status: Mobile Data Enabled: Yes", "no"),
    (P.ALREADY_AS_WANTED, by("Addressing Data Roaming Problems"), "check_network_status: Data Roaming Enabled: No", "no"),
    (P.ALREADY_AS_WANTED, by("Addressing Data Roaming Problems"), "check_network_status: Data Roaming Enabled: Yes", "yes"),
]
ok = 0
for primitive, rule, reading, want in cases:
    text = f"[{rule.section}] {rule.condition}" if primitive is P.DOES_CONDITION_HOLD else f"{rule.fix}: [{rule.section}] {rule.condition}"
    out = run(primitive, Context(), rule=text, rule_text=rule.policy, reading=reading)
    ok += out.answer == want
    print(f"{'OK ' if out.answer == want else 'BAD'} want={want:3} got={out.answer:11} {rule.section[:24]:24} | {reading[16:70]:54} | {out.model_dump()} {run.calls[-1].rejection or ''}")
print(f"{ok}/{len(cases)} correct")
