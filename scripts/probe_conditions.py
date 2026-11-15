"""Check, on readings from dev runs and controls, whether the program decides each rule needs its fix.

It mirrors `TelecomProgram._rule_step` for a reading that decides the rule. A toggle (the policy names the state its fix
leaves) needs its fix unless the reading already shows that state. A fix that is safe to repeat is applied when either the
condition holds or the reading is not yet in the fixed state. An escalation rule applies when its condition holds.
Run against a live Qwen3-8B server: `python -I scripts/probe_conditions.py`.
"""

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


def by(section: str, n: int = 0):
    return [r for r in rules if r.section == section][n]


def needs_fix(rule, reading: str) -> str:
    condition = run(P.DOES_CONDITION_HOLD, Context(), rule=rule.condition, rule_text=rule.policy, reading=reading).answer
    if rule.escalate or rule.fix not in compiled.device_functions:
        return condition
    in_place = run(
        P.ALREADY_AS_WANTED, Context(), rule=f"{rule.fix}: {rule.fixed_state or rule.condition}", rule_text=rule.policy, reading=reading
    ).answer
    if rule.fixed_state:
        return condition if in_place == "cannot tell" else ("no" if in_place == "yes" else "yes")
    return "yes" if condition == "yes" or in_place == "no" else condition


CASES = [
    ("SIM Card Issues", 0, "check_sim_status: The SIM card is active.", "no"),
    ("SIM Card Issues", 1, "check_sim_status: The SIM card is active.", "no"),
    ("SIM Card Issues", 1, "check_sim_status: Your SIM card is locked with a PIN code.", "yes"),
    ("Airplane Mode", 0, 'check_network_status: Airplane Mode is on, and my phone says "no service."', "yes"),
    ("Airplane Mode", 0, "check_network_status: Airplane Mode: OFF, Mobile Data Enabled: Yes", "no"),
    ("Data Saver Mode", 0, "check_data_restriction_status: Data Saver mode is off", "no"),
    ("Data Saver Mode", 0, "check_data_restriction_status: Oh, it says Data Saver mode is ON", "yes"),
    ("VPN Connection Issues", 0, "check_vpn_status: VPN is ON and connected, performance level Poor", "yes"),
    ("VPN Connection Issues", 0, "check_vpn_status: The VPN is off", "no"),
    ("Mobile Data Disabled", 0, "check_network_status: it says mobile data is not enabled", "yes"),
    ("Mobile Data Disabled", 0, 'check_network_status: It says "Mobile Data Enabled: No."', "yes"),
    ("Mobile Data Disabled", 0, "check_network_status: Mobile Data Enabled: Yes", "no"),
    ("Optimizing Network Mode Preferences", 0, "check_network_mode_preference: 2g_only", "yes"),
    ("Optimizing Network Mode Preferences", 0, "check_network_mode_preference: 4g_5g_preferred", "no"),
    ("Messaging App Lacks Necessary Permissions", 0, "check_app_permissions: The messaging app has permission for storage and phone.", "yes"),
    ("Messaging App Lacks Necessary Permissions", 0, "check_app_permissions: messaging has storage and sms permissions granted", "no"),
    ("Messaging App Lacks Necessary Permissions", 0, "check_app_permissions: The messaging app only has the sms permission.", "yes"),
    ("Investigating Wi-Fi Calling Interference with MMS", 0, "check_wifi_calling_status: Wi-Fi Calling is on", "yes"),
    ("Investigating Wi-Fi Calling Interference with MMS", 0, "check_wifi_calling_status: Wi-Fi Calling is OFF", "no"),
]

correct = 0
for section, n, reading, want in CASES:
    got = needs_fix(by(section, n), reading)
    correct += got == want
    print(f"{'OK ' if got == want else 'BAD'} want={want:3} got={got:11} {section[:30]:30} | {reading[:70]}")
print(f"{correct}/{len(CASES)} correct")
