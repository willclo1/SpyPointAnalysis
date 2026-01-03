
from pathlib import Path
from pathlib import Path
from animal_filter import decide_keep

IMAGES = Path("images")

for p in sorted(IMAGES.glob("*.jpg")):
    dec = decide_keep(str(p))
    print("-" * 60)
    print(p.name)
    print("  KEEP:", dec.keep)
    print("  REASON:", dec.reason)
    print("  ALL OBJECTS:", [(d.name, round(d.score, 2)) for d in dec.all_objects])
    print("  ANIMALS:", [(d.name, round(d.score, 2)) for d in dec.animals])
    print("  VEHICLES_AT_GATE:", [(d.name, round(d.score, 2)) for d in dec.vehicles_at_gate])
    print("  PEOPLE_AT_GATE:", [(d.name, round(d.score, 2)) for d in dec.people_at_gate])
