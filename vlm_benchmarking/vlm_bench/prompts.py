BIDIRECTIONAL_RELATIONS = frozenset([
    "is_left_of", "is_right_of",
    "is_in_front_of", "is_behind",
    "is_on_top_of", "is_below_of",
    "is_inside", "contains",
])

INVERSE_MAP = {
    "is_left_of": "is_right_of",   "is_right_of": "is_left_of",
    "is_in_front_of": "is_behind", "is_behind": "is_in_front_of",
    "is_on_top_of": "is_below_of", "is_below_of": "is_on_top_of",
    "is_inside": "contains",       "contains": "is_inside",
}


def _rules_block(objects: list[str], task_hint: str) -> str:
    obj_str = "\n".join(f"- {o}" for o in objects)
    return f"""TASK CONTEXT:
This scene is from a robot task described as: "{task_hint}"
IMPORTANT: Pay close attention to spatial cues in the task description (e.g., "between X and Y", "on the Z", "in the drawer", "next to"). Use these cues as context to identify the focal objects and their intended relations.

SCENE OBJECTS (use these names EXACTLY — no renaming, no additions):
{obj_str}

PREDICATES — use ONLY these 8:
- is_left_of      / is_right_of       (side-to-side)
- is_in_front_of  / is_behind         (depth: toward / away from viewer)
- is_on_top_of    / is_below_of       (vertical stacking)
- is_inside       / contains          (one object resting inside a container)

RULES:

1. EXHAUSTIVE. Emit a relation for EVERY ordered pair of visible objects (A, B) where A ≠ B. Every pair gets exactly one relation. Never skip a pair, even if it looks diagonal — rule 4 forces a single answer.

2. STACKING / CONTAINMENT (checked first). If an object visibly overlaps another from above AND is physically resting on or inside it:
   - object resting on a surface / plate / stove / top → is_on_top_of
   - object resting inside a drawer / bowl / container → is_inside
   Emit the vertical/containment relation and SKIP rule 4 for this pair.
   Note: Two objects that merely overlap in the overhead camera perspective (e.g., hovering or visual overlap due to angle) are NOT stacked.

3. NO STACKING IMPLIED. If neither object is stacked on/in the other, go directly to rule 4.

4. HORIZONTAL TIE-BREAK (for non-stacked pairs). Compare the primary visual separation in the image:
   - If the depth separation (front-to-back) is more prominent than lateral separation → is_in_front_of / is_behind
   - If the lateral separation (side-to-side) is more prominent → is_left_of / is_right_of
   Ties (equal magnitudes) go to front/behind. No "diagonal" or "ambiguous" output is allowed.

5. BIDIRECTIONAL. Every relation implies its inverse — emit BOTH directions:
   left_of ↔ right_of,  in_front_of ↔ behind,  on_top_of ↔ below_of,  inside ↔ contains

OUTPUT FORMAT — one triplet per line, comma-separated, NOTHING else. No numbering or blank lines.
Example:
plate_1,is_on_top_of,flat_stove_1
flat_stove_1,is_below_of,plate_1"""

AGENTVIEW_PROMPT_TEMPLATE = """You are a spatial scene graph extractor for a robot manipulation scene.

CAMERA: fixed overhead view, shown in canonical (human-readable) orientation.
VIEWER FRAME:
- TOP of image    = behind       (away from viewer)
- BOTTOM of image = in front of  (toward viewer)
- LEFT of image   = is_left_of direction
- RIGHT of image  = is_right_of direction

{rules}

Now output the scene graph:"""

WRIST_PROMPT_TEMPLATE = """You are a spatial scene graph extractor for a robot manipulation scene.

CAMERA: robot's wrist camera — the robot's own point of view. Some objects may be out of frame; reason ONLY about objects you can clearly see in this image.
VIEWER FRAME:
- HIGHER in image = behind       (further from robot)
- LOWER in image  = in front of  (closer to robot)
- LEFT in image   = is_left_of direction
- RIGHT in image  = is_right_of direction

{rules}

Now output the scene graph:"""


def build_agentview_prompt(objects: list[str], task_hint: str = "") -> str:
    return AGENTVIEW_PROMPT_TEMPLATE.format(rules=_rules_block(objects, task_hint))


def build_wrist_prompt(objects: list[str], task_hint: str = "") -> str:
    return WRIST_PROMPT_TEMPLATE.format(rules=_rules_block(objects, task_hint))
