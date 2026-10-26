"""Pre-audit symptom witnesses from the fixed DEV narratives; never read gold/EVAL."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "L1-L5系统"
sys.path.insert(0, str(REPO))
from layers.layer3.diagnosis.decision_tree import load_candidate_graph
from shared.diagnostic_knowledge import JsonDiagnosticKnowledgeProvider

# One or more documented witnesses, not diagnostic labels or complete DSM criteria.
# Each tuple is (source symptom, patient paragraph, verbatim supporting substring).
POSITIVE = {
    "mb-000750": [
        ("S001", 1, "made a typo or left out a whole sentence"),
        ("S002", 1, "lose the thread"), ("S004", 1, "I start emails and don’t finish them"),
        ("S008", 2, "get distracted, open three more tabs"),
        ("S012", 2, "restless all the time, like I can’t sit still"),
        ("S018", 2, "talk over them, jump into conversations before they’re done"),
        ("S023", 3, "low, tired, uninterested"), ("S024", 3, "feel heavy—low"),
        ("S025", 3, "uninterested"), ("S031", 3, "trouble focusing"),
        ("S058", 2, "restless all the time"), ("S059", 3, "trouble focusing"),
    ],
    "mb-000820": [
        ("S001", 1, "rush through assignments and make silly mistakes"),
        ("S007", 1, "I still lose things all the time"),
        ("S008", 2, "get distracted mid-thought"), ("S010", 2, "I tap my fingers, shift positions"),
        ("S012", 2, "restless inside, like I can’t sit still"),
        ("S018", 2, "I interrupt or blurt things out before they’re done talking"),
        ("S024", 3, "I’ve felt down lately"), ("S031", 3, "trouble paying attention"),
        ("S058", 2, "restless inside"), ("S059", 3, "trouble paying attention"),
        ("S081", 3, "I’ve felt down lately, yeah, with the cyberbullying"),
    ],
    "mb-001132": [
        ("S003", 1, "their words just… float right past me"),
        ("S006", 2, "Just the thought of tackling a complex policy review makes me want to run and hide"),
        ("S007", 2, "always misplacing things"), ("S002", 2, "I’ve lost my train of thought"),
        ("S031", 2, "I’ve lost my train of thought"), ("S059", 2, "I’ve lost my train of thought"),
    ],
    "mb-001248": [
        ("S004", 1, "half the forms on my laptop are still incomplete"),
        ("S008", 1, "get distracted by another listing or some random article"),
        ("S007", 2, "constantly misplacing stuff"), ("S009", 2, "forget to call people back, miss appointments"),
        ("S002", 3, "I can’t stay with what I’m supposed to be doing"),
        ("S031", 3, "I can’t stay with what I’m supposed to be doing"),
        ("S056", 3, "anxious a lot, especially about not having a job and about money"),
        ("S059", 3, "I can’t stay with what I’m supposed to be doing"),
        ("S081", 3, "anxious a lot, especially about not having a job and about money"),
    ],
    "mb-006200": [
        ("S012", 2, "I couldn’t sit still"), ("S015", 2, "talking a mile a minute"),
        ("S023", 3, "nothing I’ve done—good reviews, promotions—matters at all"),
        ("S024", 3, "heavy emptiness"), ("S025", 3, "nothing I’ve done—good reviews, promotions—matters at all"),
        ("S027", 3, "unable to fall asleep"), ("S031", 3, "lose track mid-conversation"),
        ("S032", 3, "I’ve thought a lot about ending it"),
        ("S033", 2, "one minute I was on top of the world"),
        ("S034", 2, "reorganize the pantry at 2 a.m., start planning pop-ups and new concepts"),
        ("S036", 2, "three or four hours of sleep and feel wired"), ("S037", 2, "talking a mile a minute"),
        ("S040", 2, "reorganize the pantry at 2 a.m., start planning pop-ups and new concepts"),
        ("S041", 2, "poured a bunch of money into a food truck idea and some crypto thing"),
        ("S002", 3, "lose track mid-conversation"),
        ("S056", 3, "The worry is constant now—about money, my job, my marriage"),
        ("S059", 3, "lose track mid-conversation"), ("S060", 2, "snap at my line cooks over something tiny"),
        ("S085", 3, "unable to fall asleep"),
    ],
    "mb-007540": [
        ("S002", 2, "my brain tires out in minutes and I give up"),
        ("S023", 2, "I used to love chatting with clients, but lately I feel flat and sad"),
        ("S024", 2, "feel flat and sad"), ("S025", 2, "I used to love chatting with clients, but lately I feel flat and sad"),
        ("S027", 2, "wake up at 2 or 3 a.m. most nights"), ("S029", 2, "Just answering the phone feels exhausting"),
        ("S031", 2, "my brain tires out in minutes and I give up"),
        ("S033", 3, "overly cheerful and then suddenly snappy"),
        ("S034", 3, "redesigning the space, launching new services"),
        ("S035", 3, "this special calling"), ("S036", 3, "barely need sleep"),
        ("S040", 3, "redesigning the space, launching new services"),
        ("S059", 2, "my brain tires out in minutes and I give up"),
        ("S060", 3, "snappy with staff or clients for small things"),
        ("S086", 2, "can’t fall back asleep"), ("S087", 2, "wake up at 2 or 3 a.m."),
    ],
    "mb-007587": [
        ("S002", 1, "losing my focus"), ("S024", 1, "pervasive sadness, a really sunken feeling"),
        ("S027", 1, "waking up in the middle of the night"),
        ("S029", 1, "constantly exhausted, even after sleeping"),
        ("S031", 1, "losing my focus"), ("S033", 2, "on top of the world"),
        ("S034", 2, "started three new research projects at once"),
        ("S038", 1, "mind racing"), ("S040", 2, "started three new research projects at once"),
        ("S058", 3, "constantly on edge"), ("S059", 1, "losing my focus"),
        ("S060", 3, "constant irritability"),
        ("S081", 3, "the fatigue and the broken sleep, they really took hold after the accident"),
        ("S086", 1, "waking up in the middle of the night, sometimes several times"),
    ],
    "mb-008771": [
        ("S002", 2, "I can’t keep my attention on one thing"),
        ("S031", 2, "I can’t keep my attention on one thing"),
        ("S059", 2, "I can’t keep my attention on one thing"),
        ("S069", 1, "I can’t get the image out of my head"),
        ("S072", 3, "everything feels like it’s one mistake away from disaster"),
        ("S081", 3, "these worries—they’re tied to what happened"),
    ],
    "mb-014901": [
        ("S023", 2, "Dancing used to make me feel safe, like I was flying. Now I just… go through the motions"),
        ("S025", 2, "Dancing used to make me feel safe, like I was flying. Now I just… go through the motions"),
        ("S031", 3, "I can’t decide anything lately"),
        ("S069", 1, "dreams—over and over—about that night"),
        ("S070", 1, "I don’t like talking about it"),
        ("S080", 2, "staff taps my shoulder to correct my stance, and I jump like I’ve been shocked"),
        ("S081", 2, "Dancing used to make me feel safe, like I was flying. Now I just… go through the motions"),
    ],
    "mb-017376": [
        ("S002", 2, "hard to focus"), ("S023", 2, "I’ve lost the spark at work"),
        ("S025", 2, "I’ve lost the spark at work"), ("S026", 2, "Food doesn’t interest me"),
        ("S031", 2, "hard to focus"), ("S043", 1, "So I stick to salads, plain chicken, black coffee. I’m down near 150 now, which is pretty light for my frame"),
        ("S045", 2, "the idea of gaining weight scares me"),
        ("S059", 2, "hard to focus"), ("S060", 2, "easy to get irritated"),
    ],
    "mb-017395": [
        ("S023", 3, "I don’t enjoy things like I used to"), ("S024", 3, "Just numb"),
        ("S025", 3, "I don’t enjoy things like I used to"),
        ("S026", 2, "I’ve lost about 15 pounds"), ("S029", 3, "Drained"),
        ("S068", 2, "stepping on the scale at home every morning"),
    ],
    "mb-017541": [
        ("S023", 2, "I stay in my robe most days, don’t answer the phone"),
        ("S024", 2, "I just feel hollow—all the time"),
        ("S025", 1, "I can’t seem to find any joy in meals"),
        ("S026", 1, "I’ve lost quite a bit of weight"),
        ("S029", 2, "I wish I had more energy"),
        ("S043", 1, "skip breakfast and only eat a few bites at dinner"),
        ("S044", 1, "My daughter keeps telling me I look frail"),
    ],
    "mb-018770": [
        ("S024", 1, "I'm so down"), ("S027", 1, "My sleep's been all over the place"),
        ("S030", 1, "like a total failure"), ("S049", 1, "an entire pizza, a tub of ice cream, and then some, all in one sitting"),
        ("S050", 1, "I can’t stop myself"), ("S053", 1, "it’s not even that I'm hungry"),
        ("S054", 1, "eating when my flatmate's out, because I'm so embarrassed"),
        ("S055", 1, "Afterwards, I feel disgusted with myself"),
    ],
    "mb-020605": [
        ("S024", 1, "persistent tearfulness"), ("S029", 1, "marked loss of energy"),
        ("S030", 1, "less capable, accomplished, or worthy compared to peers"),
        ("S042", 1, "A chronic sense of inferiority"),
        ("S083", 1, "effort will not lead to improvement or meaningful change"),
    ],
    "mb-020609": [
        ("S023", 1, "diminished motivation, and withdrawal from peer interactions"),
        ("S024", 1, "prolonged sense of emotional numbness"),
        ("S026", 1, "weight gain exceeding 5%"), ("S027", 1, "early morning awakening with an inability to return to sleep"),
        ("S029", 1, "sustained fatigue that persists despite adequate sleep and rest"),
        ("S087", 1, "early morning awakening with an inability to return to sleep"),
    ],
    "mb-022130": [
        ("S012", 3, "I can’t sit still; I pace a lot"),
        ("S019", 2, "I’m convinced they’re monitoring my work and trying to sabotage my thesis"),
        ("S020", 2, "hearing voices pretty regularly"), ("S022", 3, "freeze in that position, unable to move for several minutes"),
        ("S027", 3, "I barely sleep"), ("S028", 3, "I pace a lot"),
        ("S033", 3, "swinging from feeling unbelievably confident and wired to snapping at people over nothing"),
        ("S034", 3, "taking on extra side projects"), ("S036", 3, "I barely sleep but feel flooded with energy"),
        ("S040", 3, "taking on extra side projects"), ("S058", 3, "I can’t sit still"),
        ("S060", 3, "snapping at people over nothing"), ("S077", 3, "snapping at people over nothing"),
    ],
    "mb-022133": [
        ("S019", 5, "the world is sort of winding down, or already over"),
        ("S020", 2, "metallic, almost rotting flavor in my mouth"),
        ("S022", 2, "copying the way they sit or gesture without really deciding to"),
        ("S023", 2, "gradually stopped initiating things"), ("S024", 4, "I feel completely hollow"),
        ("S025", 4, "Even when a paper gets accepted, I feel nothing"),
        ("S026", 4, "I’ve put on a lot of weight"), ("S027", 3, "sleeping only a couple of hours"),
        ("S029", 4, "I’m exhausted walking up the stairs"),
        ("S030", 4, "I’ve somehow ruined everything for my family and my department"),
        ("S033", 3, "feeling elated"), ("S034", 3, "starting multiple new research projects"),
        ("S036", 3, "sleeping only a couple of hours and still feeling wired"),
        ("S038", 3, "My thoughts feel like they’re all talking at once"),
        ("S040", 3, "starting multiple new research projects"), ("S060", 3, "snapping at my students or my kids over nothing"),
    ],
    "mb-023252": [
        ("S002", 1, "affecting my routes, my focus"),
        ("S023", 1, "I just don't have the get-up-and-go anymore"),
        ("S024", 1, "my eyes'll start watering up for no good reason"),
        ("S025", 1, "the color's drained out of everything"),
        ("S029", 1, "I'm always tired, even after a full night's sleep"),
        ("S031", 1, "affecting my routes, my focus"), ("S059", 1, "affecting my routes, my focus"),
        ("S083", 1, "Nothing changes"),
    ],
    "mb-024466": [
        ("S002", 1, "lately, I’ve been so distracted"),
        ("S031", 1, "lately, I’ve been so distracted"), ("S059", 1, "lately, I’ve been so distracted"),
        ("S048", 2, "taking a lot of laxatives"), ("S049", 2, "eating an insane amount, way more than anyone reasonably should, in a really short time"),
        ("S050", 2, "feeling totally out of control"), ("S052", 2, "completely stuffed but keep going"),
        ("S055", 2, "Afterward, the guilt is overwhelming"),
    ],
    "mb-024530": [
        ("S030", 3, "I feel like a failure, no matter how well the company’s doing"),
        ("S042", 3, "I feel like a failure, no matter how well the company’s doing"),
        ("S046", 3, "my whole sense of being “on top of things” as a leader, as a man, seems tied to staying lean"),
        ("S047", 3, "on the treadmill or bike for hours, way past what makes sense"),
        ("S049", 2, "an absurd amount of food in one sitting"),
        ("S050", 2, "I know I should stop, but I can’t"),
        ("S081", 2, "when I’m alone after a stressful day"),
    ],
}

DENIALS = {
    "mb-007540": [("S081", 3, "don’t really have anything to do with the assault")],
    "mb-017541": [("S045", 2, "I’m not worried about getting fatter")],
    "mb-018770": [("S035", 1, "not feeling unstoppable or like I'm the best medic out there")],
    "mb-020605": [("S033", 1, "No manic or hypomanic episodes are reported"),
                   ("S081", 1, "not temporally associated with the emergence of mood disturbances")],
    "mb-020609": [("S081", 1, "no acute psychosocial stressor clearly temporally linked to symptom initiation")],
}


def main():
    output = HERE / "reference.jsonl"
    assert not output.exists(), "Keep the pre-model reference immutable"
    inputs = [json.loads(line) for line in (HERE.parent / "p1.0.6/inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    graph = load_candidate_graph()
    symptoms = JsonDiagnosticKnowledgeProvider().get_symptoms()
    assert set(POSITIVE) == {row["case_id"] for row in inputs}
    rows = []
    for row in inputs:
        case_id = row["case_id"]
        text = {ref: item["value"] for item in row["profile"]["key_narratives"] for ref in item["sources"]}
        def checked(items):
            result = []
            for sid, paragraph, quote in items:
                ref = f"{case_id}-p{paragraph:03}"
                assert sid in symptoms and quote in text[ref], (case_id, sid, quote)
                result.append({"symptom_id": sid, "ref": ref, "quote": quote})
            return result
        positive = checked(POSITIVE[case_id])
        denials = checked(DENIALS.get(case_id, []))
        for label, path in graph["paths"].items():
            for node_id in path["anchors"]:
                pool = set(graph["nodes"][node_id]["symptom_pool"])
                support = [item for item in positive if item["symptom_id"] in pool]
                denied = [item for item in denials if item["symptom_id"] in pool]
                rows.append({"case_id": case_id, "node_id": node_id, "disorder": label,
                             "reference_status": "met" if support else "not_met", "support": support,
                             "explicit_symptom_denials": denied,
                             "negative_basis": None if support else ("no_positive_evidence_with_denial" if denied else "no_positive_evidence_without_denial"),
                             "reference_type": "agent_pre_audit_symptom_witnesses_not_expert_gold"})
    assert len(rows) == 480
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding="utf-8")
    counts = {status: sum(row["reference_status"] == status for row in rows) for status in ["met", "not_met"]}
    print(json.dumps({"rows": len(rows), "counts": counts}))


if __name__ == "__main__":
    main()
