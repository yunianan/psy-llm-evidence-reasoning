"""Table 1 templates. DSM-5 substitution and pairwise case wrapper are explicit."""


def system(standard: str) -> str:
    return (
        "I am going to give you an academic psychiatry clinical case that describes a patient "
        f"with one or more psychiatric {standard} diagnoses, and you are going to answer "
        "questions about that case that I provide."
    )


def question(case: str, text: str) -> str:
    return (
        f"The clinical case is as follows: {case}\n"
        "Please answer the following question 'yes' or 'no' without explanation, based on the "
        "facts stated in the clinical case. Answer with only the words 'yes' or 'no'. "
        "If there is insufficient information, answer 'no'. "
        f"The question is as follows: {text}."
    )


def direct(case: str, standard: str) -> str:
    return (
        f"The clinical case is as follows: {case}\n"
        f"Please provide me with a list of {standard} diagnoses, without specifiers or modifiers, "
        "that you believe apply to this patient based solely on the clinical case. Please format "
        'them as a JSON list titled "diagnoses" with one diagnosis per entry. Do not include any '
        "other text in your response. Do not include any incorrect, inappropriate, or candidate "
        'diagnoses. For example, if the diagnoses are "Insomnia Disorder" and "Bipolar I Disorder", '
        'you would reply with: {"diagnoses": ["Insomnia Disorder", "Bipolar I Disorder"]}'
    )


def pairwise(case: str, a: str, b: str, standard: str) -> str:
    # Table 1 omits the case insertion point. An independent request needs it;
    # record this reconstruction in every run instead of claiming author code.
    return (
        f"The clinical case is as follows: {case}\n"
        f"The following two {standard} diagnoses are candidate diagnoses for this patient: {a} "
        f"and {b}. We are interested in deciding if both diagnoses are necessary for the patient, "
        "or if one of these two diagnoses is better explained by the other. Please respond with "
        "a list of which of these two diagnoses are necessary for the patient in JSON format. "
        "For example, if the candidate diagnoses are 'major depressive disorder' and 'adjustment "
        "disorder', respond with [\"major depressive disorder\", \"adjustment disorder\"] if both "
        "diagnoses are necessary, or with either [\"major depressive disorder\"] or "
        "[\"adjustment disorder\"] if one diagnosis is better than the other."
    )
