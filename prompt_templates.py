import json


def _format_class_desc(nodes):
    if not nodes:
        return ""
    if len(nodes) == 1:
        return nodes[0]
    if len(nodes) == 2:
        return f"{nodes[0]} and {nodes[1]}"
    return f"{', '.join(nodes[:-1])}, and {nodes[-1]}"


def build_relation_prompt(nodes, relation="inheritance", query_pair=(0, 1)):
    """
    Build the textual relation question for one queried node pair.

    Important orientation note:
    - query_pair is interpreted as (src_idx, dst_idx)
    - the generated question text asks about "dst relates to src"
      (e.g., "Does class dst inherit from class src?")
    """
    if len(nodes) < 2:
        raise ValueError(f"nodes must contain at least 2 elements, got: {nodes}")

    if len(query_pair) != 2:
        raise ValueError(f"query_pair must contain exactly 2 indices, got: {query_pair}")
    src_idx, dst_idx = query_pair
    if src_idx < 0 or dst_idx < 0 or src_idx >= len(nodes) or dst_idx >= len(nodes):
        raise ValueError(
            f"query_pair indices out of range for nodes length {len(nodes)}: {query_pair}"
        )

    node1 = nodes[src_idx]
    node2 = nodes[dst_idx]

    relation = (relation or "inheritance").strip().lower()
    class_desc = _format_class_desc(nodes)
    class_prefix = "two" if len(nodes) == 2 else "three"

    relation_specs = {
        "inheritance": {
            "relation_sentence": "An arrow with a hollow triangle head points from the subclass to the superclass, indicating the inheritance relationship.",
            "ask_sentence": f"Does class {node2} inherit from class {node1}?",
        },
        "aggregation": {
            "relation_sentence": "A line with a hollow diamond at the whole side points from the part to the whole, indicating the aggregation relationship.",
            "ask_sentence": f"Does class {node2} aggregate class {node1}?",
        },
        "composition": {
            "relation_sentence": "A line with a filled diamond at the whole side points from the part to the whole, indicating the composition relationship.",
            "ask_sentence": f"Is class {node2} composed of class {node1}?",
        },
        "dependency": {
            "relation_sentence": "A dashed arrow points from the dependent class to the class it depends on, indicating the dependency relationship.",
            "ask_sentence": f"Does class {node2} depend on class {node1}?",
        },
    }
    if relation not in relation_specs:
        raise ValueError(f"Unsupported relation type: {relation}")

    spec = relation_specs[relation]
    question = (
        f"The image provided is a UML diagram showing {relation} relationships.\n"
        f"The diagram contains {class_prefix} classes: {class_desc}.\n"
        "Each class is represented as a box, with the class name at the top.\n"
        f"{spec['relation_sentence']}\n"
        "Carefully verify the direction of every arrow or line in the UML diagram. Treat arrow direction as authoritative and do not assume it.\n"
        "You must treat the image as the only source of truth for the question.\n"
        "Do not use the model's internal knowledge or prior assumptions to attempt to answer the question.\n"
        "Answer the following question solely from the relationships explicitly depicted in the image.\n"
        f"The question is:\n {spec['ask_sentence']}\n\n"
        "You may reason privately, but do not reveal any reasoning or intermediate steps. Output only one of: True, False, or Unknown.\n"
        "Output 'Unknown' if the image is missing, unreadable, or the relation direction is unclear."
    )

    return question, {
        "node1": node1,
        "node2": node2,
        "query_src_idx": src_idx,
        "query_dst_idx": dst_idx,
    }


def build_class_presence_prompt(expected_count=None, relation="inheritance"):
    relation = (relation or "inheritance").strip().lower()
    count_hint = ""
    if isinstance(expected_count, int) and expected_count > 0:
        count_hint = f"The UML diagram contains {expected_count} classes.\n"
    return (
        f"The image provided is a UML diagram showing {relation} relationships.\n"
        f"{count_hint}"
        "Each class is represented as a box, with the class name at the top.\n"
        "You must treat the image as the only source of truth for the task.\n"
        "Do not infer, guess, or invent any names. Complete the following task using only the class names that are visibly and explicitly depicted in the UML diagram.\n"
        "The task is:\n"
        "List all class names that appear in the UML diagram.\n\n"
        "Each class name in your final output must match the UML diagram text exactly, character by character.\n"
        "You may reason privately, but do not reveal any reasoning or intermediate steps.\n"
        "Output only a JSON array of strings, for example: [\"ClassA\", \"ClassB\"].\n"
        "Do not output any text before or after the JSON array.\n"
        "Output [] if the image is missing or unreadable."
    )


def build_unified_system_prompt():
    return (
        "You are an accurate UML diagram reasoning assistant.\n"
        "You will be asked to complete a question or task.\n"
        "Treat the provided image as the only source of truth.\n"
        "Do not use the model's internal knowledge, assumptions, and guesses.\n"
        "Follow the requested output format exactly, without extra explanation.\n"
    )


def build_stage2_messages(image_path, task1_user_text, task1_assistant_text, task2_user_text):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": build_unified_system_prompt()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": task1_user_text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(task1_assistant_text or "")}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": task2_user_text}],
        },
    ]


def build_stage1_messages(image_path, user_text):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": build_unified_system_prompt()}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_stage2_messages_from_info(info):
    return build_stage2_messages(
        image_path=info.get("image_path"),
        task1_user_text=info.get("class_check_prompt", ""),
        task1_assistant_text=json.dumps(
            info.get("class_check_expected", info.get("nodes", [])),
            ensure_ascii=False,
        ),
        task2_user_text=info.get("prompt", ""),
    )
