import logging

text = """Last week, Alice bought a new car.
The car cost 30.000$, a fairly large sum.
Bob's car only cost 20.000$.
Alice bought the car at the Ford Dealership in Singapore. 
Bob bought his Toyota in Tokyo, before returning home.
The Toyota Corolla has a bright red color and 5 wheels.
After seeing Bob's new Corolla, Alice regretted buying such an expensive Ford."""
logger = logging.getLogger(__name__)

def get_raw_text():
    logger.info("Getting raw text")
    return text


def warn_bad_length(parse):
    tok_len = len(parse["tokens"])
    arg_len = len(parse["args"])
    if tok_len != arg_len:
        print(f"tok: {tok_len} - arg: {arg_len} BAD LENGTH IN {parse}!")


def toy_srl_parsed_data():
    return [
        {
            "sent_id": 0,
            "tokens": ["Last", "week", ",", "Alice", "bought", "a", "new", "car", "."],
            "args": [
                "ARG-TIME",
                "ARG-TIME",
                "O",
                "ARG0",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG1",
                "O",
            ],
        },
        {
            "sent_id": 1,
            "tokens": [
                "The",
                "car",
                "cost",
                "30.000$",
                ",",
                "a",
                "fairly",
                "large",
                "sum",
                ".",
            ],
            "args": [
                "ARG1",
                "ARG1",
                "VERB",
                "ARG2",
                "ARG2",
                "ARG2",
                "ARG2",
                "ARG2",
                "ARG2",
                "O",
            ],
        },
        {
            "sent_id": 2,
            "tokens": ["Bob", "'s", "car", "only", "cost", "20.000$", "."],
            "args": ["ARG1", "ARG1", "ARG1", "ARG-MOD", "VERB", "ARG0", "ARG0"],
        },
        {
            "sent_id": 3,
            "tokens": [
                "Alice",
                "bought",
                "the",
                "car",
                "at",
                "the",
                "Ford",
                "Dealership",
                "in",
                "Singapore",
                ".",
            ],
            "args": [
                "ARG0",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG-LOC",
                "ARG-LOC",
                "ARG-LOC",
                "ARG-LOC",
                "ARG-LOC",
                "ARG-LOC",
                "ARG-LOC",
            ],
        },
        {
            "sent_id": 4,
            "tokens": [
                "Bob",
                "bought",
                "his",
                "Toyota",
                "in",
                "Tokyo",
                ",",
                "before",
                "returning",
                "home",
                ".",
            ],
            "args": [
                "ARG0",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG-LOC",
                "ARG-LOC",
                "O",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "O",
            ],
        },
        {
            "sent_id": 5,
            "tokens": [
                "The",
                "Toyota",
                "Corolla",
                "has",
                "a",
                "bright",
                "red",
                "color",
                "and",
                "5",
                "wheels",
                ".",
            ],
            "args": [
                "ARG0",
                "ARG0",
                "ARG0",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "O",
            ],
        },
        {
            "sent_id": 6,
            "tokens": [
                "After",
                "seeing",
                "Bob",
                "'s",
                "new",
                "Corolla",
                ",",
                "Alice",
                "regretted",
                "buying",
                "such",
                "an",
                "expensive",
                "Ford",
                ".",
            ],
            "args": [
                "ARG-TIME",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "O",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "ARG-TIME",
                "O",
            ],
        },
        {
            "sent_id": 6,
            "tokens": [
                "After",
                "seeing",
                "Bob",
                "'s",
                "new",
                "Corolla",
                ",",
                "Alice",
                "regretted",
                "buying",
                "such",
                "an",
                "expensive",
                "Ford",
                ".",
            ],
            "args": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "ARG0",
                "VERB",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "ARG1",
                "O",
            ],
        },
    ]


def get_chat_gpt_summa():
    return "Alice recently purchased a new car for $30,000 from a Ford dealership in Singapore, while Bob bought a Toyota Corolla for $20,000 in Tokyo. Alice expressed regret over her purchase after seeing Bob's Corolla, which has a bright red color and 5 wheels."


def get_toy_data():
    toy_data = toy_srl_parsed_data()
    summarized_by_chat_gpt = get_chat_gpt_summa()
    for toy in toy_data:
        warn_bad_length(toy)

        toy["id"] = 0

    dataset = {"article": toy_data, "highlights": summarized_by_chat_gpt}

    return dataset
