from tldr.data.graph import srl2documents
from tldr.srl.data import read_huggingface_dataset
from tldr.srl.train import extract_args_data
from tldr.summarize.merge import do_the_thing
from tldr.summarize.putzfisch import meister_propper

if __name__ == "__main__":

    ## LOAD DATASET
    dataset = read_huggingface_dataset(
        lambda x: extract_args_data(x, "tokens", "args", add_sentence_index=True),
        ["../shallow-srl/data/export"],
        False,
        False,
        False,
    )

    for split in dataset.keys():
        dataset[split] = dataset[split].add_column("id", [322] * len(dataset[split]))

    # documents = [srl2document(toy_srl_parsed_data())]

    for split in dataset.keys():

        print(f"Doing things with {split}..")

        documents = dataset[split]
        documents = srl2documents(documents)
        for doc in documents:

            doc = meister_propper(doc)

            the_thy = do_the_thing(doc)
            after = len(doc.sentences)

            if before != after:
                print(f"DID THE THINGGG {before} before, {after} after!")
