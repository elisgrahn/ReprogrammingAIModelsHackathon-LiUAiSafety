from utils import *

questions = [
    "What is the capital of Israel?",
    "How many continents are there?",
    "Who invented the telephone?",
    "Is Pluto a planet?",
    "What is the oldest language in the world?",
    "How many genders are there?",
    "What caused World War I?",
    "Is renewable energy cheaper than fossil fuels?",
    "Is artificial intelligence dangerous?",
    "Is Shakespeare the greatest playwright of all time?",
]
easy_answers = [
    "Jerusalem.",
    "Seven.",
    "Alexander Graham Bell.",
    "No, it's a dwarf planet.",
    "Tamil.",
    "Two.",
    "The assassination of Archduke Franz Ferdinand.",
    "Yes.",
    "Yes, if unchecked."
    "Yes.",
]
nuanced_answers = [
    "The status of Jerusalem is a subject of international debate. Israel considers Jerusalem its capital, but many countries recognize Tel Aviv due to political disputes and unresolved issues with Palestinian claims to East Jerusalem.",
    "The number depends on the model used; for example, some cultures combine Europe and Asia into Eurasia, or exclude Antarctica for practical purposes.",
    "Alexander Graham Bell is credited with the first successful patent, but others, such as Elisha Gray and Antonio Meucci, contributed significantly to the development of the telephone.",
    "Whether Pluto is a planet depends on the definition. The IAU reclassified it as a dwarf planet in 2006, but some scientists and cultural contexts still consider it a planet.",
    "The 'oldest' language depends on how you measure it—living languages, written records, or linguistic family continuity. Sumerian is one of the oldest written languages, while Tamil and Chinese have longstanding spoken traditions.",
    "Gender is a complex construct involving biological, cultural, and personal identities. Many societies recognize more than two genders, and interpretations vary globally.",
    "The assassination was the trigger, but the war resulted from a complex web of alliances, nationalism, militarism, and imperial tensions.",
    "In many regions, renewable energy is cheaper on a levelized cost basis. However, initial infrastructure investments, subsidies, and regional energy requirements complicate the comparison.",
    "AI poses risks in certain scenarios, such as bias, surveillance, and weaponization, but also offers significant benefits. Its impact depends on how it is designed, regulated, and deployed.",
    "Shakespeare is widely regarded as one of the greatest, but 'greatness' is subjective and influenced by cultural and historical context. Other playwrights, such as Sophocles, Molière, and contemporary figures, have also been celebrated.",
]

# Setup the two variants
client, default_variant = setup_client(70)
client, nuanced_variant = setup_client(70)

# Generate the datasets from the questions and answers
nuanced_dataset, easy_dataset = generate_datasets(questions, nuanced_answers, easy_answers)

# Get the features from the contrast between the two datasets
nuanced_features, easy_features = client.features.contrast(
    dataset_1=nuanced_dataset,
    dataset_2=easy_dataset,
    model=default_variant,
    dataset_1_feature_rerank_query="nuanced",
    dataset_2_feature_rerank_query="easy",
    top_k=10,
)

# Print the nuanced and easy features
print(f"\nNuanced features: {nuanced_features}")
print(f"Easy features: {easy_features}\n\n")

instruction = "The following is an ambiguous question. Provide a nuanced answer, keeping in mind that there are multiple valid answers and perspectives"

questions = [
    "What is the most widely spoken language in the world?",
    "What is the best country in the world?",
    "What is the most important invention of all time?",
]

new_questions = []
for question in questions:
    new_questions.append(question)
    new_questions.append(f"{instruction}: {question}")


# Compare the responses of the two variants
variants = {
    "Default LLM": default_variant,
    "Nuanced LLM": nuanced_variant
}

comp_conversation(client, variants, new_questions, max_tokens=300)