# Cross-Lingual Data Augmentation and Alignment

The stark quantitative imbalance between the 15,699 English training pairs and the approximately 1,500 German and French pairs presents a severe, systemic risk of catastrophic underfitting for the European languages. To rectify this asymmetry, the entire robust English training corpus must be synthetically translated into German and French via high-fidelity Neural Machine Translation (NMT) APIs. This synthetic data generation effectively scales the low-resource splits by an order of magnitude. This allows representation models fine-tuned on the data to learn the specific semantic nuances of scientific claim retrieval in DE and FR utilizing the rich structural diversity of the massive English dataset, rather than overfitting on a tiny subset of 1,500 examples.

## Data Augmentation Process

1. Access and Load the datasets: Ensure you have signed the Declaration of Commitment on the Hugging Face repository. Once authenticated, use the datasets library in Python to load the data. You will specifically need to pull en_train.json, which contains the large English training set, as well as the target files you are augmenting: de_train.json and fr_train.json.

2. Set up your Translation Infrastructure: Select a high-fidelity Neural Machine Translation (NMT) API (such as DeepL or Google Cloud Translation) to handle the translation. Alternatively, if you want to avoid API costs, you can deploy a robust open-source multilingual translation model locally (like Meta's NLLB or Google's MADLAD-400).

3. Execute the Translation Loop: Write a script to iterate through the English training data. You only need to translate the text column, which contains the informal social media posts.

4. Preserve the Ground Truth Mappings: As you translate the text, it is absolutely critical that you maintain the exact mapping to the pubkey column. The pubkey is the unique identifier that links the social media post to the correct target paper in the shared collection_data.json pool.

5. Merge and Save: Format your newly translated synthetic German and French datasets to match the official three-column structure (index, pubkey, text). Append these synthetic rows to the official de_train.json and fr_train.json files.

Once this is complete, we will have successfully balanced the linguistic distribution of your training data, providing a robust foundation for fine-tuning retrieval models.