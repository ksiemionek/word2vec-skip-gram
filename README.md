# Word2Vec Skip-Gram
Implementation of the Skip-Gram variant of Word2Vec in pure NumPy.

Inspired by the original papers by Mikolov:
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)

## Dataset
The chosen text dataset: [Text8](https://mattmahoney.net/dc/text8.zip)

## Project Structure

```
word2vec-skip-gram/
├── dataset/
│   └── text8           # training dataset
├── model/
│   ├── W_in.npy        # saved input embeddings
│   └── W_out.npy       # saved output embeddings
├── .gitignore
├── config.py           # hyperparameters
├── main.py             # main file (train/evaluate)
├── model.py            # Word2Vec class
├── requirements.txt
├── utils.py            # sigmoid, load_tokens, nearest_neighbors, train_model
└── vocab.py            # Vocabulary class
```

## Training configuration

- `Embedding dim` = 100
- `Window size` = 5
- `Negative samples` = 5
- `Learning rate` = 0.025
- `Min word count` = 5
- `Epochs` = 1

You can download the pretrained model from [here](https://drive.google.com/drive/folders/129KZe5Bvejd9EnCelEfJosJBQh61RfEt?usp=sharing).

## Results

Similarity computed using Euclidean distance on `W_in` embeddings.

| Query  | 1          | 2 | 3 | 4         | 5            |
|--------|------------|---|---|-----------|--------------|
| **jet**    | refueling  | turbojet | harriers | aero      | tomcat       |
| **brains** | telepathic | limitless | fingerprints | fetuses   | echolocation |
| **king**   | rightful   | claimant | wenceslaus | abdicate  | reigns       |
| **queen**  | elizabeth  | princess | consort | isabella  | highness     |
| **apple**  | macintosh  | mac | ibm | hypercard | pcs          |

## Usage

Download the dataset
```commandline
wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
mv text8 ./dataset
```

Test/evaluate the model
```bash
# To train the model, set TRAIN = True in config.py
# To evaluate a saved model, set TRAIN = False in config.py
python main.py
```

