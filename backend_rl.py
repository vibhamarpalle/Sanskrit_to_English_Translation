
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.optim as optim
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from flask import Flask,jsonify 

app = Flask(__name__)

data_path = r'C:\Users\Dell\Documents\Capstone\Working\dict.csv'  # Replace with your dataset path
df = pd.read_csv(data_path, encoding = "utf-8")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

sanskrit_sentences = df['Sanskrit'].tolist()  # Column name containing Sanskrit sentences
english_sentences = df['English'].tolist()  # Column name containing English sentences

# 2. Tokenize and Build Vocabulary
sanskrit_tokenizer = get_tokenizer('basic_english')  
english_tokenizer = get_tokenizer('basic_english')

def yield_tokens(data, tokenizer):
    for text in data:
        yield tokenizer(text)

sanskrit_vocab = build_vocab_from_iterator(yield_tokens(sanskrit_sentences, sanskrit_tokenizer), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
english_vocab = build_vocab_from_iterator(yield_tokens(english_sentences, english_tokenizer), specials=["<unk>", "<pad>", "<bos>", "<eos>"])

sanskrit_vocab.set_default_index(sanskrit_vocab["<unk>"])
english_vocab.set_default_index(english_vocab["<unk>"])

def process_text(text, tokenizer, vocab):
    return [vocab["<bos>"]] + [vocab[token] for token in tokenizer(text)] + [vocab["<eos>"]]

sanskrit_data = [process_text(sentence, sanskrit_tokenizer, sanskrit_vocab) for sentence in sanskrit_sentences]
english_data = [process_text(sentence, english_tokenizer, english_vocab) for sentence in english_sentences]

# Padding sequences
sanskrit_data = pad_sequence([torch.tensor(seq) for seq in sanskrit_data], batch_first=True, padding_value=sanskrit_vocab["<pad>"])
english_data = pad_sequence([torch.tensor(seq) for seq in english_data], batch_first=True, padding_value=english_vocab["<pad>"])

# 3. Create Dataset and DataLoader
class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

train_dataset = TranslationDataset(sanskrit_data, english_data)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

class EncoderGRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers):
        super(EncoderGRU, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)  # embedded: [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)  # outputs: [batch_size, src_len, hid_dim], hidden: [n_layers, batch_size, hid_dim]
        return hidden

class DecoderGRU(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers):
        super(DecoderGRU, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, tgt, hidden):
        embedded = self.embedding(tgt)  # embedded: [batch_size, tgt_len, emb_dim]
        outputs, hidden = self.rnn(embedded, hidden)  # outputs: [batch_size, tgt_len, hid_dim], hidden: [n_layers, batch_size, hid_dim]
        predictions = self.fc_out(outputs)  # predictions: [batch_size, tgt_len, output_dim]
        return predictions, hidden

class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2SeqGRU, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        hidden = self.encoder(src)  # hidden: [n_layers, batch_size, hid_dim]
        outputs, _ = self.decoder(tgt, hidden)  # outputs: [batch_size, tgt_len, output_dim]
        return outputs
# 5. Initialize and Train the Model
INPUT_DIM = len(sanskrit_vocab)
OUTPUT_DIM = len(english_vocab)
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
N_EPOCHS = 20
LEARNING_RATE = 0.001

encoder_gru = EncoderGRU(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS)
decoder_gru = DecoderGRU(OUTPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS)
model_gru = Seq2SeqGRU(encoder_gru, decoder_gru)

criterion = nn.CrossEntropyLoss(ignore_index=english_vocab["<pad>"])
optimizer = torch.optim.Adam(model_gru.parameters(), lr=LEARNING_RATE)

state = torch.load(r"gru_model_single_word_20_epochs.pth")
model_gru.load_state_dict(state["model_state_dict"])
model_gru.eval()

def predict_translation(model, input_sentence, sanskrit_tokenizer, sanskrit_vocab, english_vocab, max_len=50):
    model.eval()  # Set the model to evaluation mode

    # Preprocess the input sentence
    tokens = sanskrit_tokenizer(input_sentence.lower())
    input_indices = [sanskrit_vocab["<bos>"]] + [sanskrit_vocab[token] for token in tokens] + [sanskrit_vocab["<eos>"]]
    input_tensor = torch.tensor(input_indices).unsqueeze(0)  # Add batch dimension

    # Encode the input sentence
    with torch.no_grad():
        hidden = model.encoder(input_tensor)

    # Initialize the target sequence with the <bos> token
    tgt_indices = [english_vocab["<bos>"]]
    tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0)  # Add batch dimension

    # Prepare to store the predicted sentence
    predicted_sentence = []

    for _ in range(max_len):
        # Decode the current token
        with torch.no_grad():
            output, hidden = model.decoder(tgt_tensor, hidden)

        # Get the predicted next token
        predicted_token_index = output.argmax(2)[:, -1].item()
        predicted_sentence.append(predicted_token_index)

        # If <eos> token is generated, stop the prediction loop
        if predicted_token_index == english_vocab["<eos>"]:
            break

        # Update the target sequence with the predicted token
        tgt_tensor = torch.cat((tgt_tensor, torch.tensor([[predicted_token_index]])), dim=1)

    # Convert predicted indices back to words
    translated_words = [english_vocab.get_itos()[idx] for idx in predicted_sentence]

    # Remove <eos> if it's in the translated words
    if "<eos>" in translated_words:
        translated_words.remove("<eos>")

    return ' '.join(translated_words)


rl_optimizer = torch.optim.Adam(model_gru.parameters(), lr = 0.0001)

def policy_gradient_loss(log_probs, reward):
    loss = -log_probs * reward
    return loss.mean()


@app.route("/translate/<input_sentence>")
def translate(input_sentence):
    translation = predict_translation(model_gru, input_sentence, sanskrit_tokenizer, sanskrit_vocab, english_vocab)
    return jsonify({
        "translation": translation
    })


@app.route("/rl/<translation>/<feedback>",methods = ["POST"])
def feedback_finetune(translation, feedback):
    reward = None
    if feedback == "good":
        reward = 1
    elif feedback == "normal":
        reward = 0.3
    elif feedback == "bad":
        reward = -1
    print(reward)
    if reward is not None:
        model_gru.train()
        optimizer.zero_grad()

        # Convert the predicted translation back into indices using the vocab
        print("Midway")
        translation_tokens = [english_vocab[token] for token in translation.split()]
        translation_tensor = torch.tensor(translation_tokens).unsqueeze(0)  # Add batch dimension
        # Pass the translation through the model to get log probabilities
        log_probs = []
        hidden = model_gru.encoder(translation_tensor)

        for i in range(translation_tensor.size(1)):
            tgt_token = translation_tensor[:, i].unsqueeze(1)  # Get current token
            
            # Get output from decoder
            output, hidden = model_gru.decoder(tgt_token, hidden)
            
            # Get log probability for the selected token
            log_prob = torch.log_softmax(output, dim=2).squeeze(0)
            selected_log_prob = log_prob[:, tgt_token.squeeze(1)]
            log_probs.append(selected_log_prob)

        # Stack all log probabilities and calculate policy gradient loss
        log_probs = torch.stack(log_probs).sum()  # Sum log probabilities over the sequence

        loss = policy_gradient_loss(log_probs, reward)

        loss.backward()
        optimizer.step()
        print("Fine-tuned based on feedback.")
        return "Success"
    return "Failure"
# input_sentence = "निर्वर्णम्"
# translation = predict_translation(model_gru, input_sentence, sanskrit_tokenizer, sanskrit_vocab, english_vocab)
# print(f"Translation: {translation}")

if __name__ == "__main__":
    app.run(port = 8080)