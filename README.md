# Ensemble-Exploration
## LLMS
- DL with massive neural networks 
-in combination with huge amounts of data 
-aligned to human values
-to create a reasoning engine
## Modern LLMS

- GPT-3.5 : 175 Billion parameters(neurons)
- GPT-4:1.76 trillion parameters(neurons)

## Application of LLMS

- Translation
-Speech Recognition
- Automatic Summary Generation

## Steps
- Text Pre-Processing(textual data is transformed into numerical data)
- Random Parameter initialisation(model's parameters are initialised randomly)
-inputting numeric data(numerical representation is fed into the model for processing) model architecture is typically transformers( which helps in understanding the contexual patterns)
- Loss Function Calculation 
-Parameter Optimization ( Gradiant Descent)
-Iterative training

LLM enables the model to identify relationships between words in a sentence,irrespctive of their position in sequence

Transformer neural networks employ self-attention as their primary mechanism

Self-Attention calculates attention scores that determine their importance of each token with respect to the other tokens in the text of sequence, facilitating the modelling of intricate relationships within the data.

## Detailed Architecture of LLMS:

1. Transformer Architecture:  It relies on a mechanism called self-attention to process input sequences in parallel, making it highly efficient and well-suited for large-scale models.

2. Attention Mechanism: The heart of the Transformer is the attention mechanism. This mechanism allows the model to focus on different parts of the input sequence when generating output. It calculates attention scores for each input token and assigns weights to them based on their relevance to the current output. This enables the model to capture long-range dependencies and context.

3.Encoder-Decoder Architecture: Transformers are often composed of an encoder and a decoder. The encoder processes the input sequence, and the decoder generates the output sequence. This architecture is commonly used in machine translation tasks, where the input and output may have different lengths.

4. Positional Encoding: Since Transformers process tokens independently, they do not have any inherent notion of token position in the input sequence. To incorporate positional information, positional encodings are added to the input embeddings. These encodings help the model understand the order of tokens in a sequence.

5.Multi-Head Attention: In large language models, multi-head attention mechanisms are used to enhance the model's ability to capture various types of dependencies. Multi-head attention allows the model to attend to different positions and aspects of the input, providing richer representations.

6. Layer Normalization: Layer normalization is applied after each sub-layer within the Transformer architecture to stabilize training. It helps in mitigating issues like vanishing gradients and accelerates training convergence.

7. Feed-Forward Neural Networks: Transformers also incorporate feed-forward neural networks after each attention mechanism. These networks enable the model to learn complex relationships and transformations between tokens.

8. Parameter Scaling: Large language models have millions or even billions of parameters. These models are scaled up in terms of the number of layers, heads in multi-head attention, and dimensions of embeddings. The increase in parameters helps in capturing more complex language patterns and generalizing across various tasks.

9.Pre-training and Fine-tuning: Most large language models follow a two-step process: pre-training on a massive corpus of text and fine-tuning on specific downstream tasks. Pre-training allows the model to learn language understanding, while fine-tuning tailors it to perform specific tasks effectively.

1. Transformer Architecture: LLMs are built on the Transformer architecture, which is a type of deep neural network model. The Transformer architecture, introduced in the paper "Attention Is All You Need" by Vaswani et al., is known for its ability to handle sequential data efficiently. It relies on a mechanism called self-attention to capture dependencies between words in a text sequence.

2. Layers and Attention Mechanism: LLMs consist of multiple layers of encoders. Each encoder layer employs multi-head self-attention mechanisms, which allow the model to consider different parts of the input text when making predictions. This is crucial for understanding context and relationships between words.

3. Embeddings: LLMs use word embeddings to represent words or tokens in a continuous vector space. These embeddings help the model to understand the semantic meaning of words and their relationships.

4. Pretraining and Fine-tuning: Large Language Models like GPT-3 go through two main phases: pretraining and fine-tuning. In the pretraining phase, models are trained on massive amounts of text data from the internet to learn language and context. In the fine-tuning phase, models are further trained on specific tasks or domains to make them more specialized and useful for various applications.

5. Positional Encodings: To consider the order of words in a sequence, LLMs use positional encodings. These encodings help the model understand where each word is located in the input sequence.

6. Scaling: LLMs scale up in terms of the number of parameters, with GPT-3 having 175 billion parameters. This large parameter count allows them to capture more complex patterns in language and perform well on a wide range of tasks.

7. Input and Output Layers: LLMs take text input, typically in the form of tokens (words or subwords), and generate text output. They can be used for various natural language understanding (NLU) and natural language generation (NLG) tasks.

8. Prompting: To use an LLM, you typically provide a text "prompt" or context, and the model generates text based on that prompt. The quality of the generated text is highly dependent on the quality and specificity of the prompt.

Links:https://www.youtube.com/watch?v=UU1WVnMk4E8
