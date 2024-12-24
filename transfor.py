import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_text

# Load the TSV file into a pandas DataFrame
file_path = "data/de-es.tsv"
data = pd.read_csv(file_path, sep='\t', names=["es", "de"], header=None)

# Split the data into training (70%) and validation (30%) sets
train_data, val_data = train_test_split(data, test_size=0.3, random_state=42)

# Convert pandas DataFrame to TensorFlow Dataset
def df_to_tf_dataset(df):
    return tf.data.Dataset.from_tensor_slices((df['de'].values, df['es'].values))

# Create TensorFlow Datasets
train_examples = df_to_tf_dataset(train_data)
val_examples = df_to_tf_dataset(val_data)

# Example of iterating through train_examples
for es, de in train_examples.take(1):
    print("Español: ", es.numpy().decode('utf-8'))
    print("Alemán:   ", de.numpy().decode('utf-8'))


# Carga los tokenizadores para español-inglés previamente entrenados
tokenizers = tf.saved_model.load('ted_hrlr_translate_de_es_converter')
[item for item in dir(tokenizers.de) if not item.startswith('_')]

# Analiza la longitud de las secuencias de tokens en el dataset
lengths = []
for de_examples, de_examples in train_examples.batch(1024):
    es_tokens = tokenizers.es.tokenize(de_examples)
    lengths.append(es_tokens.row_lengths())
    
    de_tokens = tokenizers.de.tokenize(de_examples)
    lengths.append(de_tokens.row_lengths())
    print('.', end='', flush=True)


# Define el número máximo de tokens que el modelo procesará
MAX_TOKENS = 128

# Función para preparar los lotes
def prepare_batch(es, de):
    es = tokenizers.es.tokenize(es)  # Tokeniza las frases en español
    es = es[:, :MAX_TOKENS]         # Recorta al máximo permitido
    es = es.to_tensor()             # Convierte a tensor denso

    de = tokenizers.de.tokenize(de)  # Tokeniza las frases en inglés
    de = de[:, :(MAX_TOKENS + 1)]
    de_inputs = de[:, :-1].to_tensor()  # Elimina el token [END]
    de_labels = de[:, 1:].to_tensor()   # Elimina el token [START]

    return (es, de_inputs), de_labels

# Configuración de los parámetros del dataset
BUFFER_SIZE = 20000
BATCH_SIZE = 64

# Función para crear lotes procesables por el modelo
def make_batches(ds):
    return (
        ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .map(prepare_batch, tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

# Prepara los lotes de entrenamiento y validación
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

# Función para generar codificaciones posicionales
def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # Vector de posiciones
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    # Codificación posicional con senos y cosenos
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

# Clase para incrustaciones posicionales
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Escala las incrustaciones
        x = x + self.pos_encoding[tf.newaxis, :length, :]  # Suma las codificaciones posicionales
        return x

# Instancia de incrustaciones para español e inglés
embed_es = PositionalEmbedding(vocab_size=tokenizers.es.get_vocab_size().numpy(), d_model=512)
embed_de = PositionalEmbedding(vocab_size=tokenizers.de.get_vocab_size().numpy(), d_model=512)

# Clase base para atención
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

# Atención cruzada para el decodificador
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        self.last_attn_scores = attn_scores  # Guarda las puntuaciones de atención
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

# Atención global para el codificador
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

# Atención causal para secuencias autoregresivas
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

# Red feed-forward para procesamiento adicional
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

# Crea el codificador del modelo Transformer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

# Ensambla el codificador
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x

# Implementa el decodificador con capas de atención y feed-forward
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, dff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x

# Construye el decodificador completo
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, context)
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x

# Define el modelo Transformer completo
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)
        try:
            del logits._keras_mask  # Corrige errores de escalado en pérdidas/métricas
        except AttributeError:
            pass
        return logits

# Hiperparámetros del Transformer
num_layers = 6
d_model = 206
dff = 103
num_heads = 8
dropout_rate = 0.1

# Crea una instancia del Transformer
transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.es.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.de.get_vocab_size().numpy(),
    dropout_rate=dropout_rate
)

# Define un programador de tasa de aprendizaje personalizada
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# Configura el optimizador con la tasa de aprendizaje personalizada
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# Función de pérdida con máscara
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )
    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

# Métrica de precisión enmascarada
def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    mask = label != 0
    match = match & mask
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match) / tf.reduce_sum(mask)

# Compila el Transformer
transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy]
)

# Entrena el modelo
transformer.fit(
    train_batches,
    epochs=25,
    validation_data=val_batches
)

# Clase para traducción
class Translator(tf.Module):
    def __init__(self, tokenizers, transformer):
        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        sentence = self.tokenizers.es.tokenize(sentence).to_tensor()
        encoder_input = sentence

        start_end = self.tokenizers.de.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]  # Obtén el último token
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        text = tokenizers.de.detokenize(output)[0]
        tokens = tokenizers.de.lookup(output)[0]
        self.transformer([encoder_input, output[:, :-1]], training=False)
        attention_weights = self.transformer.decoder.last_attn_scores
        return text, tokens, attention_weights

# Instancia el traductor
translator = Translator(tokenizers, transformer)

# Clase para exportar el modelo de traducción
class ExportTranslator(tf.Module):
    def __init__(self, translator):
        self.translator = translator

    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def __call__(self, sentence):
        result, tokens, attention_weights = self.translator(sentence, max_length=MAX_TOKENS)
        return result

# Exporta el modelo entrenado
translator = ExportTranslator(translator)
tf.saved_model.save(translator, export_dir='translatorES-DEhalf')