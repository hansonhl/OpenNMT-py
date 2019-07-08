# The following is some pseudocode for how the translator in the OpenNMT module works

def translate_batch(batch, ...):
    # batch: torchtext.data.batch.Batch object --
    # TODO: figure out more about structure of Batch object

    # define some variables for convenience
    B = batch size
    L = maximum length in batch of input text
    b = beam size
    d = word embedding dimension and hidden state dimension
    V = vocab size

    # ENCODING
    # these names are identical to those used in the actual _translate_batch() function
    # currently I'm using the transformer model as encoder

    src, enc_states, memory_bank, src_lengths = encode(batch)

    # src: input word ids, shape [L, B, 1]
    # enc_states: word embeddings for every word in batch, shape [L, B, d]
    # memory_bank: hidden states of the transformer, shape [L, B, d]
    # src_lengths: a list of lengths for each member in the batch

    # < ... some initialization setup ... >

    ## DECODING AND GENERATION
    for step in range(L):

        decoder_input = beam.current_predictions.view(1, -1, 1)

        # the beam object keeps track of all top b predictions made in every
        # time step for each batch element in a tensor called `beam.alive_seq`.
        # current_predictions is the last column in `beam.alive_seq`

        # decoder_input has shape [1, B*b, 1], i.e. for each batch element, get
        # top b possible predicted outputs in the last time step, and concatenate
        # all of them together

        log_probs, attn = decode_and_generate(decoder_input, memory_bank, batch, step, ...)

        # log probs: for each element in decoder_input, what is the log prob of
        #     generating each word in vocab as next word, shape: [B*b, V]
        # attn: for each element in decoder_input, the attention over each word
        #     in the source text, shape: [1, B*b, L]

        beam.advance(log_probs, attn)

        # given log_probs, predict top b elements for the current time step.
        # `beam.advance()` updates `beam.alive_seq`

        # <... some additional cleanup to deal with unequal lengths in batch ...>
    # endfor

    predictions = beam.predictions
    # `beam.predictions` are obtained from `beam.alive_seq`
    return predictions

# I propose the following method for incorporating RSA into the original
# translator framework:
def RSA_translate_batch(batch, distractor_generator, ...):

    aug_batch = distractor_generator(batch)
    # aug_batch contains the target text together with distractor text
    # TODO: find a way to operate on the torchtext.data.batch.Batch object. Need
    #   to ensure that aug_batch can be fed into the encoder normally, so that
    #   the framework treats distractors as normal match elements

    # NOTE: we're assuming here that distractor generation does not need
    #   information like attention, which is only later obtained by calling the
    #   decoder on the target sentences. To do that we may need an additional
    #   decoding step before using the distractor generator

    # TODO: one idea is to just use other elements in the batch as distractors, then
    #   we don't need this step

    # define some variables for convenience
    B = batch size
    L = maximum length in aug_batch of input text
    # TODO: modify maximum_length after generating distractors
    b = beam size
    d = word embedding dimension and hidden state dimension
    V = vocab size
    D = number of distractors plus target # new

    # ENCODING
    src, enc_states, memory_bank, src_lengths = encode(batch)
    # src: input word ids, shape [L, B*D, 1]
    # enc_states: word embeddings for every word in batch, shape [L, B*D, d]
    # memory_bank: hidden states of the transformer, shape [L, B*D, d]
    # src_lengths: a list of lengths for each member in the batch

    # < ... some initialization setup ... >

    ## DECODING AND GENERATION
    for step in range(L):
        decoder_input = beam.current_predictions.view(1, -1, 1)
        # Here we let the beam search object only keep track of top b predictions
        # for the *target* texts, hence `decoder_input` should have shape
        # [1, B*b, 1]

        aug_decoder_input = augment(decoder_input)
        # Duplicate `decoder_input` to shape [1, B*D*b, 1], so that probs given
        # for each distractor are based on the same unfinished sequence as the
        # corresponding target

        log_probs, attn = decode_and_generate(aug_decoder_input, memory_bank, batch, step, ...)

        # log_probs: for each element in decoder_input, what is the log prob of
        #     generating each word in vocab as next word, shape: [B*D*b, V]
        # attn: for each element in decoder_input, the attention over each word
        #     in the source text, shape: [1, B*D*b, L]

        ## RSA reasoning
        S0_log_probs = reshape(log_probs)
        # [B*D*b, V] --> [B*b, D, V], S_0(wd | w , c)
        L1_log_probs = normalize_in_log_space(transpose(S0_log_probs))
        # L1_log_probs has shape [B*b, V, D], L1(w | wd, c)
        S1_log_probs = normalize_in_log_space(alpha * transpose(L1_log_probs) + S0_log_probs)
        # I used this equation from the "Lost in Machine Translation" paper
        # S1_log_probs has shape [B*b, D, V], S1(wd | w, c)

        target_log_probs = extract_target(S1_log_probs)
        # extract probabiliy for target text, i.e. S_1(wd | w_target , c)
        # target_log_probs has shape [B*b, V]

        beam.advance(log_probs, attn)

        # given log_probs, predict top b elements for the current time step.
        # `beam.advance()` updates `beam.alive_seq`

        # <... some additional cleanup to deal with unequal lengths in batch ...>
    # endfor

    predictions = beam.predictions
    # `beam.predictions` are obtained from `beam.alive_seq`
    return predictions
