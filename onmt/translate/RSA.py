""" RSA model utility functions """

import torch

from onmt.translate.translator import Translator

from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores

class RSATranslator(Translator):
    """Translate a batch of sentences using the RSA model. Inherits from
    :class:`Translator`

    (Currently) inherits functions :func:`__init__()`, :func:`from_opt()`,
    :func:`_log()`, :func:`_gold_score()`, :func:`translate()`,
    :func:`translate_batch()`.
    """
    def _naive_distractor():
        pass

    def _translate_batch(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        #### TODO: Augment batch with distractors

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        # src has shape [1311, 2, 1]
        # enc_states has shape [1311, 2, 512],
        # Memory_bank has shape [1311, 2, 512]
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)
        print('memory_bank size after tile:', memory_bank.shape)#[1311, 20, 512]

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            ratio=ratio,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths)

        all_log_probs = []
        all_attn = []

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)
            # decoder_input has shape[1,20,1]
            # decoder_input gives top 10 predictions for each batch element
            verbose = True if step == 10 else False
            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=beam._batch_offset,
                verbose=verbose)

            # log_probs and attn are the probs for next word given that the
            # current word is that in decoder_input
            all_log_probs.append(log_probs)
            all_attn.append(attn)

            beam.advance(log_probs, attn, verbose=verbose)

            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        print('batch_size:', batch_size)
        print('max_length:', max_length)
        print('all_log_probs has len', len(all_log_probs))
        print('all_log_probs[0].shape', all_log_probs[0].shape)
        print('comparing log_probs[0]', all_log_probs[2][:,0])

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results
