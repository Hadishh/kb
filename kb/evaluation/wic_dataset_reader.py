from typing import Iterable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField
from allennlp.data.instance import Instance
from kb.bert_tokenizer_and_candidate_generator import TokenizerAndCandidateGenerator
from allennlp.common.file_utils import cached_path


@DatasetReader.register("wic")
class WicDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer_and_candidate_generator: TokenizerAndCandidateGenerator,
                 entity_markers: bool = False):
        super().__init__()
        self.label_to_index = {'T': 1, 'F': 0}
        self.tokenizer = tokenizer_and_candidate_generator
        self.tokenizer.whitespace_tokenize = True
        self.entity_markers = entity_markers

    def text_to_instance(self, line) -> Instance:
        raise NotImplementedError

    def _read(self, file_path: str) -> Iterable[Instance]:
        """Creates examples for the training and dev sets."""

        # with open(cached_path(file_path + '_labels.txt'), 'r') as f:
        #     labels = f.read().split()

        with open(cached_path(file_path + '.data.extra.txt'), 'r') as f:
            sentences = f.read().splitlines()
            # assert len(labels) == len(sentences), f'The length of the labels and sentences must match. ' \
            #     f'Got {len(labels)} and {len(sentences)}.'

            for line in sentences:
                tokens = line.split('\t')
                assert len(tokens) == 5, tokens

                context = tokens[0]
                def_pred = tokens[1]
                def_gold = tokens[2]
                idx = int(tokens[3])
                label = int(tokens[4])

                if self.entity_markers:
                    # insert entity markers
                    tokens_context = context.strip().split()

                    # def_a = f"in that sentence, {context[idx1]} is defined as {def_pred}"
                    # def_b = f"in that sentence, {tokens_b[idx2]} is defined as {def_b}"
                    tokens_context.insert(idx, '[e1start]')
                    tokens_context.insert(idx + 2, '[e1end]')

                    context = ' '.join(tokens_context)

                    # text_b = ' '.join(tokens_b)
                # print(text_a, def_a, text_b, def_b)
                token_candidates = self.tokenizer.tokenize_and_generate_candidates(context, def_gold, def_pred)
                fields = self.tokenizer.convert_tokens_candidates_to_fields(token_candidates)
                fields['label_ids'] = LabelField(self.label_to_index[label], skip_indexing=True)

                # get the indices of the marked words
                # index in the original tokens
                # idx1, idx2 = [int(ind) for ind in tokens[2].split('-')]
                offsets_c = [1] + token_candidates['offsets_a'][:-1]
                idx_offset = offsets_c[idx]
                # offsets_b = [token_candidates['offsets_defa'][-1] + 1] + token_candidates['offsets_b'][:-1]
                # idx2_offset = offsets_b[idx2]

                fields['index_a'] = LabelField(idx_offset, skip_indexing=True)
                # fields['index_b'] = LabelField(idx2_offset, skip_indexing=True)

                instance = Instance(fields)

                yield instance
