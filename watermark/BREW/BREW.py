from sage.all import *
from sage.all import GF, codes, vector
import torch
from math import sqrt
from functools import partial
from ..base import BaseWatermark, BaseConfig
from utils.utils import load_config_file
from utils.transformers_config import TransformersConfig
from transformers import LogitsProcessor, LogitsProcessorList
import random

class BREWConfig(BaseConfig):

    def initialize_parameters(self) -> None:
        self.gamma = self.config_dict['gamma']
        self.delta = self.config_dict['delta']
        self.hash_key = self.config_dict['hash_key']
        self.z_threshold = self.config_dict['z_threshold']
        self.prefix_length = self.config_dict['prefix_length']

        self.bch_t = self.config_dict['bch_t']
        self.bch_m = self.config_dict['bch_m']

        self.max_shift_bit = self.config_dict['max_shift_bit']
        self.scheme = self.config_dict['scheme']

    @property
    def algorithm_name(self) -> str:
        return 'BREW'

class BREWUtils:
    def __init__(self, config):
        self.config = config
        self.rng = torch.Generator(device=self.config.device)
        self.rng.manual_seed(self.config.hash_key)

        self.vocab_size = config.vocab_size
        self.green_mask = self._init_fixed_vocab_split()

        self.F = GF(2)
        self.n = 2**self.config.bch_m - 1
        self.d = 15
        self.C = codes.BCHCode(self.F, self.n, self.d)
        self.k = self.C.dimension()

        self.codeword_pair = self._find_distant_codeword_pair()

    def _init_fixed_vocab_split(self):
        self.allowed_token_ids = torch.arange(self.vocab_size, device=self.config.device)
        perm = torch.randperm(self.vocab_size,device=self.config.device, generator=self.rng)
        green_ids = self.allowed_token_ids[perm[: self.vocab_size // 2]]
        red_ids = self.allowed_token_ids[perm[self.vocab_size // 2:]]

        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        mask[green_ids.cpu()] = True
        self._red_ids = red_ids.tolist()
        self._green_ids = green_ids.tolist()
        return mask

    def get_greenlist_ids(self):
        return self._green_ids

    def get_redlist_ids(self):
        return self._red_ids

    def get_allowed_token_ids(self):
        return self.allowed_token_ids.tolist()

    def _encode_message(self, message_bits: list[int]) -> list[int]:
        m = vector(self.F, message_bits)
        c = self.C.encode(m)
        return list(c)

    def _find_distant_codeword_pair(self):

        max_weight = -1
        message_for_max_weight = None

        num_total_messages = 2**self.k
        for i in range(1, num_total_messages):
            msg = [int(bit) for bit in format(i, f'0{self.k}b')]

            codeword = self._encode_message(msg)

            weight = sum(int(bit) for bit in codeword)

            if weight > max_weight:
                max_weight = weight
                message_for_max_weight = msg

        codeword_max_weight = self._encode_message(message_for_max_weight)

        while True:

            random_int = random.randint(1, num_total_messages - 1)
            random_message1 = [int(bit) for bit in format(random_int, f'0{self.k}b')]

            if random_message1 != message_for_max_weight:
                break
        codeword1 = self._encode_message(random_message1)

        codeword2 = [int(b1) ^ int(b2) for b1, b2 in zip(codeword1, codeword_max_weight)]

        return [codeword1, codeword2]

    def sample_message_and_codeword(self):
        return random.choice(self.codeword_pair)

    def get_green_mask(self, device=None):
        return self.green_mask if device is None else self.green_mask.to(device)

    def tokens_to_bits(self, token_ids: torch.Tensor) -> torch.Tensor:
        vocab_sz = self.vocab_size
        gm = self.green_mask.to(token_ids.device)
        valid = token_ids < vocab_sz

        bits = torch.zeros_like(token_ids, dtype=torch.int8)
        bits[valid] = gm[token_ids[valid]].to(torch.int8)
        return bits

class BREWLogitsProcessor(LogitsProcessor):
    def __init__(self, config, utils):
        self.config = config
        self.utils = utils
        self.codeword_queue = []
        self.token_bit_log = []

    def _get_codeword_bit(self, position: int) -> int:
        codeword_index = position // self.utils.n
        bit_index = position % self.utils.n
        while len(self.codeword_queue) <= codeword_index:
            new_codeword = self.utils.sample_message_and_codeword()
            self.codeword_queue.append(new_codeword)

        return self.codeword_queue[codeword_index][bit_index]

    def _get_target_token_ids(self, bit: int, input_ids: torch.LongTensor) -> list[int]:
        allowed = set(self.utils.get_allowed_token_ids())

        target_ids = self.utils.get_greenlist_ids() if bit == 1 else self.utils.get_redlist_ids()
        return list(set(target_ids) & allowed)

    def _get_bias_mask(self, scores: torch.Tensor, target_ids: list[int]) -> torch.BoolTensor:
        mask = torch.zeros_like(scores, dtype=torch.bool)
        indices = torch.tensor(target_ids, dtype=torch.long, device=scores.device)
        mask[indices] = True
        return mask

    def _bias_logits_soft(self, scores: torch.Tensor, target_mask: torch.Tensor, bias: float) -> torch.Tensor:
        scores[target_mask] += bias

        return scores

    def _bias_logits_hard(self, scores: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        non_target_mask = ~target_mask

        scores[non_target_mask] -= 10000
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.config.prefix_length:
            return scores

        for b in range(scores.shape[0]):
            token_position = len(self.token_bit_log)
            bit = self._get_codeword_bit(token_position)
            self.token_bit_log.append(bit)

            target_ids = self._get_target_token_ids(bit, input_ids[b])
            mask = self._get_bias_mask(scores[b], target_ids)

            if self.config.scheme == 'hard':
                scores[b] = self._bias_logits_hard(scores[b], mask)
            elif self.config.scheme == 'soft':
                scores[b] = self._bias_logits_soft(scores[b], mask, self.config.delta)
            else:
                raise ValueError(f"Invalid scheme: {self.config.scheme}. Choose 'soft' or 'hard'.")

        return scores

class BREW(BaseWatermark):

    def __init__(self, algorithm_config: str | BREWConfig, transformers_config: TransformersConfig | None = None, *args, **kwargs) -> None:
        if isinstance(algorithm_config, str):
            self.config = BREWConfig(algorithm_config, transformers_config)
        elif isinstance(algorithm_config, BREWConfig):
            self.config = algorithm_config
        else:
            raise TypeError("algorithm_config must be either a path string or a BREWConfig instance")

        self.utils = BREWUtils(self.config)
        self.logits_processor = BREWLogitsProcessor(self.config, self.utils)

    @staticmethod
    def cyclic_shift(bits: list[int], shift: int, direction: str = 'left') -> list[int]:
        if direction == 'left':
            return bits[shift:] + bits[:shift]
        elif direction == 'right':
            return bits[-shift:] + bits[:-shift]
        else:
            raise ValueError(f"Invalid shift direction: {direction}")

    def generate_watermarked_text(self, prompt: str, *args, **kwargs) -> str:
        generate_with_watermark = partial(
            self.config.generation_model.generate,
            logits_processor=LogitsProcessorList([self.logits_processor]),
            **self.config.gen_kwargs
        )

        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        prompt_ids = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]

        encoded_prompt = {k: v[:1] for k, v in encoded_prompt.items()}

        encoded_watermarked_text = generate_with_watermark(**encoded_prompt)
        watermarked_text = self.config.generation_tokenizer.batch_decode(encoded_watermarked_text, skip_special_tokens=True)[0]

        return watermarked_text

    def generate_unwatermarked_text(self, prompt: str, *args, **kwargs) -> str:

        generate_without_watermark = partial(
            self.config.generation_model.generate,

            **self.config.gen_kwargs
        )

        encoded_prompt = self.config.generation_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.config.device)
        encoded_prompt = {k: v[:1] for k, v in encoded_prompt.items()}

        encoded_unwatermarked_text = generate_without_watermark(**encoded_prompt)
        unwatermarked_text = self.config.generation_tokenizer.batch_decode(encoded_unwatermarked_text, skip_special_tokens=True)[0]
        return unwatermarked_text

    def detect_watermark(self, prompt: str, text: str, return_dict: bool = True, *args, **kwargs):

        tokenizer = self.config.generation_tokenizer
        device = self.config.device
        n = self.utils.n
        C = self.utils.C
        F = self.utils.F
        A = C.ambient_space()
        dec = C.decoder()
        max_shift = self.config.max_shift_bit

        detect_prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        encoded_text      = tokenizer(text,   return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

        bits_t = self.utils.tokens_to_bits(encoded_text)
        bit_stream = bits_t.tolist()

        prompt_len = detect_prompt_ids.shape[1]
        start_idx  = 0 if len(encoded_text) <= (prompt_len - 1) else (prompt_len - 1)
        bit_segments = [bit_stream[i:i+n] for i in range(start_idx, len(bit_stream)-n+1, n)]

        full_gt_list = getattr(self.logits_processor, "codeword_queue", None)
        if not full_gt_list:
            out = {"is_watermarked": False, "reason": "no_ground_truth_codewords", "matched": 0, "total": 0}
            return out if return_dict else False

        num_segments = len(bit_segments)
        gt_list = full_gt_list[:num_segments]

        def _decode_to_code_safe(bits_list):
            try:
                v = A(vector(F, bits_list))
                c_hat = dec.decode_to_code(v)
                return list(A(c_hat))
            except Exception as e:

                msg = str(e)
                if ("Decoding failed because the number of errors exceeded the decoding radius" in msg
                    or e.__class__.__name__ == "DecodingError"):
                    return None
                raise

        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        matched = 0
        match_info = []

        for i, (seg_bits, gt_bits) in enumerate(zip(bit_segments, gt_list)):

            segment_errors = hamming(seg_bits, gt_bits)

            c_hat_bits = _decode_to_code_safe(seg_bits)
            if c_hat_bits is not None and c_hat_bits == gt_bits:

                matched += 1
                match_info.append({"success": True, "dir": "none", "shift": 0, "raw_errors": segment_errors})
                continue

            success = False
            for shift in range(1, max_shift + 1):
                test_bits = BREW.cyclic_shift(seg_bits, shift, 'left')
                c_hat_bits = _decode_to_code_safe(test_bits)
                shift_errors = hamming(test_bits, gt_bits)
                if c_hat_bits is not None and c_hat_bits == gt_bits:

                    matched += 1
                    match_info.append({"success": True, "dir": "left", "shift": shift, "raw_errors": shift_errors})
                    success = True
                    break

            if not success:
                for shift in range(1, max_shift + 1):
                    test_bits = BREW.cyclic_shift(seg_bits, shift, 'right')
                    c_hat_bits = _decode_to_code_safe(test_bits)
                    shift_errors = hamming(test_bits, gt_bits)
                    if c_hat_bits is not None and c_hat_bits == gt_bits:

                        matched += 1
                        match_info.append({"success": True, "dir": "right", "shift": shift, "raw_errors": shift_errors})
                        success = True
                        break

            if not success:

                match_info.append({"success": False, "dir": None, "shift": None, "raw_errors": segment_errors})

        total = num_segments
        threshold = self.config.z_threshold * total / 100.0
        is_watermarked = (matched > threshold)

        result = {
            "is_watermarked": is_watermarked,
            "matched": matched,
            "total": total,
            "match_percent": (matched / total * 100.0) if total > 0 else 0.0,
            "match_info": match_info,
        }
        return result if return_dict else is_watermarked

    def analyze_watermark_errors(self, prompt: str, text: str, return_dict: bool = True, debug: bool = True):

        tokenizer = self.config.generation_tokenizer
        device    = self.config.device
        n         = self.utils.n
        C         = self.utils.C
        F         = self.utils.F
        A         = C.ambient_space()
        dec       = C.decoder()
        max_shift = self.config.max_shift_bit

        def _decode_to_code_safe(bits_list):
            try:
                v = A(vector(F, bits_list))

                c_hat = dec.decode_to_code(v)
                return list(A(c_hat))
            except Exception as e:
                msg = str(e)
                if ("Decoding failed" in msg or e.__class__.__name__ == "DecodingError"):
                    return None
                raise

        def hamming(a, b):
            return sum(x != y for x, y in zip(a, b))

        detect_prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        encoded_text      = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

        bits_t = self.utils.tokens_to_bits(encoded_text)
        bit_stream = bits_t.tolist()

        prompt_len = detect_prompt_ids.shape[1]
        start_idx  = 0 if len(encoded_text) <= (prompt_len - 1) else (prompt_len - 1)
        bit_segments = [bit_stream[i:i+n] for i in range(start_idx, len(bit_stream)-n+1, n)]

        full_gt_list = getattr(self.logits_processor, "codeword_queue", None)
        if not full_gt_list:
            return {"is_watermarked": False, "reason": "no_ground_truth_codewords"}

        num_segments = len(bit_segments)
        gt_list = full_gt_list[:num_segments]

        matched = 0
        total_final_errors = 0

        for i, (seg_bits, gt_bits) in enumerate(zip(bit_segments, gt_list)):
            direct_errors = hamming(seg_bits, gt_bits)

            chosen_dir = "none"
            chosen_shift = 0
            chosen_raw_errors = direct_errors
            matched_here = False

            best_left_errors, best_left_shift = None, None
            best_right_errors, best_right_shift = None, None

            c_hat_bits = _decode_to_code_safe(seg_bits)
            if c_hat_bits is not None and c_hat_bits == gt_bits:
                matched_here = True
                if debug:
                    print(f"[Compare #{i}] ✅ Direct match ({direct_errors} errors)")
            else:

                for s in range(1, max_shift + 1):
                    tbits = BREW.cyclic_shift(seg_bits, s, 'left')
                    errs  = hamming(tbits, gt_bits)
                    if best_left_errors is None or errs < best_left_errors:
                        best_left_errors, best_left_shift = errs, s

                    c_hat_bits_shifted = _decode_to_code_safe(tbits)
                    if (not matched_here) and c_hat_bits_shifted is not None and c_hat_bits_shifted == gt_bits:
                        matched_here = True
                        chosen_dir, chosen_shift, chosen_raw_errors = "left", s, errs
                        if debug:
                            print(f"[Compare #{i}] 🔄 Match with {s}-bit left shift ({errs} errors)")

                for s in range(1, max_shift + 1):
                    tbits = BREW.cyclic_shift(seg_bits, s, 'right')
                    errs  = hamming(tbits, gt_bits)
                    if best_right_errors is None or errs < best_right_errors:
                        best_right_errors, best_right_shift = errs, s

                    c_hat_bits_shifted = _decode_to_code_safe(tbits)
                    if (not matched_here) and c_hat_bits_shifted is not None and c_hat_bits_shifted == gt_bits:
                        matched_here = True
                        chosen_dir, chosen_shift, chosen_raw_errors = "right", s, errs
                        if debug:
                            print(f"[Compare #{i}] 🔄 Match with {s}-bit right shift ({errs} errors)")

            if matched_here:
                total_final_errors += chosen_raw_errors
            else:
                total_final_errors += direct_errors
                if debug:
                    print(f"[Compare #{i}] ❌ Too many errors: {direct_errors} / {n}")

            if matched_here:
                matched += 1

        total = num_segments
        total_bits = total * n if total > 0 else 1
        match_rate = matched / total if total > 0 else 0.0

        summary = {
            "is_watermarked": (matched > 0),
            "matched": matched,
            "total": total,
            "match_rate": match_rate,
            "total_final_errors": total_final_errors,
            "total_final_errors_rate": total_final_errors / total_bits
        }

        return summary

