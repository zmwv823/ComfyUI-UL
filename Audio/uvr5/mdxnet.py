import os
import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from cuda_malloc import cuda_malloc_supported

cpu = torch.device("cpu" if not cuda_malloc_supported() else "cuda")


class ConvTDFNetTrim:
    def __init__(
        self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(ConvTDFNetTrim, self).__init__()

        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )
        self.target_name = target_name
        self.blender = "blender" in model_name

        self.dim_c = 4
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

        self.n = L // 2

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])


def get_models(device, dim_f, dim_t, n_fft):
    return ConvTDFNetTrim(
        device=device,
        model_name="Conv-TDF",
        target_name="vocals",
        L=11,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


class Predictor:
    def __init__(self, args):
        import onnxruntime as ort

        logger.info(ort.get_available_providers())
        self.args = args
        self.model_ = get_models(
            device=cpu, dim_f=args.dim_f, dim_t=args.dim_t, n_fft=args.n_fft
        )
        import folder_paths
        onnx_path = os.path.join(folder_paths.models_dir, r'audio_checkpoints\ExtraModels\uvr5\uvr5_weights\Onnx_dereverb_By_FoxJoy')
        self.model = ort.InferenceSession(
            os.path.join(onnx_path, "vocals.onnx"),
            providers=[
                "CUDAExecutionProvider",
                # "DmlExecutionProvider",
                # "TensorrtExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        logger.info("ONNX load done")

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.args.margin
        chunk_size = self.args.chunks * 44100
        assert not margin == 0, "margin cannot be zero!"
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")
        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float16).to(cpu)
            # mix_waves = torch.tensor(np.array(mix_waves), dtype=torch.float32).to(cpu)
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.args.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred).to(cpu))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0]).to(cpu)
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        progress_bar.close()
        return _sources

    def prediction(self, m, vocal_root, others_root, format):
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        basename = os.path.basename(m)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T
        if format in ["wav", "flac"]:
            vocal_AUDIO = "%s/%s_main_vocal.%s" % (vocal_root, basename, format)
            sf.write(
                vocal_AUDIO, mix - opt, rate
            )
            bgm_AUDIO = "%s/%s_others.%s" % (others_root, basename, format)
            sf.write(bgm_AUDIO, opt, rate)
        else:
            vocal_AUDIO = "%s/%s_main_vocal.wav" % (vocal_root, basename)
            bgm_AUDIO = "%s/%s_others.wav" % (others_root, basename)
            sf.write(vocal_AUDIO, mix - opt, rate)
            sf.write(bgm_AUDIO, opt, rate)
            opt_path_vocal = vocal_AUDIO[:-4] + ".%s" % format
            opt_path_other = bgm_AUDIO[:-4] + ".%s" % format
            if os.path.exists(vocal_AUDIO):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y" % (vocal_AUDIO, opt_path_vocal)
                )
                if os.path.exists(opt_path_vocal):
                    try:
                        os.remove(vocal_AUDIO)
                    except:
                        pass
            if os.path.exists(bgm_AUDIO):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y" % (bgm_AUDIO, opt_path_other)
                )
                if os.path.exists(opt_path_other):
                    try:
                        os.remove(bgm_AUDIO)
                    except:
                        pass
        return vocal_AUDIO,bgm_AUDIO


class MDXNetDereverb:
    def __init__(self, chunks):
        self.onnx = "%s/uvr5_weights/onnx_dereverb_By_FoxJoy"%os.path.dirname(os.path.abspath(__file__))
        self.shifts = 10  # 'Predict with randomised equivariant stabilisation'
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        self.chunks = chunks
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.pred = Predictor(self)
        self.device = cpu

    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        return self.pred.prediction(input, vocal_root, others_root, format)
