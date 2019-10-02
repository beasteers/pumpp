import numpy as np

from ..base import FeatureExtractor, Scope


class VGGish(FeatureExtractor):
    '''
    Compute VGGish features.

    Attributes
    ----------
    name : str or None
        naming scope for this feature extractor

    compress : bool
        Whether to apply post-processing (PCA)

    **kw :
        pass to pumpp.feature.base.FeatureExtractor

    Examples
    --------

    >>> pump = pumpp.Pump(VGGish())
    >>> pump.transform(filename)
    {'vggish/ts': ..., 'vggish/emb': ...}

    '''
    def __init__(self, name='vggish', duration=None, hop_duration=None,
                 include_top=True, compress=True, **kw):
        import vggish_keras as vgk

        # setup model
        self.model, self.pump, self.sampler = vgk.get_embedding_model(
            duration=duration, hop_duration=hop_duration,
            include_top=include_top, compress=compress,
        )
        op = self.pump['mel']

        super().__init__(name, sr=op.sr, hop_length=op.hop_length, conv=None, **kw)

        # register outputs
        Scope.register(self, 'ts', (None,), np.float_)
        Scope.register(self, 'emb', self.model.output_shape, np.float_)


    def transform(self, y, sr):
        import vggish_keras as vgk
        ts, z = vgk.get_embeddings(
            y=y, sr=sr, model=self.model, pump=self.pump, sampler=self.sampler)
        return {self.scope('ts'): ts, self.scope('emb'): z}
