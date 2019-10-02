from ..base import FeatureExtractor, Scope


class OpenL3(FeatureExtractor):
    '''
    Compute OpenL3 features.

    Attributes
    ----------
    name : str or None
        naming scope for this feature extractor

    ...

    **kw :
        pass to pumpp.feature.base.FeatureExtractor

    Examples
    --------

    >>> pump = pumpp.Pump(OpenL3())
    >>> pump.transform(filename)
    {'openl3/ts': ..., 'openl3/emb': ...}

    '''
    def __init__(self, name='openl3',
                 input_repr='mel128',
                 content_type='env',
                 embedding_size=128,
                 hop_size=0.1,
                 **kw):
        import openl3

        self.model = openl3.models.load_embedding_model(
            input_repr, content_type, embedding_size)

        Scope.register(self, 'ts', (None,), float)
        Scope.register(self, 'emb', (None, embedding_size), float)

        sr = openl3.core.TARGET_SR
        hop_length = int(hop_size * sr)
        super().__init__(name, sr=sr, hop_length=hop_length, **kw)

    def transform(self, y, sr):
        import openl3
        emb, ts = openl3.get_embedding(
            y, sr, model=self.model,
            hop_size=self.hop_length / self.sr)
        return {'ts': ts, 'emb': emb}
