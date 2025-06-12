class ContextNamespace(dict):
    """
    A dictionary-like class that allows attribute access to its keys.
    """
    def __init__(self, args = None):
        # Initialize state parameters
        super().__init__(args or {})
    
    def _raise(self, key):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            self._raise(key)
    
    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            self._raise(key)