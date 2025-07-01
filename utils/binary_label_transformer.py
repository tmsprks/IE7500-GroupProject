class BinaryLabelTransformer:
    def __init__(self):
        # Default mapping: 1 -> 0 (negative), 2 -> 1 (positive)
        self.mapping = {1: 0, 2: 1}
    
    def transform(self, label):
        """
        Transform the input label based on the mapping.
        
        Args:
            label (int): Original label (1 or 2)
        
        Returns:
            int: Transformed label (e.g., 0 or 1)
        
        Raises:
            ValueError: If label is not in the mapping
        """
        if label not in self.mapping:
            raise ValueError(f"Label {label} not found in mapping {self.mapping}")
        return self.mapping[label]