import numpy as np
import torch

class DataExportMixin:
    """Mixin class for exporting validation metrics or prediction results in various formats.

    This class provides utilities to export performance metrics (e.g., mAP, precision, recall) or prediction results
    from classification, object detection, segmentation, or pose estimation tasks into various formats: Polars
    DataFrame, CSV, and JSON.

    Methods:
        to_df: Convert summary to a Polars DataFrame.
        to_csv: Export results as a CSV string.
        to_json: Export results as a JSON string.
        tojson: Deprecated alias for `to_json()`.

    Examples:
        >>> model = YOLO("yolo26n.pt")
        >>> results = model("image.jpg")
        >>> df = results.to_df()
        >>> print(df)
        >>> csv_data = results.to_csv()
    """

    def to_df(self, normalize=False, decimals=5):
        """Create a Polars DataFrame from the prediction results summary or validation metrics.

        Args:
            normalize (bool, optional): Normalize numerical values for easier comparison.
            decimals (int, optional): Decimal places to round floats.

        Returns:
            (polars.DataFrame): Polars DataFrame containing the summary data.
        """
        import polars as pl  # scope for faster 'import ultralytics'

        return pl.DataFrame(self.summary(normalize=normalize, decimals=decimals))

    def to_csv(self, normalize=False, decimals=5):
        """Export results or metrics to CSV string format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): CSV content as string.
        """
        import polars as pl

        df = self.to_df(normalize=normalize, decimals=decimals)

        try:
            return df.write_csv()
        except Exception:
            # Minimal string conversion for any remaining complex types
            def _to_str_simple(v):
                if v is None:
                    return ""
                elif isinstance(v, (dict, list, tuple, set)):
                    return repr(v)
                else:
                    return str(v)

            df_str = df.select(
                [pl.col(c).map_elements(_to_str_simple, return_dtype=pl.String).alias(c) for c in df.columns]
            )
            return df_str.write_csv()

    def to_json(self, normalize=False, decimals=5):
        """Export results to JSON format.

        Args:
            normalize (bool, optional): Normalize numeric values.
            decimals (int, optional): Decimal precision.

        Returns:
            (str): JSON-formatted string of the results.
        """
        return self.to_df(normalize=normalize, decimals=decimals).write_json()


class SimpleClass:
    """A simple base class for creating objects with string representations of their attributes.

    This class provides a foundation for creating objects that can be easily printed or represented as strings, showing
    all their non-callable attributes. It's useful for debugging and introspection of object states.

    Methods:
        __str__: Return a human-readable string representation of the object.
        __repr__: Return a machine-readable string representation of the object.
        __getattr__: Provide a custom attribute access error message with helpful information.

    Examples:
        >>> class MyClass(SimpleClass):
        ...     def __init__(self):
        ...         self.x = 10
        ...         self.y = "hello"
        >>> obj = MyClass()
        >>> text = str(obj)  # "<module>.MyClass object with attributes:" followed by "x: 10" and "y: 'hello'"

    Notes:
        - This class is designed to be subclassed. It provides a convenient way to inspect object attributes.
        - The string representation includes the module and class name of the object.
        - Callable attributes and attributes starting with an underscore are excluded from the string representation.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {v!r}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Provide a custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")