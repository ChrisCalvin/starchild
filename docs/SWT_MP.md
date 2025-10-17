 Using a Scattering Wavelet Transform (SWT) before computing the Matrix Profile can offer several significant benefits, essentially transforming the domain of pattern
  discovery from raw time series values to a more robust and informative feature space.

  Here are the key advantages:

   1. Enhanced Robustness to Noise and Minor Deformations:
       * Problem with Raw Data MP: The standard Matrix Profile, when applied directly to raw time series, is highly sensitive to noise and small misalignments or
         deformations in patterns. Even minor shifts or random fluctuations can significantly increase the Euclidean distance between two otherwise similar subsequences,
         making them harder for the MP to identify as motifs.
       * Benefit of SWT: Scattering coefficients are inherently stable to small deformations (e.g., slight stretching or compression of a pattern) and robust to noise. By
         computing the Matrix Profile on these stable scattering coefficients, patterns that are conceptually similar but slightly noisy or misaligned in the raw data will
         yield much smaller distances in the scattering feature space. This makes motif discovery more reliable and less prone to being misled by irrelevant variations.

   2. Approximate Translation Invariance:
       * Problem with Raw Data MP: The Matrix Profile is not inherently translation-invariant. A pattern that appears at t=10 and an identical pattern appearing at t=12 will
          be treated as distinct subsequences by the MP, potentially increasing their distance if they are not perfectly aligned.
       * Benefit of SWT: Scattering coefficients provide approximate translation invariance. This means that if a pattern shifts slightly in time, its scattering
         representation will remain very similar. Computing the Matrix Profile on these coefficients allows it to effectively find motifs regardless of their exact starting
         point in the time series, leading to more robust and generalizable pattern detection.

   3. Capturing Multi-Scale Information More Abstractly:
       * Problem with Raw Data MP: While HierarchicalStreamingMP uses multiple window_sizes (m) to capture patterns at different scales, each individual Matrix Profile is
         still operating on the raw signal.
       * Benefit of SWT: Scattering coefficients inherently encode multi-scale and multi-frequency information about the time series. When you compute a Matrix Profile on
         these coefficients, the "subsequences" being compared are no longer just raw data points; they are rich, multi-scale representations of segments of the original
         time series. This allows the Matrix Profile to discover motifs in the structural and textural properties of the time series across different scales, rather than
         just in the raw value movements. This can lead to the discovery of more abstract and meaningful patterns (e.g., "periods of increasing volatility followed by
         stability" rather than just "this specific price movement").

   4. Potential for Dimensionality Reduction and Computational Efficiency:
       * Problem with Raw Data MP: For very long or high-dimensional raw time series, computing the Matrix Profile can be computationally expensive (due to the O(N^2)
         nature).
       * Benefit of SWT: The SWT can often provide a compact, lower-dimensional representation of the time series while preserving critical information. If the scattering
         coefficients are significantly lower-dimensional than the raw time series, computing the Matrix Profile on this reduced feature set could be substantially faster.
         This could potentially allow for the analysis of longer effective context lengths for the Matrix Profile, or enable faster processing for existing lengths.

   5. Focus on "Texture" and "Dynamics" over Raw Values:
       * Benefit of SWT: Scattering coefficients are particularly good at characterizing the "texture," local variability, and dynamic properties of a signal. By applying
         the Matrix Profile to these features, you are essentially looking for recurring patterns in these textural and dynamic characteristics, which can be more indicative
          of underlying regimes or states than simple patterns in raw price levels.

  In essence, using SWT before the Matrix Profile allows you to leverage the SWT's strengths in creating stable, multi-scale, and robust representations, and then apply
  the Matrix Profile's efficiency in finding recurring patterns within that more informative feature space. This can lead to more meaningful, robust, and potentially more
  efficient pattern discovery.
