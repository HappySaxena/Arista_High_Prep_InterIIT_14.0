import statistics


class DistanceBuffer:
    """
    DistanceBuffer stores the most recent N distance values and offers
    variance/mean/std calculations and an "adjusted distance" output
    based on RSSI trend (increase/decrease).

    Features added/changed as requested:
    1. Stores last `max_size` distances (default 100).
    2. Calculates variance and mean starting from the *second* entry
       (i.e. statistics are computed over buffer[1:current]).
    3. If variance crosses `min_variance_threshold`, the buffer clears
       immediately and the method signals a reset.
    4. If no reset happens, and an RSSI value is provided, the method
       compares the RSSI with the previous RSSI sample. If RSSI has
       increased (meaning distance should have decreased), the returned
       adjusted distance = mean - std. If RSSI has decreased, the
       adjusted distance = mean + std.

    Note: the buffer stores only distances as requested. RSSI values
    are only used for trend detection and comparison and are not stored
    in the main distance FIFO. The most-recent RSSI value is kept in
    `self._prev_rssi` to allow comparison with the incoming RSSI.
    """

    def __init__(self, max_size=100, min_variance_threshold=5.0):
        """
        Initialize the DistanceBuffer.

        Parameters
        ----------
        max_size : int
            Maximum number of distances to store (FIFO). Default 100.

        min_variance_threshold : float
            Minimum variance threshold; if computed variance > this value
            the buffer will be cleared immediately.
        """
        self.max_size = int(max_size)
        self.min_variance_threshold = float(min_variance_threshold)

        # internal FIFO buffer storing distances (floats)
        self._buffer = []

        # store the last RSSI seen (float, in dBm or power units that you
        # choose to compare consistently). None until first RSSI provided.
        self._prev_rssi = None

    def add_distance(self, distance, rssi=None):
        """
        Add a new distance sample and optionally its corresponding RSSI.

        Parameters
        ----------
        distance : float
            The newly measured distance (units: meters or as you use).

        rssi : float or None
            Optional: the RSSI reading corresponding to this distance.
            RSSI must be given in the same unit every time (e.g. dBm or
            linear power). If provided, the method will compare it to
            the previous RSSI to determine whether RSSI increased or
            decreased and will compute an adjusted distance accordingly.

        Returns
        -------
        result : dict
            A dictionary containing the following keys:
                - 'variance' : float or None
                  The variance computed over buffer[1:] (None if undefined).

                - 'mean' : float or None
                  Mean computed over buffer[1:] (None if undefined).

                - 'std' : float or None
                  Standard deviation computed over buffer[1:] (None if undefined).

                - 'adjusted_distance' : float or None
                  The adjusted distance according to RSSI trend (mean +/- std)
                  (None if RSSI not provided, or insufficient samples,
                  or if a reset occurred).

                - 'reset' : bool
                  True if buffer was cleared because variance > threshold.

                - 'buffer_length' : int
                  Current number of samples stored AFTER processing.
        """

        reset_flag = False

        # -------------------------
        # 1) Append distance and maintain FIFO size
        # -------------------------
        self._buffer.append(float(distance))
        if len(self._buffer) > self.max_size:
            # drop the oldest
            self._buffer.pop(0)

        # -------------------------
        # 2) Compute stats starting from the *second* entry
        #    i.e. operate on buffer[1:]
        # -------------------------
        buf = self._buffer
        stats_slice = buf[1:] if len(buf) >= 2 else []

        variance = None
        mean = None
        std = None

        if len(stats_slice) >= 2:
            # Use sample variance (statistics.variance) as before
            variance = statistics.variance(stats_slice)
            mean = statistics.mean(stats_slice)
            std = statistics.stdev(stats_slice)
        elif len(stats_slice) == 1:
            # With a single value variance is undefined; mean exists
            variance = None
            mean = float(stats_slice[0])
            std = 0.0

        # -------------------------
        # 3) If variance crosses minimum threshold => reset buffer
        # -------------------------
        if variance is not None and variance > self.min_variance_threshold:
            # clear entire buffer
            self._buffer.clear()
            reset_flag = True

            # Update prev rssi so future comparisons don't mistakenly
            # compare against a stale value. We'll set it to the current
            # rssi if provided, otherwise None.
            self._prev_rssi = float(rssi) if rssi is not None else None

            return {
                'variance': variance,
                'mean': mean,
                'std': std,
                'adjusted_distance': None,
                'reset': True,
                'buffer_length': len(self._buffer),
            }

        # -------------------------
        # 4) If no reset happened, compute adjusted distance based on RSSI trend
        # -------------------------
        adjusted_distance = None

        if rssi is not None and mean is not None and std is not None:
            # If there is a previous RSSI to compare with
            if self._prev_rssi is not None:
                # If RSSI increased (rssi > prev) -> distance decreased
                if rssi > self._prev_rssi:
                    adjusted_distance = mean - std
                # If RSSI decreased (rssi < prev) -> distance increased
                elif rssi < self._prev_rssi:
                    adjusted_distance = mean + std
                else:
                    # No change in RSSI: adjusted distance equals mean
                    adjusted_distance = mean
            else:
                # No previous RSSI to compare to: cannot determine trend
                adjusted_distance = None

        # -------------------------
        # 5) Update previous RSSI for next comparison
        # -------------------------
        if rssi is not None:
            self._prev_rssi = float(rssi)

        # -------------------------
        # 6) Return a detailed dictionary
        # -------------------------
        return {
            'variance': variance,
            'mean': mean,
            'std': std,
            'adjusted_distance': adjusted_distance,
            'reset': reset_flag,
            'buffer_length': len(self._buffer),
        }

    # ------------------------------------------------------------------
    # Utility / helper methods
    # ------------------------------------------------------------------
    def get_buffer(self):
        """Return a shallow copy of the stored distance buffer."""
        return list(self._buffer)

    def reset(self):
        """Manually clear the buffer and stored RSSI history."""
        self._buffer.clear()
        self._prev_rssi = None

    def is_full(self):
        """Return True if the buffer is filled to max_size."""
        return len(self._buffer) >= self.max_size
