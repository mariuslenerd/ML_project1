import data_preprocessing
import numpy as np

def deal_with_frequencies(data, data_annotated):
    """
    data:            shape (N_samples, N_features)
    data_annotated:  shape (N_features, M_cols_at_least_11)
                     column 10 = frequency code for that feature

    returns: modified copy of data with per-column frequency treatment
    """
    # copy so we don't overwrite the original unless you want in-place
    data_out = data.copy()

    # 1. Get frequency type per feature (as int)
    freq_types = data_annotated[:, 10].astype(int)

    # 2. Define what to do for each freq code

    def transform_type1(col_values):
        """
        Convert weekly/daily/monthly encoded consumption into a unified monthly scale.

        Rules:
        - 101–199: daily   -> (x - 100) * 30
        - 201–299: weekly  -> (x - 200) / 7 * 30
        - 300: 0
        - 301–399: monthly -> x - 300
        - 555: 0

        Any other values (including NaN) are left unchanged.
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy()  # start from the original → unchanged values remain as they are

        # define masks
        mask_daily = (x >= 101) & (x <= 199)
        mask_weekly = (x >= 201) & (x <= 299)
        mask_monthly = (x >= 301) & (x <= 399)
        mask_zero = (x == 300) | (x == 555)

        # apply transformations in-place
        result[mask_daily] = (x[mask_daily] - 100) * 30
        result[mask_weekly] = (x[mask_weekly] - 200) / 7 * 30
        result[mask_monthly] = x[mask_monthly] - 300
        result[mask_zero] = 0

        return result


    def transform_type2(col_values, start=12, end=4328, replacement=5):
        """
        Convert multiple responses to a single response.

        Rules:
        - All values in [start, end] are replaced by `replacement`.

        Returns values between 0 and 5 (by convention for replacement=5).
        NaNs and unrelated values are preserved (left unchanged).

        Args:
            x (np.array): data
            col_idx (int): column index (not used directly)
            start (int): start of range (inclusive)
            end (int): end of range (inclusive)
            replacement (int): replacement value
        
        Returns:
            np.array: data with the converted column
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy()  # keep original values for untouched entries

        # define mask for replacement range
        mask_replace = (x >= start) & (x <= end)

        # apply replacement
        result[mask_replace] = replacement

        return result


    def transform_type3(col_values):
        """
        Convert weekly consumption to monthly consumption.
        NaNs and unrelated values are preserved (left unchanged).

        Rules:
        - 101–199: weekly  -> (x - 100) / 7 * 30
        - 201–299: monthly -> x - 200
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy()  # keep all original values unless replaced

        # define masks
        mask_weekly = (x >= 101) & (x <= 199)
        mask_monthly = (x >= 201) & (x <= 299)

        # apply transformations
        result[mask_weekly] = (x[mask_weekly] - 100) / 7 * 30
        result[mask_monthly] = x[mask_monthly] - 200
        
        return result


    def transform_type4(col_values):
        """
        Convert weekly/daily/monthly/yearly encoded consumption into a unified monthly scale.

        Rules:
        - 101–199: Daily  -> (x - 100) * 30
        - 201–299: Weekly -> (x - 200) / 7 * 30
        - 301–399: Monthly -> x - 300
        - 401–499: Yearly -> (x - 400) / 12
        - 888: Zero

        Returns values between 0 and 99*30 (2970).
        NaNs and unrelated values are preserved (left unchanged).
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy() 

        # define masks
        mask_daily = (x >= 101) & (x <= 199)
        mask_weekly = (x >= 201) & (x <= 299)
        mask_monthly = (x >= 301) & (x <= 399)
        mask_yearly = (x >= 401) & (x <= 499)
        mask_zero = (x == 888)

        # apply transformations
        result[mask_daily] = (x[mask_daily] - 100) * 30
        result[mask_weekly] = (x[mask_weekly] - 200) / 7 * 30
        result[mask_monthly] = x[mask_monthly] - 300
        result[mask_yearly] = (x[mask_yearly] - 400) / 12
        result[mask_zero] = 0
        
        return result

    

    def transform_type5(col_values):
        """
        Convert hours and minutes to total minutes.

        Rules:
        - 1–759:   Hours and minutes → (hours * 60 + minutes)
        - 777:     Refused → NaN
        - 800–959: Hours and minutes → (hours * 60 + minutes)

        Returns values between 0 and 1440 (i.e. 24 hours).
        NaNs and unrelated values are preserved (left unchanged).

        Args:
            x (np.array): data
            col_idx (int): column index (not used directly)
        
        Returns:
            np.array: data with the converted column
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy()  

        # define masks
        mask_1 = (x >= 1) & (x <= 759)
        mask_2 = (x >= 800) & (x <= 959)
        mask_refused = (x == 777)

        # helper: convert hhmm format to total minutes
        def hhmm_to_minutes(values):
            hours = np.floor(values / 100)
            minutes = values % 100
            return hours * 60 + minutes

        # apply conversions
        result[mask_1] = hhmm_to_minutes(x[mask_1])
        result[mask_2] = hhmm_to_minutes(x[mask_2])
        result[mask_refused] = np.nan

        return result


    def transform_type6(col_values):
        """
        Convert MET values.

        Rules:
        - 0: keep as is
        - 1–128: MET value → divide by 10

        Returns values between 0 and 12.8.
        NaNs and unrelated values are preserved (left unchanged).

        Args:
            x (np.array): data
            col_idx (int): column index (not used directly)
        
        Returns:
            np.array: data with the converted column
        """
        x = np.asarray(col_values, dtype=float)
        result = x.copy()  # keep all values unless transformed

        # define mask for 1–128
        mask_met = (x >= 1) & (x <= 128)

        # apply transformation
        result[mask_met] = x[mask_met] / 10.0

        # note: values == 0 and NaNs remain unchanged
        return result

    

    freq_map = {
        1: transform_type1,
        2: transform_type2,
        3: transform_type3,
        4: transform_type4,
        5: transform_type5,
        6: transform_type6,
        
    }

    # 3. Go through each feature/column j in data
    n_features = data_out.shape[1]
    for j in range(n_features):
        code = freq_types[j]

        # code == 0 → do nothing
        func = freq_map.get(code, None)

        if func is not None:
            # apply transformation column-wise
            data_out[:, j] = func(data_out[:, j])

    return data_out
