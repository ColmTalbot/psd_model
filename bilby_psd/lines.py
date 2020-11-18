import pandas as pd


class Line(object):
    def __init__(self, frequency, comb_type, delta_frequency_of_first_harmonic,
                 index_first_harmonic, index_last_harmonic, width_left,
                 width_right):
        self.frequency = frequency
        self.comb_type = comb_type
        self.delta_frequency_of_first_harmonic = delta_frequency_of_first_harmonic
        self.index_first_harmonic = index_first_harmonic
        self.index_last_harmonic = index_last_harmonic
        self.width_left = width_left
        self.width_right = width_right

    def get_prior_range_for_nth_harmonic(self, n):
        if self.comb_type in [0, 1]:
            dfmin = - self.width_left
            dfmax = + self.width_right
        elif self.comb_type == 2:
            dfmin = - n * self.width_left
            dfmax = + n * self.width_right

        fmin = self.delta_frequency_of_first_harmonic + (n + 1) * self.frequency + dfmin
        fmax = self.delta_frequency_of_first_harmonic + (n + 1) * self.frequency + dfmax
        return fmin, fmax

    def get_harmonics(self):
        harmonics = [0]
        if self.comb_type > 0:
            harmonics += range(int(self.index_first_harmonic),
                               int(self.index_last_harmonic + 1))
        return harmonics


class LineList(object):
    def __init__(self, file, minimum_frequency, maximum_frequency):
        self.line_list = []
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        names = [
            "frequency", "comb_type", "delta_frequency_of_first_harmonic",
            "index_first_harmonic", "index_last_harmonic", "width_left",
            "width_right"
        ]
        df = pd.read_csv(file, comment="%", delim_whitespace=True, names=names)
        for ii, row in df.iterrows():
            self.line_list.append(Line(**dict(row)))

    def get_fmin_fmax_list(self):
        fmin_fmax_list = []
        for line in self.line_list:
            for n in line.get_harmonics():
                fmin, fmax = line.get_prior_range_for_nth_harmonic(n)
                if fmin > self.minimum_frequency and fmax < self.maximum_frequency:
                    fmin_fmax_list.append((fmin, fmax))

        return fmin_fmax_list
