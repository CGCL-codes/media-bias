class DrawingData:
    def __init__(self, labels, data1, data2, data1_label, data2_label, title):
        self.labels = labels
        self.data1 = data1
        self.data2 = data2
        self.data1_label = data1_label
        self.data2_label = data2_label
        self.title = title #

    def get_all_data(self):
        return self.labels, self.data1, self.data2, self.data1_label, self.data2_label, self.title

    def transform_count_to_ratio(self):
        data_1 = []
        data_2 = []
        data_len = len(self.data1)
        for index in range(0, data_len):
            data_1.append(round(float(self.data1[index] / (self.data1[index] + self.data2[index])), 2))
            data_2.append(round(float(self.data2[index] / (self.data1[index] + self.data2[index])), 2))
        self.data1 = data_1
        self.data2 = data_2
        return self
