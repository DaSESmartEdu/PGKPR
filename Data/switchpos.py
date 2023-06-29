import pandas as pd


class SwitchPOS:
    def __init__(self, path, get_same_index=False):
        super(SwitchPOS, self).__init__()
        self.df = pd.read_csv(path)
        if 's_index' in self.df.columns:
            self.s_index = self.df.s_index
        if get_same_index:
            self.df['same_index'] = SwitchPOS.get_same_index(self.df.sen1.to_list(), self.df.sen2.to_list())

    def set_s_index(self, s_index):
        self.df['s_index'] = s_index
        self.keep_struct_by_index()

    def keep_struct_by_index(self):
        sens = self.df.sen1.to_list()
        poss = self.df.pos1.to_list()

        s_sens, s_poss = [], []
        for x, y, k in zip(sens, poss, self.df.s_index):
            xx, yy = SwitchPOS.swich_by_index(x, y, k)
            s_sens.append(xx)
            s_poss.append(yy)
        self.df['s_sen1'] = [" ".join(i) for i in s_sens]
        self.df['s_pos1'] = [" ".join(i) for i in s_poss]

    def to_csv(self, path):
        self.df.to_csv(path, index=False)

    @staticmethod
    def word_to_index(sen, kw):
        sen = sen.strip().split(" ")
        index = []
        kw = list(set(kw))
        for item in kw:
            for idx, word in enumerate(sen):
                if item == word:
                    index.append(idx)
        if not index:
            index = [-1]
        index = sorted(index)
        return " ".join([str(i) for i in index])

    @staticmethod
    def swich_by_index(sen, pos, indexs):
        sen = sen.strip().split(" ")
        pos = pos.strip().split(" ")
        assert len(sen) == len(pos)
        indexs = indexs.split(" ")
        indexs = [int(i) for i in indexs]
        for i in indexs:
            if i != -1 and i < len(sen):
                sen[i], pos[i] = pos[i], sen[i]
        return sen, pos

    @staticmethod
    def word_by_index(sen, index):
        sen = sen.strip().split(" ")
        index = index.strip().split(" ")
        index = [int(i) for i in index]
        if -1 in index:
            return []
        return [sen[i] for i in index]

    @staticmethod
    def get_same_index(sen1, sen2):
        """
        :return: index of same words of sen1 and sen2 in sen1
        """
        res = []
        for x, y in zip(sen1, sen2):
            x = x.strip().split(" ")
            y = y.strip().split(" ")
            same = list(set(x) & set(y))
            index_of_same = []
            for i in same:
                for idx, word in enumerate(x):
                    if word == i:
                        index_of_same.append(idx)
            if not index_of_same:
                index_of_same = [-1]
            index_of_same = sorted(index_of_same)
            index_of_same = " ".join([str(i) for i in index_of_same])
            res.append(index_of_same)
        return res
